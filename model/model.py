import json
import logging
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Union
import time

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import AutoencoderKL, DDIMScheduler
from .modules.zero_snr_ddpm import DDPMScheduler ###FIXME changed zero-SNR
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from torch.optim import AdamW
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor

from components.vae import TemporalAutoencoderKL
from data.utils import load_image
from utils import dist_envs
from utils import slugify
from .modules.unet import UNet3DConditionModel
from .pipeline import SDVideoPipeline
from .utils import save_videos_grid, compute_clip_score, prepare_masked_latents, prepare_entity_latents, FrozenOpenCLIPImageEmbedderV2, Resampler

logger = logging.getLogger(__name__)


class SDVideoModel(pl.LightningModule):
    def __init__(self, pretrained_model_path, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained_model_path"], logger=False)
        # main training module
        self.unet: Union[str, UNet3DConditionModel] = Path(pretrained_model_path, "unet").as_posix()
        # components for training
        self.noise_scheduler_dir = Path(pretrained_model_path, "scheduler").as_posix()
        self.vae = Path(pretrained_model_path, "vae").as_posix()
        self.text_encoder = Path(pretrained_model_path, "text_encoder").as_posix()
        self.tokenizer: Union[str, CLIPTokenizer] = Path(pretrained_model_path, "tokenizer").as_posix()
        # clip model for metric
        self.clip = Path(pretrained_model_path, "clip").as_posix()
        self.clip_processor = Path(pretrained_model_path, "clip").as_posix()
        # define pipeline for inference
        self.val_pipeline = None
        # video frame resolution
        self.resolution = kwargs.get("resolution", 512)
        # use temporal_vae
        self.temporal_vae_path = kwargs.get("temporal_vae_path", None)
        # use prompt image
        self.in_channels = kwargs.get("in_channels", 4)
        self.use_prompt_image = self.in_channels > 4
        self.add_entity_vae = kwargs.get("add_entity_vae", False)
        self.add_entity_clip = kwargs.get("add_entity_clip", False)

        ### add open clip model
        if self.add_entity_clip:
            self.embedding_dim = 1280
            self.entity_clip_model = FrozenOpenCLIPImageEmbedderV2(arch="ViT-H-14") ###FIXME
            self.enclip_projector = Resampler(dim=1024, depth=4, dim_head=64, heads=12, num_queries=16, embedding_dim=self.embedding_dim, output_dim=1024, ff_mult=4)
        else:
            self.entity_clip_model = None
            self.enclip_projector = None
    
    def setup(self, stage: str) -> None:
        # build modules
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.noise_scheduler_dir)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.tokenizer)

        if self.temporal_vae_path:
            self.vae = TemporalAutoencoderKL.from_pretrained(self.temporal_vae_path)
        else:
            self.vae = AutoencoderKL.from_pretrained(self.vae)
        self.text_encoder = CLIPTextModel.from_pretrained(self.text_encoder)
        self.unet = UNet3DConditionModel.from_pretrained_2d(
            self.unet, sample_size=self.resolution // (2 ** (len(self.vae.config.block_out_channels) - 1)),
            in_channels=self.in_channels,
            add_temp_transformer=self.hparams.get("add_temp_transformer", False),
            add_temp_attn_only_on_upblocks=self.hparams.get("add_temp_attn_only_on_upblocks", False),
            prepend_first_frame=self.hparams.get("prepend_first_frame", False),
            add_temp_embed=self.hparams.get("add_temp_embed", False),
            add_temp_ff=self.hparams.get("add_temp_ff", False),
            add_temp_conv=self.hparams.get("add_temp_conv", False),
            num_class_embeds=self.hparams.get("num_class_embeds", None)
        )
        
        # load previously trained components for resumed training
        ckpt_path = self.hparams.get("ckpt_path", None)
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            
            mod_list = ["unet", "text_encoder"] if self.temporal_vae_path \
                else ["unet", "text_encoder", "vae", "enclip_projector"]
            for mod in mod_list:
                if any(filter(lambda x: x.startswith(mod), state_dict.keys())):
                    mod_instance = getattr(self, mod)
                    mod_instance.load_state_dict(
                        {k[len(mod) + 1:]: v for k, v in state_dict.items() if k.startswith(mod)}, strict=False
                    )
        
        # null text for classifier-free guidance
        self.null_text_token_ids = self.tokenizer(  # noqa
            "", max_length=self.tokenizer.model_max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        ).input_ids[0]

        # train only the trainable modules
        trainable_modules = self.hparams.get("trainable_modules", None)
        if trainable_modules is not None:
            self.unet.requires_grad_(False)
            for pname, params in self.unet.named_parameters():
                if any([re.search(pat, pname) for pat in trainable_modules]):
                    params.requires_grad = True
        if self.add_entity_clip:
            for pname, params in self.entity_clip_model.named_parameters():
                params.requires_grad = False
            for pname, params in self.enclip_projector.named_parameters():
                params.requires_grad = True
            
        # raise error when `in_channel` > 4 and `conv_in` is not trainable
        if self.use_prompt_image and not self.unet.conv_in.weight.requires_grad:
            raise AssertionError(f"use_prompt_image=True but `unet.conv_in` is frozen.")
        if not self.use_prompt_image and self.unet.conv_in.weight.requires_grad:
            logger.warning(f"use_prompt_image=False but `unet.conv_in` is trainable.")

        # load clip modules for evaluation
        self.clip = CLIPModel.from_pretrained(self.clip)
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_processor)
        # prepare modules
        for component in [self.vae, self.text_encoder, self.clip]:
            if not isinstance(component, CLIPTextModel) or self.hparams.get("freeze_text_encoder", False):
                component.requires_grad_(False).eval()
            if stage != "test" and self.trainer.precision.startswith("16"):
                component.to(dtype=torch.float16)
        # [DEBUG] show which parameters are trainable
        if os.environ.get("DEBUG_ON", None):
            params_trainable, params_frozen = [], []
            for name, params in self.named_parameters():
                if params.requires_grad:
                    params_trainable.append(name)
                else:
                    params_frozen.append(name)
            logger.info(f"*** [Trainable parameters]: {params_trainable}")
            logger.info(f"*** [Frozen parameters]: {params_frozen}")
        # use gradient checkpointing
        if self.hparams.get("enable_gradient_checkpointing", True):
            if not self.hparams.get("freeze_text_encoder", False):
                self.text_encoder.gradient_checkpointing_enable()
            self.unet.enable_gradient_checkpointing()
        # use xformers for efficient training
        if self.hparams.get("enable_xformers", True) and not hasattr(F, "scaled_dot_product_attention"):
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
                if self.unet.temp_transformer is not None:
                    # FIXME: disable this specific layer otherwise CUDA error occurred
                    self.unet.temp_transformer.set_use_memory_efficient_attention_xformers(False)
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
        # construct pipeline for inference
        self.val_pipeline = SDVideoPipeline(
            vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer, unet=self.unet,
            entity_clip_model=self.entity_clip_model,enclip_projector=self.enclip_projector,
            scheduler=DDIMScheduler.from_pretrained(self.noise_scheduler_dir),
        )

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"].to(dtype=torch.float16) \
            if self.trainer.precision.startswith("16") else batch["pixel_values"]
        entity_vae = batch["entity_vae"].to(dtype=torch.float16) \
            if self.trainer.precision.startswith("16") else batch["entity_vae"]
        entity_clip = batch["entity_clip"].to(dtype=torch.float16) \
            if self.trainer.precision.startswith("16") else batch["entity_clip"]
        
        text_token_ids = batch["text_token_ids"]
        video_len = pixel_values.shape[1]
        # inference arguments
        num_inference_steps = self.hparams.get("num_inference_steps", 50)
        guidance_scale = self.hparams.get("guidance_scale", 7.5)
        noise_alpha = self.hparams.get("noise_alpha", .0)
        # parase prompts
        prompts = self.tokenizer.batch_decode(text_token_ids, skip_special_tokens=True)
        generator = torch.Generator(device=self.device)
        seed = 42
        generator.manual_seed(seed)
        # compose args
        pipeline_args = dict(
            generator=generator, num_frames=video_len,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, noise_alpha=noise_alpha,
            frame_rate=1 if self.hparams.get("num_class_embeds", None) else None
        )
        samples = []
        
        for example_id in range(len(prompts)):
            prompt = prompts[example_id]
            prompt_image = pixel_values[example_id][:1, ...]
            entity_vae_image = torch.unsqueeze(entity_vae[example_id], 0)
            entity_clip_image = torch.unsqueeze(entity_clip[example_id], 0)
            sample = self.val_pipeline(
                prompt, prompt_image if self.use_prompt_image else None,
                entity_vae=entity_vae_image, entity_clip=entity_clip_image,
                add_entity_vae=self.add_entity_vae, add_entity_clip=self.add_entity_clip,
                **pipeline_args
            ).videos
            if self.trainer.is_global_zero:
                num_step_str = str(self.global_step).zfill(len(str(self.trainer.estimated_stepping_batches)))
                if prompt == "":
                    prompt = "text_prompt_equal_null"
                save_videos_grid(sample, Path(f"samples_s{num_step_str}", f"{prompt}.gif"))
            samples.append(sample)
        # clip model for metric
        clip_scores = compute_clip_score(
            model=self.clip, model_processor=self.clip_processor,
            images=torch.cat(samples), texts=list(prompts), rescale=False,
        )
        self.log("val_clip_score", clip_scores.mean(), on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer_args = dict(
            params=filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.get("lr", 1e-3),
            weight_decay=self.hparams.get("weight_decay", 1e-2)
        )
        optimizer = AdamW(**optimizer_args)
        # valid scheduler names: diffusers.optimization.SchedulerType
        scheduler_name = self.hparams.get("scheduler_name", "cosine")
        scheduler = get_scheduler(
            scheduler_name, optimizer=optimizer,
            num_warmup_steps=self.hparams.get("warmup_steps", 8),
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        lr_scheduler = dict(scheduler=scheduler, interval="step", frequency=1)
        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler)


class SDVideoModelEvaluator:
    def __init__(self, **kwargs):
        torch.multiprocessing.set_start_method("spawn", force=True)
        torch.multiprocessing.set_sharing_strategy("file_system")

        self.seed = kwargs.pop("seed", 42)
        self.prompts = kwargs.pop("prompts", None)
        if self.prompts is None:
            raise ValueError(f"No prompts provided.")
        elif isinstance(self.prompts, str) and not Path(self.prompts).exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompts}")
        elif isinstance(self.prompts, str):
            if self.prompts.endswith(".txt"):
                with open(self.prompts, "r", encoding="utf-8") as f:
                    self.prompts = [x.strip() for x in f.readlines() if x.strip()]
            elif self.prompts.endswith(".json"):
                with open(self.prompts, "r", encoding="utf-8") as f:
                    self.prompts = sorted([
                        random.choice(x) if isinstance(x, list) else x
                        for x in json.load(f).values()
                    ])
            elif self.prompts.endswith(".csv"):
                # prompt images can be set in this condition.
                csv_path = self.prompts
                df_prompts = pd.read_csv(self.prompts)
                self.prompts = df_prompts.iloc[:, 0].tolist()
                if len(df_prompts.columns) >= 2:
                    self.prompts_img = [Path(csv_path).parent.joinpath(x).as_posix() for x in df_prompts.iloc[:, 1]]
                else:
                    self.prompts_img = None

        self.add_file_logger(logger, kwargs.pop("log_file", None))
        self.output_file = kwargs.pop("output_file", "results.csv")
        self.batch_size = kwargs.pop("batch_size", 4)
        self.val_params = kwargs

    @staticmethod
    def add_file_logger(logger, log_file=None, log_level=logging.INFO):
        if dist_envs.global_rank == 0 and log_file is not None:
            log_handler = logging.FileHandler(log_file, "w")
            log_handler.setFormatter(
                logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s"))
            log_handler.setLevel(log_level)
            logger.addHandler(log_handler)

    @staticmethod
    def infer(rank, model, model_params, q_input, q_output, seed=42):
        device = torch.device(f"cuda:{rank}")
        model.to(device)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + rank)
        output_video_dir = Path("output_videos")
        output_video_dir.mkdir(parents=True, exist_ok=True)
        while True:
            inputs = q_input.get()
            if inputs is None:  # check for sentinel value
                print(f"[{datetime.now()}] Process #{rank} ended.")
                break
            start_idx, prompts, prompts_img = inputs
            if prompts_img is not None:
                prompts_img= prompts_img.to(device)
            videos = model.val_pipeline(
                prompts, prompts_img, generator=generator, negative_prompt=["watermark"] * len(prompts),
                **model_params
            ).videos
            for idx, prompt in enumerate(prompts):
                gif_file = output_video_dir.joinpath(f"{start_idx + idx}_{prompt}.gif")
                save_videos_grid(videos[idx:idx + 1, ...], gif_file)
                print(f"[{datetime.now()}] Sample is saved #{start_idx + idx}: \"{prompt}\"")
            clip_scores = compute_clip_score(
                model=model.clip, model_processor=model.clip_processor,
                images=videos, texts=prompts, rescale=False,
            )
            q_output.put((prompts, clip_scores.cpu().tolist()))
        return None

    def __call__(self, model):
        model.eval()

        # load prompts images if exist
        if model.use_prompt_image and self.prompts_img is not None:
            self.prompts_img = torch.stack([
                load_image(x, model.resolution, True, True)
                for x in self.prompts_img
            ])

        if not torch.cuda.is_available():
            raise NotImplementedError(f"No GPU found.")

        self.val_params.setdefault(
            "num_inference_steps", model.hparams.get("num_inference_steps", 50)
        )
        self.val_params.setdefault(
            "guidance_scale", model.hparams.get("guidance_scale", 7.5)
        )
        self.val_params.setdefault(
            "noise_alpha", model.hparams.get("noise_alpha", .0)
        )
        logger.info(f"val_params: {self.val_params}")

        q_input = torch.multiprocessing.Queue()
        q_output = torch.multiprocessing.Queue()
        processes = []
        for rank in range(torch.cuda.device_count()):
            p = torch.multiprocessing.Process(
                target=self.infer,
                args=(rank, model, self.val_params, q_input, q_output, self.seed)
            )
            p.start()
            processes.append(p)
        # send model inputs to queue
        result_num = 0
        for start_idx in range(0, len(self.prompts), self.batch_size):
            result_num += 1
            ref_images = self.prompts_img[start_idx:start_idx + self.batch_size] \
                if model.use_prompt_image and self.prompts_img is not None else None
            q_input.put((
                start_idx,
                self.prompts[start_idx:start_idx + self.batch_size],
                ref_images
            ))
        for _ in processes:
            q_input.put(None)  # sentinel value to signal subprocesses to exit
        # The result queue has to be processed before joining the processes.
        results = [q_output.get() for _ in range(result_num)]
        # joining the processes
        for p in processes:
            p.join()  # wait for all subprocesses to finish
        all_prompts, all_clip_scores = [], []
        for prompts, clip_scores in results:
            all_prompts.extend(prompts)
            all_clip_scores.extend(clip_scores)
        output_df = pd.DataFrame({
            "prompt": all_prompts, "clip_score": all_clip_scores
        })
        output_df.to_csv(self.output_file, index=False)
        logger.info(f"--- Metrics ---")
        logger.info(f"Mean CLIP_SCORE: {sum(all_clip_scores) / len(all_clip_scores)}")
        logger.info(f"Test results saved in: {self.output_file}")
