import logging
import random
from math import sqrt
from pathlib import Path
from typing import Union, Tuple, List, Optional

import imageio
import torch
import torchvision
from einops import rearrange
from torch.utils.checkpoint import checkpoint

import torch.nn as nn
import kornia
import open_clip
import math

logger = logging.getLogger(__name__)


def randn_base(
        shape: Union[Tuple, List],
        mean: float = .0,
        std: float = 1.,
        generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
        device: Optional["torch.device"] = None,
        dtype: Optional["torch.dtype"] = None
):
    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        tensor = [
            torch.normal(
                mean=mean, std=std, size=shape, generator=generator[i],
                device=device, dtype=dtype
            )
            for i in range(len(generator))
        ]
        tensor = torch.cat(tensor, dim=0).to(device)
    else:
        tensor = torch.normal(
            mean=mean, std=std, size=shape, generator=generator, device=device,
            dtype=dtype
        )
    return tensor


def randn_mixed(
        shape: Union[Tuple, List],
        dim: int,
        alpha: float = .0,
        generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
        device: Optional["torch.device"] = None,
        dtype: Optional["torch.dtype"] = None
):
    """ Refer to Section 4 of Preserve Your Own Correlation:
        [A Noise Prior for Video Diffusion Models](https://arxiv.org/abs/2305.10474)
    """
    shape_shared = shape[:dim] + (1,) + shape[dim + 1:]

    # shared random tensor
    shared_std = alpha ** 2 / (1. + alpha ** 2)
    shared_tensor = randn_base(
        shape=shape_shared, mean=.0, std=shared_std, generator=generator,
        device=device, dtype=dtype
    )

    # individual random tensor
    indv_std = 1. / (1. + alpha ** 2)
    indv_tensor = randn_base(
        shape=shape, mean=.0, std=indv_std, generator=generator, device=device,
        dtype=dtype
    )

    return shared_tensor + indv_tensor


def randn_progressive(
        shape: Union[Tuple, List],
        dim: int,
        alpha: float = .0,
        generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
        device: Optional["torch.device"] = None,
        dtype: Optional["torch.dtype"] = None
):
    """ Refer to Section 4 of Preserve Your Own Correlation:
        [A Noise Prior for Video Diffusion Models](https://arxiv.org/abs/2305.10474)
    """
    num_prog = shape[dim]
    shape_slice = shape[:dim] + (1,) + shape[dim + 1:]
    tensors = [randn_base(shape=shape_slice, mean=.0, std=1., generator=generator, device=device, dtype=dtype)]
    beta = alpha / sqrt(1. + alpha ** 2)
    std = 1. / (1. + alpha ** 2)
    for i in range(1, num_prog):
        tensor_i = beta * tensors[-1] + randn_base(
            shape=shape_slice, mean=.0, std=std, generator=generator, device=device, dtype=dtype
        )
        tensors.append(tensor_i)
    tensors = torch.cat(tensors, dim=dim)
    return tensors


def prepare_masked_latents(images, vae_encode_func, scaling_factor=0.18215, sample_size=32, null_img_ratio=0):
    masks = torch.ones_like(images)  # shape: [b, f, c, h, w]
    if random.random() < (1 - null_img_ratio):
        masks[:, 0, ...] = 0
        # masks[:, random.randrange(masks.shape[1]), ...] = 0  # TK
    masked_latents = images * (masks < 0.5)
    # map masks into latent space
    masks = masks[:, :, :1, :sample_size, :sample_size]
    masks = rearrange(masks, "b f c h w -> b c f h w")
    # map masked_latents into latent space
    masked_latents = vae_encode_func(
        masked_latents.view(masked_latents.shape[0] * masked_latents.shape[1], *masked_latents.shape[2:])
    ).latent_dist.sample() * scaling_factor
    masked_latents = rearrange(masked_latents, "(b f) c h w -> b c f h w", f=images.shape[1])

    return masks, masked_latents

def decode_latents(vae_func, latents):
    scaling_factor = 0.18215
    video_length = latents.shape[2]
    latents = 1 / scaling_factor * latents
    latents = rearrange(latents, "b c f h w -> (b f) c h w")
    video = vae_func(latents).sample
    video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
    video = (video / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    video = video.float()
    return video

def prepare_entity_latents(images, vae_encode_func, vae_decode_func, scaling_factor=0.18215, sample_size=32, null_img_ratio=0):    
    # map masked_latents into latent space
    masked_latents = images
    masked_latents = vae_encode_func(
        masked_latents.view(masked_latents.shape[0] * masked_latents.shape[1], *masked_latents.shape[2:])
    ).latent_dist.sample() * scaling_factor
    entity_vae_latent = rearrange(masked_latents, "(b f) c h w -> b c f h w", f=images.shape[1])
    
    # save decode video
    # video = decode_latents(vae_decode_func, entity_vae_latent)
    # save_videos_grid(video, Path(f"samples_s{3000}", f"test decode.gif"))
    return entity_vae_latent

def save_videos_grid(videos, path, rescale=False, n_rows=4, fps=4):
    if videos.dim() == 4:
        videos = videos.unsqueeze(0)
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # [-1, 1) -> [0, 1)
        x = (x * 255).to(dtype=torch.uint8, device="cpu")
        outputs.append(x)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(Path(path).as_posix(), outputs, duration=1000 / fps, loop=0)


@torch.no_grad()
def compute_clip_score(model, model_processor, images, texts, local_bs=32, rescale=False):
    if rescale:
        images = (images + 1.0) / 2.0  # -1,1 -> 0,1
    images = (images * 255).to(torch.uint8)
    clip_scores = []
    for start_idx in range(0, images.shape[0], local_bs):
        img_batch = images[start_idx:start_idx + local_bs]
        batch_size = img_batch.shape[0]  # shape: [b c t h w]
        img_batch = rearrange(img_batch, "b c t h w -> (b t) c h w")
        outputs = []
        for i in range(len(img_batch)):
            images_part = img_batch[i:i + 1]
            model_inputs = model_processor(
                text=texts, images=list(images_part), return_tensors="pt", padding=True
            )
            model_inputs = {
                k: v.to(device=model.device, dtype=model.dtype)
                if k in ["pixel_values"] else v.to(device=model.device)
                for k, v in model_inputs.items()
            }
            logits = model(**model_inputs)["logits_per_image"]
            # For consistency with `torchmetrics.functional.multimodal.clip_score`.
            logits = logits / model.logit_scale.exp()
            outputs.append(logits)
        logits = torch.cat(outputs)
        logits = rearrange(logits, "(b t) p -> t b p", b=batch_size)
        frame_sims = []
        for logit in logits:
            frame_sims.append(logit.diagonal())
        frame_sims = torch.stack(frame_sims)  # [t, b]
        clip_scores.append(frame_sims.mean(dim=0))
    return torch.cat(clip_scores)


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()
    
    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        # "pooled",
        "last",
        "penultimate"
    ]

    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        pretrained = "/home/user/model/open_clip/open_clip_ViT_H_14/open_clip_pytorch_model.bin"
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=pretrained)
        # model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'))
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        self.device = self.model.positional_embedding.device
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPImageEmbedderV2(AbstractEncoder):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """
    
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda",
                 freeze=True, layer="pooled", antialias=True):
        super().__init__()
        
        pretrained = "/home/user/model/open_clip/open_clip_ViT_H_14/open_clip_pytorch_model.bin"
        # model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=pretrained)
        
        del model.transformer
        self.model = model
        self.device = device

        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "penultimate":
            raise NotImplementedError()
            self.layer_idx = 1

        self.antialias = antialias
        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic', align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image, no_dropout=False):
        ## image: b c h w
        z = self.encode_with_vision_transformer(image)
        return z

    def encode_with_vision_transformer(self, x):
        # x.shape: [1, 3, 320, 512]
        x = self.preprocess(x)
        # x.shape: [1, 3, 224, 224]
        
        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        if self.model.visual.input_patchnorm: # False
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(x.shape[0], x.shape[1], self.model.visual.grid_size[0], self.model.visual.patch_size[0], self.model.visual.grid_size[1], self.model.visual.patch_size[1])
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.model.visual.grid_size[0] * self.model.visual.grid_size[1], -1)
            x = self.model.visual.patchnorm_pre_ln(x)
            x = self.model.visual.conv1(x)
        else:
            x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # x.shape: [1, 256, 1280]

        # class embeddings and positional embeddings
        x = torch.cat(
            [self.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.visual.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.model.visual.patch_dropout(x)
        x = self.model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.visual.transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape: [1, 257, 1280]
        return x

def reshape_tensor(x, heads):
    bs, length, width = x.shape
    #(bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)
        
        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)

def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
    ):
        super().__init__()
        
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        
        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        latents = self.latents.repeat(x.size(0), 1, 1)
        
        x = self.proj_in(x)
        
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
            
        latents = self.proj_out(latents)
        return self.norm_out(latents)
    
class ImageProjModel(nn.Module):
    """Projection Model"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()        
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)
        
    def forward(self, image_embeds):
        #embeds = image_embeds
        embeds = image_embeds.type(list(self.proj.parameters())[0].dtype)
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens
