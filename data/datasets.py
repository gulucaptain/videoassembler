import logging
import random
import warnings
from ast import literal_eval
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, Lambda
from torchvision.transforms.functional import pil_to_tensor
from transformers import CLIPTokenizer

from utils import dist_envs
import os
import json
import open_clip
import kornia

from .constants import DEFAULT_TOKENIZER_DIR
from .registry import DATASETS
from .utils import read_multi_csv, load_video, load_entity_vae, load_entity_clip

logger = logging.getLogger(__name__)

warnings.simplefilter("error", Image.DecompressionBombWarning)

DATALOAD_TRY_TIMES = 64

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

class GeneralPairDataset(Dataset):
    def __init__(
            self, data_dir, csv_subdir, tar_subdir, resolution, tokenizer_dir,
            random_sample,
    ):
        self.random_sample = random_sample
        # data related
        self.data_dir = data_dir
        self.anns_dir = os.path.join(data_dir, csv_subdir)
        self.frame_anns = load_jsonl(self.anns_dir)
        # image related
        # image_crop_op = RandomCrop if random_sample else CenterCrop
        image_crop_op = CenterCrop
        # resize; crop; scale pixels from [0, 255) to [-1, 1)
        self.transform = Compose([
            Resize(resolution, antialias=False), image_crop_op(resolution),
            Lambda(lambda pixels: pixels / 127.5 - 1.0)
        ])
        self.entity_clip_transform = Compose([
            Resize(448, antialias=False), image_crop_op(448)
        ])
        # text related
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_dir)

        self.preprocess = None
    
    def __len__(self):
        return len(read_multi_csv(self.meta_dir))

    def __getitem__(self, idx):
        raise NotImplementedError


class VideoTextDataset(GeneralPairDataset):
    def __init__(
            self, data_dir, csv_subdir, tar_subdir, tokenizer_dir=DEFAULT_TOKENIZER_DIR,
            num_frames=8, resolution=512, random_sample=True, **kwargs,
    ):
        super().__init__(
            data_dir=data_dir, csv_subdir=csv_subdir, tar_subdir=tar_subdir,
            resolution=resolution, tokenizer_dir=tokenizer_dir,
            random_sample=random_sample
        )
        # sample config
        self.num_frames = num_frames
        # process dataframe
        # self.videos, self.texts = self.read_data(self.meta_dir)
        self.videos, self.tars, self.texts, self.coordinates = self.read_anns(self.frame_anns)

    @staticmethod
    def read_anns(anns):
        videos = []
        tars = []
        texts = []
        coordinates = []
        for ann in anns:
            videos.append(ann['vid'])
            tars.append(ann['tar'])
            texts.append(ann['text'])
            coordinates.append(ann['coordinates'])
        return videos, tars, texts, coordinates

    @staticmethod
    def read_data(meta_dir):
        df = read_multi_csv(meta_dir)
        videos, texts = df["video"].to_list(), df["caption"].to_list()
        return videos, texts

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        try_num = 0
        while try_num < DATALOAD_TRY_TIMES:
            try:
                video_path = Path(self.data_dir, "videos", self.tars[idx], "raw_vid", self.videos[idx])
                text = self.texts[idx]
                coords = self.coordinates[idx]
                if isinstance(text, list):
                    text = random.choice(text)
                # VIDEO
                frame_rate = 1
                pixel_values, sample_ids = load_video(
                    video_path.as_posix(),
                    n_sample_frames=self.num_frames,
                    sample_rate=frame_rate,
                    random_sample=self.random_sample,
                    transform=self.transform,
                    selected_frames=coords
                )
                use_rand_entity_sample = True
                chosed_index = 0.2
                entity_vae = load_entity_vae(
                    video_path, sample_ids=sample_ids, transform=self.transform, 
                    use_rand_entity_sample=use_rand_entity_sample, chosed_index=chosed_index
                )
                entity_clip = load_entity_clip(
                    video_path, sample_ids=sample_ids, 
                    preprocess=self.preprocess, use_rand_entity_sample=use_rand_entity_sample, 
                    chosed_index=chosed_index, transform = self.entity_clip_transform
                )
                # TEXT
                text_token_ids = self.tokenizer(
                    text,
                    max_length=self.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids[0]
                break
            except Exception as e:
                logger.warning(f"Exception occurred parsing video file ({self.videos[idx]}): {e}")
                idx = random.randrange(
                    len(self) // dist_envs.num_nodes * dist_envs.node_rank,
                    len(self) // dist_envs.num_nodes * (dist_envs.node_rank + 1)
                ) if try_num < DATALOAD_TRY_TIMES // 2 else random.randrange(len(self))
                try_num += 1
        # output
        output = dict(
            pixel_values=pixel_values, entity_vae=entity_vae, entity_clip=entity_clip, text_token_ids=text_token_ids, frame_rates=frame_rate
        )
        return output
        

@DATASETS.register_module()
class TgifDataset(VideoTextDataset):
    @staticmethod
    def read_data(meta_dir):
        df = read_multi_csv(meta_dir)
        df["video"] = df[["package", "id"]].agg(
            lambda xs: "{0}/{1}.gif".format(*xs), axis=1)
        # load data paths
        videos = df["video"].to_list()
        texts = list(map(literal_eval, df["caption"]))
        return videos, texts


@DATASETS.register_module()
class VatexDataset(VideoTextDataset):
    @staticmethod
    def read_data(meta_dir):
        df = read_multi_csv(meta_dir)
        df["video"] = df[["package", "videoID"]].agg(
            lambda xs: "{0}/{1}".format(*xs), axis=1)
        # load data paths
        videos = df["video"].to_list()
        texts = list(map(literal_eval, df["enCap"]))
        return videos, texts


@DATASETS.register_module()
class WebvidDataset(VideoTextDataset):
    @staticmethod
    def read_data(meta_dir):
        df = read_multi_csv(meta_dir)
        df["video"] = df[["package", "videoid"]].agg(
            lambda xs: "{0}/{1}.mp4".format(*xs), axis=1)
        # load data paths
        videos = df["video"].to_list()
        # add watermark as keyword since all videos in WebVid have watermarks
        texts = [[f"{x}, watermark", f"{x} with watermark"]
                 for x in df["name"].to_list()]
        return videos, texts


@DATASETS.register_module()
class K700CaptionDataset(VideoTextDataset):
    @staticmethod
    def read_data(meta_dir):
        df = read_multi_csv(meta_dir)
        df["video"] = df[["package", "id"]].agg(
            lambda xs: "{0}/{1}".format(*xs), axis=1)
        df["text"] = df[["class", "caption"]].agg(
            lambda xs: ["{0}: {1}".format(*xs),
                        "{1}, {0}".format(*xs)], axis=1)
        # load data paths
        videos = df["video"].to_list()
        texts = df["text"].to_list()
        return videos, texts


@DATASETS.register_module()
class MidjourneyVideoDataset(VideoTextDataset):
    @staticmethod
    def read_data(meta_dir):
        df = read_multi_csv(meta_dir)
        df["video"] = df[["package", "videoname"]].agg(
            lambda xs: "{0}/{1}".format(*xs), axis=1)
        # load data paths
        videos = df["video"].to_list()
        texts = df["caption"].to_list()
        return videos, texts


@DATASETS.register_module()
class MomentsInTimeDataset(VideoTextDataset):
    @staticmethod
    def read_data(meta_dir):
        df = read_multi_csv(meta_dir)
        df["video"] = df[["package", "Video Name"]].agg(
            lambda xs: "{0}/{1}".format(*xs), axis=1)
        # load data paths
        videos = df["video"].to_list()
        texts = df["caption"].to_list()
        return videos, texts


@DATASETS.register_module()
class PexelsDataset(VideoTextDataset):
    @staticmethod
    def read_data(meta_dir):
        df = read_multi_csv(meta_dir)
        df["video"] = df[["package", "id"]].agg(
            lambda xs: "{0}/{1}.mp4".format(*xs), axis=1)
        # load data paths
        videos = df["video"].to_list()
        texts = df["caption"].to_list()
        return videos, texts


class ImageTextDataset(GeneralPairDataset):
    def __init__(
            self, data_dir, csv_subdir, tar_subdir, tokenizer_dir=DEFAULT_TOKENIZER_DIR,
            resolution=512, random_sample=True, **kwargs,
    ):
        super().__init__(
            data_dir=data_dir, csv_subdir=csv_subdir, tar_subdir=tar_subdir,
            resolution=resolution, tokenizer_dir=tokenizer_dir,
            random_sample=random_sample
        )
        self.images, self.texts = self.read_data(self.meta_dir)

    @staticmethod
    def read_data(meta_dir):
        df = read_multi_csv(meta_dir)
        images, texts = df["image"].to_list(), df["text"].to_list()
        return images, texts

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try_num = 0
        while try_num < DATALOAD_TRY_TIMES:
            try:
                # IMAGE
                image_path = Path(self.data_dir, self.images[idx])
                image = pil_loader(image_path.as_posix())
                pixel_values = self.transform(pil_to_tensor(image))
                # TEXT
                text = self.texts[idx]
                text_token_ids = self.tokenizer(
                    text,
                    max_length=self.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids[0]
                break
            except Exception as e:
                logger.warning(f"Exception occurred parsing image file ({self.images[idx]}): {e}")
                idx = random.randrange(
                    len(self) // dist_envs.num_nodes * dist_envs.node_rank,
                    len(self) // dist_envs.num_nodes * (dist_envs.node_rank + 1)
                ) if try_num < DATALOAD_TRY_TIMES // 2 else random.randrange(len(self))
                try_num += 1
        # output
        frame_rates = random.randrange(30)
        output = dict(
            pixel_values=pixel_values, text_token_ids=text_token_ids, frame_rates=frame_rates
        )
        return output


@DATASETS.register_module()
class LaionDataset(ImageTextDataset):
    @staticmethod
    def read_data(meta_dir):
        df = read_multi_csv(meta_dir)
        df = df.dropna(subset=["dir", "text"], how="any").reset_index(drop=True)
        images, texts = df["dir"].to_list(), df["text"].to_list()
        return images, texts


@DATASETS.register_module()
class MidJourneyImageDataset(ImageTextDataset):
    @staticmethod
    def read_data(meta_dir):
        df = read_multi_csv(meta_dir)
        df = df.dropna(subset=["videoname", "caption"], how="any").reset_index(drop=True)
        df["videoname"] = df[["package", "videoname"]].agg(
            lambda xs: "{0}/{1}.png".format(*xs), axis=1)
        images, texts = df["videoname"].to_list(), df["caption"].to_list()
        return images, texts
