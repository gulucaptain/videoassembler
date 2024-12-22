import bisect
import importlib
import logging
import os
import random
from collections import Counter
from operator import itemgetter
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageSequence
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import Compose, Resize, CenterCrop, Lambda
from torchvision.transforms.functional import pil_to_tensor

from utils import dist_envs
import cv2
import random

if importlib.util.find_spec("decord"):
    import decord

logger = logging.getLogger(__name__)


def sparse_sample(total_frames, sample_frames, sample_rate, random_sample=False):
    if sample_frames <= 0:  # sample over the total sequence of frames
        ids = np.arange(0, total_frames, sample_rate, dtype=int).tolist()
    elif sample_rate * (sample_frames - 1) + 1 <= total_frames:
        offset = random.randrange(total_frames - (sample_rate * (sample_frames - 1))) \
            if random_sample else 0
        ids = list(range(offset, total_frames + offset, sample_rate))[:sample_frames]
    else:
        ids = np.linspace(0, total_frames, sample_frames, endpoint=False, dtype=int).tolist()
    return ids

def load_image(img_path, resize_res=None, crop=False, rescale=False):
    """ Load an image to tensor

    Args:
        img_path: image file path to load.
        resize_res: resolution of resize, no resize if None
        crop: center crop if True
        rescale: values in [-1, 1) if rescale=True otherwise [0, 255)

    Returns:
        torch.Tensor: image tensor in the shape of [c, h, w]
    """
    img = pil_to_tensor(pil_loader(img_path))
    tsfm_ops = []
    crop_size = min(img.shape[-2:])
    if resize_res:
        tsfm_ops.append(Resize(resize_res, antialias=False))
        crop_size = resize_res
    if crop:
        tsfm_ops.append(CenterCrop(crop_size))
    if rescale:
        tsfm_ops.append(Lambda(lambda pixels: pixels / 127.5 - 1.0))
    transform = Compose(tsfm_ops)
    return transform(img)


def load_video(file_path, n_sample_frames, sample_rate=4, random_sample=False, transform=None, selected_frames=None):
    # random_sample = True
    sample_args = dict(
        sample_frames=n_sample_frames, sample_rate=sample_rate, random_sample=random_sample)
    video = []
    if Path(file_path).is_dir():
        img_files = sorted(Path(file_path).glob("*"), key=lambda i: int(i.stem[6:]))
        if len(img_files) < 1:
            logger.error(f"No data in video directory: {file_path}")
            raise FileNotFoundError(f"No data in video directory: {file_path}")
        
        # sample_ids = sparse_sample(len(img_files), **sample_args)
        selected_frames_ids = sparse_sample(len(selected_frames), **sample_args) ##FIXME change to selected frames, not all the frames
        sample_ids = []
        for i in range(0, len(selected_frames_ids)):
            sample_ids.append(selected_frames[selected_frames_ids[i]])
        
        for img_file in itemgetter(*sample_ids)(img_files):
            img = pil_loader(img_file.as_posix())
            img = pil_to_tensor(img)
            video.append(img)
    elif file_path.endswith(".gif"):
        with Image.open(file_path) as gif:
            sample_ids = sparse_sample(gif.n_frames, **sample_args)
            sample_ids_counter = Counter(sample_ids)
            for frame_idx, frame in enumerate(ImageSequence.Iterator(gif)):
                if frame_idx in sample_ids_counter:
                    frame = pil_to_tensor(frame.convert("RGB"))
                    for _ in range(sample_ids_counter[frame_idx]):
                        video.append(frame)
    else:
        vreader = decord.VideoReader(file_path)
        sample_ids = sparse_sample(len(vreader), **sample_args)
        frames = vreader.get_batch(sample_ids).asnumpy()  # (f, h, w, c)
        for frame_idx in range(frames.shape[0]):
            video.append(pil_to_tensor(Image.fromarray(frames[frame_idx]).convert("RGB")))
    video = torch.stack(video)  # (f, c, h, w)
    if transform is not None:
        video = transform(video)
    return video, sample_ids

def load_entity_vae(file_path, sample_ids, transform=None, use_rand_entity_sample=False, chosed_index=None):
    video = []
    selected_frames = [1]
    if Path(file_path).is_dir():
        img_files = sorted(Path(file_path).glob("*"), key=lambda i: int(i.stem[6:]))
        if len(img_files) < 1:
            logger.error(f"No data in video directory: {file_path}")
            raise FileNotFoundError(f"No data in video directory: {file_path}")
        index = 1
        for img_file in itemgetter(*sample_ids)(img_files):
            img_file = str(img_file)
            mask_file = img_file.replace("raw_vid/", "").replace("videos", "mask").replace("frame_","").replace("jpg","png")
            video_img = cv2.imread(img_file)
            mask_img = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            entity = cv2.bitwise_and(video_img, video_img, mask=mask_img)
            # cv2 -> PIL
            img = Image.fromarray(cv2.cvtColor(entity, cv2.COLOR_BGR2RGB))
            img = pil_to_tensor(img)
            if use_rand_entity_sample: ###FIXME random mask some frames
                if index not in selected_frames:
                    img = torch.zeros_like(img)
            video.append(img)
            index += 1
    video = torch.stack(video)  # (f, c, h, w)
    if transform is not None:
        video = transform(video)
    return video

def load_entity_clip(file_path, sample_ids, preprocess=None, use_rand_entity_sample=False, chosed_index=None, transform=None):
    video = []
    selected_frames = [1]
    if Path(file_path).is_dir():
        img_files = sorted(Path(file_path).glob("*"), key=lambda i: int(i.stem[6:]))
        if len(img_files) < 1:
            logger.error(f"No data in video directory: {file_path}")
            raise FileNotFoundError(f"No data in video directory: {file_path}")
        index = 1
        for img_file in itemgetter(*sample_ids)(img_files):
            img_file = str(img_file)
            mask_file = img_file.replace("raw_vid/", "").replace("videos", "mask").replace("frame_","").replace("jpg","png")
            video_img = cv2.imread(img_file)
            mask_img = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            entity = cv2.bitwise_and(video_img, video_img, mask=mask_img)

            img_tensor = torch.from_numpy(entity).permute(2, 0, 1).float()
            if use_rand_entity_sample: ###FIXME random mask some frames
                if index not in selected_frames:
                    img_tensor = torch.zeros_like(img_tensor)
            img = (img_tensor / 255. - 0.5) * 2
            video.append(img)
            index += 1
    video = torch.stack(video, dim=0)  # (f, c, h, w)
    if transform is not None:
        video = transform(video)
    return video


class MergeDataset(ConcatDataset):
    r"""Dataset as a merge of multiple datasets.

    Datasets are firstly split into multiple chunks and these chunks are merged
    one after another.

    Example:
        3 Datasets with sizes: [6, 17, 27]; split_num=4
        The global indices will be like:
        [[1, 4, 6]
         [1, 4, 6]
         [1, 4, 6]
         [3, 5, 9]]

    Args:
        datasets: List of datasets to be merged
    """

    @staticmethod
    def split_cumsum(group_sizes, split_num):
        r, s = [], 0
        for split in range(split_num):
            chunk_sizes = [x // split_num for x in group_sizes]
            if split == split_num - 1:
                chunk_sizes = [chunk_sizes[i] + group_sizes[i] % split_num
                               for i in range(len(group_sizes))]
            for chunk_size in chunk_sizes:
                r.append(chunk_size + s)
                s += chunk_size
        return r

    def __init__(self, datasets) -> None:
        super(MergeDataset, self).__init__(datasets)
        num_nodes, world_size = dist_envs.num_nodes, dist_envs.world_size
        # drop_last for all datasets
        self.datasets = list(datasets)
        self.dataset_sizes = list()
        for dataset in self.datasets:
            dataset_size = len(dataset)  # type: ignore[arg-type]
            self.dataset_sizes.append((dataset_size // world_size) * world_size)
        self.chunk_sizes = [x // num_nodes for x in self.dataset_sizes]
        self.cumulative_sizes = self.split_cumsum(self.dataset_sizes, num_nodes)

    def __len__(self):
        return sum(self.dataset_sizes)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        global_chunk_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if global_chunk_idx == 0:
            dataset_idx = 0
            sample_idx = idx
        else:
            dataset_idx = global_chunk_idx % len(self.datasets)
            sample_delta = idx - self.cumulative_sizes[global_chunk_idx - 1]
            sample_idx = global_chunk_idx // len(self.datasets) * self.chunk_sizes[dataset_idx] + sample_delta
        return self.datasets[dataset_idx][sample_idx]


class GroupDistributedSampler(DistributedSampler):
    r"""Sampler that restricts grouped data loading to a subset of the dataset.

    The dataset is firstly split into groups determined by `num_nodes`. The
    grouped datasets are then shuffled and distributed among devices in each
    node.
    """

    def __iter__(self):
        num_nodes = dist_envs.num_nodes
        assert self.num_replicas % num_nodes == 0
        indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
        chunk_size = self.total_size // num_nodes
        if self.shuffle:
            indices = []
            for node_rank in range(num_nodes):
                g = torch.Generator()
                g.manual_seed(self.seed + node_rank + self.epoch)
                chunk_indices = torch.randperm(
                    chunk_size, generator=g).tolist()  # type: ignore[arg-type]
                indices.extend([x + chunk_size * node_rank for x in chunk_indices])

        # remove tail of data to make it evenly divisible.
        indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        num_devices = self.num_replicas // num_nodes
        node_rank = int(self.rank / num_devices)
        local_rank = self.rank % num_devices
        indices = indices[local_rank:chunk_size:num_devices]
        indices = [x + chunk_size * node_rank for x in indices]
        assert len(indices) == self.num_samples

        return iter(indices)


class LabelEncoder:
    """ Encodes an label via a dictionary.
    Args:
        label_source (list of strings): labels of data used to build encoding dictionary.
    Example:
        >>> labels = ['label_a', 'label_b']
        >>> encoder = LabelEncoder(labels)
        >>> encoder.encode('label_a')
        tensor(0)
        >>> encoder.decode(encoder.encode('label_a'))
        'label_a'
        >>> encoder.encode('label_b')
        tensor(1)
        >>> encoder.size
        ['label_a', 'label_b']
    """

    def __init__(self, label_source: Union[list, str, os.PathLike]):
        if isinstance(label_source, list):
            self.labels = label_source
        else:
            with open(label_source, "r", encoding="utf-8") as f:
                lines = [x.strip() for x in f.readlines() if x.strip()]
            self.labels = lines
        self.idx_to_label = {idx: lab for idx, lab in enumerate(self.labels)}
        self.label_to_idx = {lab: idx for idx, lab in enumerate(self.labels)}

    @property
    def size(self):
        """
        Returns:
            int: Number of labels in the dictionary.
        """
        return len(self.labels)

    def encode(self, label):
        """ Encodes a ``label``.

        Args:
            label (object): Label to encode.

        Returns:
            torch.Tensor: Encoding of the label.
        """
        return torch.tensor(self.label_to_idx.get(label), dtype=torch.long)

    def batch_encode(self, iterator, dim=0):
        """
        Args:
            iterator (iterator): Batch of labels to encode.
            dim (int, optional): Dimension along which to concatenate tensors.

        Returns:
            torch.Tensor: Tensor of encoded labels.
        """
        return torch.stack([self.encode(x) for x in iterator], dim=dim)

    def decode(self, encoded):
        """ Decodes ``encoded`` label.

        Args:
            encoded (torch.Tensor): Encoded label.

        Returns:
            object: Label decoded from ``encoded``.
        """
        if encoded.numel() > 1:
            raise ValueError(
                '``decode`` decodes one label at a time, use ``batch_decode`` instead.')

        return self.idx_to_label[encoded.squeeze().item()]

    def batch_decode(self, tensor, dim=0):
        """
        Args:
            tensor (torch.Tensor): Batch of tensors.
            dim (int, optional): Dimension along which to split tensors.

        Returns:
            list: Batch of decoded labels.
        """
        return [self.decode(x) for x in [t.squeeze(0) for t in tensor.split(1, dim=dim)]]


def read_multi_csv(csv_dir):
    csvs = sorted(Path(csv_dir).glob("*.csv"))
    df_all = []
    for c in csvs:
        df = pd.read_csv(c)
        df["package"] = Path(c).stem
        df_all.append(df)
    df_all = pd.concat(df_all, ignore_index=True)
    return df_all
