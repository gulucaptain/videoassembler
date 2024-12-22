import logging

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.utilities.data import CombinedLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

from .datasets import DATASETS
from .utils import GroupDistributedSampler, MergeDataset, DistributedSampler

logger = logging.getLogger(__name__)


def build_dataloader(data_config: DictConfig, batch_size=4, shuffle=False,
                     drop_last=True, num_workers=8, pin_memory=True):
    cfg_datasets = list(data_config.values())
    datasets = []
    for cfg_dataset in cfg_datasets:
        dataset_cls = DATASETS.get(cfg_dataset.get("type"))
        cfg_dataset.update({"random_sample": shuffle})
        dataset = dataset_cls(**cfg_dataset)
        datasets.append(dataset)
    if len(datasets) == 1:
        dataset = ConcatDataset(datasets)
    else:
        if not drop_last:
            logger.warning(f"Option `drop_last` is forced activated when merging multiple datasets.")
        dataset = MergeDataset(datasets)
    dataloader_args = dict(
        dataset=dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    if not drop_last:
        dataloader = DataLoader(**dataloader_args, shuffle=shuffle, drop_last=drop_last)
        # sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
        # dataloader = DataLoader(**dataloader_args, sampler=sampler)
    else:
        sampler = GroupDistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
        dataloader = DataLoader(**dataloader_args, sampler=sampler)
    return dataloader


class SDVideoDataModule(pl.LightningDataModule):
    def __init__(
            self, train, val, train_alt=None, batch_size_train=1, batch_size_val=1,
            num_workers=8, pin_memory=True, **kwargs
    ):
        super().__init__()
        # make `prepare_data` called in each node when ddp is used.
        self.prepare_data_per_node = True
        # dir config
        self.train = train
        self.val = val
        self.train_alt = train_alt
        # dataloader config
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        shuffle, drop_last = True, True
        dataloader = build_dataloader(
            self.train, shuffle=shuffle, drop_last=drop_last, batch_size=self.batch_size_train,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        dataloaders = {"train": dataloader}
        if not self.train_alt:
            return dataloaders
        dataloader_alt = build_dataloader(
            self.train_alt, shuffle=shuffle, drop_last=drop_last, batch_size=self.batch_size_train * 8,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        dataloaders.update({"train_alt": dataloader_alt})
        return CombinedLoader(dataloaders, mode="max_size_cycle")

    def val_dataloader(self):
        dataloader = build_dataloader(
            self.val, shuffle=False, drop_last=False, batch_size=self.batch_size_val,
            num_workers=self.num_workers
        )
        return dataloader
