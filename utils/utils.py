import logging
import os
import re
import unicodedata

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def save_config(config, config_file="config.yaml"):
    with open(config_file, "w") as f:
        OmegaConf.save(config, f)


def get_free_space(path):
    if not os.path.exists(path):
        logger.info(f"Path dose not exist: {path}")
        return 0
    info = os.statvfs(path)
    free_size = info.f_bsize * info.f_bavail / 1024 ** 3  # GB
    return free_size


def instantiate_multi(config, name):
    instances = [instantiate(x) for x in config.get(name).values()
                 if x is not None and "_target_" in x] if name in config else []
    return instances


def get_free_mem(device):
    """ Get free memory of device. (MB) """
    mem_used, mem_total = torch.cuda.mem_get_info(device)
    mem_free = (mem_total - mem_used) / 1024.0 ** 3
    return mem_free


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')
