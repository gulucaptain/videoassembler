import logging
import os
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

from utils import dist_envs
from utils import get_free_space
from utils import ops
from .constants import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_LABEL_FILENAME,
    DEFAULT_CSV_SUBDIR,
    DEFAULT_TAR_SUBDIR,
)
from .utils import read_multi_csv

logger = logging.getLogger(__name__)


class DataDownloader:
    def __init__(self, processes=8):
        self.pool = Pool(processes)

    def apply_job(self, remote_path, local_path, unzip_dir=None):
        # download file
        try:
            ops.copy(remote_path, local_path)
        except Exception as e:
            logger.error(
                f"File download failed (remote_path={remote_path}; local_path={local_path}): {e}"
            )
        logger.info(f"File downloaded: {os.path.basename(local_path)}")
        # unzip file
        if unzip_dir:
            self.wait_for_load_below(self.pool._processes)
            self.pool.apply_async(self.unzip, args=(local_path, unzip_dir))

    @staticmethod
    def unzip(file_path, unzip_dir):
        if not tarfile.is_tarfile(file_path):
            return
        with tarfile.open(file_path) as tarf:
            tarf.extractall(unzip_dir)
        logger.info(f"File unzipped: {os.path.basename(file_path)}")
        # delete zip file after unzipped
        os.remove(file_path)

    def wait_for_load_below(self, load_limit: int):
        t_start = time.monotonic()
        logger.info(f"Current free space: {get_free_space('/cache/'):.2f} GB.")
        while len(self.pool._cache) > load_limit:
            time.sleep(5)
            if time.monotonic() - t_start > 60:
                logger.info("Downloader is currently overloaded: waiting...")
        if load_limit == 0:
            self.pool.close()
            self.pool.join()


def download_dataset(downloader, dataset_cfg, local_dir, max_num_csv=0):
    # data are downloaded only on main process of each node
    if not dist_envs.is_initialized:
        raise ValueError(f"Distributed environments are not initialized!")
    # only the main process of each node is used for data downloading
    if dist_envs.local_rank != 0:
        return
    local_dir = Path(local_dir)
    # get directory names
    remote_dir = dataset_cfg["data_dir"]
    csv_subdir = dataset_cfg.get("csv_subdir", DEFAULT_CSV_SUBDIR)
    tar_subdir = dataset_cfg.get("tar_subdir", DEFAULT_TAR_SUBDIR)
    # download label file
    label_filename = dataset_cfg.get("label_filename", DEFAULT_LABEL_FILENAME)
    remote_label_file = os.path.join(remote_dir, label_filename)
    local_label_file = local_dir.joinpath(label_filename)
    if not local_label_file.exists() and ops.exists(remote_label_file):
        downloader.apply_job(remote_label_file, local_label_file.as_posix())
    # download csvs
    remote_csv_dir = os.path.join(remote_dir, csv_subdir)
    csv_files = [x for x in ops.listdir(remote_csv_dir) if x.endswith(".csv")]
    csv_files = csv_files[:max_num_csv] if max_num_csv > 0 else csv_files
    local_csv_dir = local_dir.joinpath(csv_subdir)
    for idx, csv_file in enumerate(csv_files):
        local_csv = local_csv_dir.joinpath(csv_file)
        if not local_csv.exists():
            logger.info(f"[{idx + 1}/{len(csv_files)}] "
                        f"Downloading csv file: {csv_file}")
            downloader.apply_job(
                os.path.join(remote_csv_dir, csv_file), local_csv.as_posix())

    tar_name = Path(csv_files[0]).stem
    tar_ext_real = None
    for tar_ext in ["", ".tar"]:
        if ops.exists(os.path.join(remote_dir, tar_subdir, tar_name + tar_ext)):
            tar_ext_real = tar_ext
            break
    if tar_ext_real is None:
        raise NotImplementedError(
            f"Extension of tar files is not recognized: {os.path.join(remote_dir, tar_subdir)}")
    tar_files = [Path(x).with_suffix(tar_ext_real) for x in csv_files]

    if dataset_cfg.get("split_among_nodes", False):
        # Only download a part of tar files according to current node_rank
        total_df = read_multi_csv(local_csv_dir)
        packages = total_df["package"].to_list()
        node_rank, num_nodes, world_size = \
            dist_envs.node_rank, dist_envs.num_nodes, dist_envs.world_size
        # drop_last is necessary when split_among_nodes=True
        packages = packages[:(len(packages) // world_size) * world_size]
        chunk_size = len(packages) // num_nodes
        packages = set(packages[node_rank * chunk_size:(node_rank + 1) * chunk_size])
        tar_files = [x for x in tar_files if Path(x).stem in packages]

    # Start download tar files
    remote_tar_dir = os.path.join(remote_dir, tar_subdir)
    local_tar_dir = local_dir.joinpath(tar_subdir)
    for idx, tar_file in enumerate(tar_files):
        local_tar = local_tar_dir.joinpath(tar_file)
        local_subdir = local_tar_dir.joinpath(local_tar.stem)
        if local_subdir.exists():
            continue
        logger.info(
            f"[{idx + 1}/{len(tar_files)}] "
            f"Downloading & extracting tar file: {tar_file}")
        downloader.apply_job(
            remote_path=os.path.join(remote_tar_dir, tar_file),
            local_path=local_tar.as_posix(),
            unzip_dir=local_tar_dir.as_posix()
        )
    logger.info(f"Dataset download complete: {Path(local_dir).name}")


def download_data(data_cfg):
    downloader = DataDownloader()
    stages = ("train", "train_alt", "val")
    for stage in stages:
        datasets = data_cfg.get(stage, dict()).items()
        for dataset_name, dataset_cfg in datasets:
            if dataset_cfg["data_dir"].startswith("s3://"):
                local_dir = Path(DEFAULT_DATA_DIR, dataset_name).as_posix()
                logger.info(f"Downloading dataset: {dataset_name}")
                download_dataset(downloader, dataset_cfg, local_dir,
                                 max_num_csv=dataset_cfg.get("max_num_csv", 0))
                data_cfg[stage][dataset_name]["data_dir"] = local_dir
    downloader.wait_for_load_below(0)
    return data_cfg


def download_model(model_path):
    if not model_path or not model_path.startswith("s3://"):
        return model_path
    local_path = Path(DEFAULT_MODEL_DIR, os.path.basename(os.path.normpath(model_path)))
    if dist_envs.local_rank == 0 and not local_path.exists():
        logger.info(f"Downloading model: {local_path.name}")
        downloader = DataDownloader()
        downloader.apply_job(model_path, local_path.as_posix())
        downloader.wait_for_load_below(0)
    return local_path.as_posix()
