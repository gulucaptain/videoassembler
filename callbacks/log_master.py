import datetime
import json
import logging
import os
import time
from copy import deepcopy
from pathlib import Path

from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

from utils import ops

logger = logging.getLogger(__name__)


class LogMaster(Callback):
    def __init__(
            self, name, log_file, monitor="val_loss", monitor_fn="min",
            save_ckpt=True, remote_dir=None
    ):
        super().__init__()
        # logger's file handler is not set here and should be set outside
        # because rank is unknown when this function is called in ddp.
        self.name = name
        self.log_file = log_file
        self.monitor = monitor
        self.monitor_fn = eval(monitor_fn)
        self.save_ckpt = save_ckpt
        self.remote_dir = os.path.join(remote_dir, self.name) \
            if remote_dir else remote_dir
        self.model_perf = {}
        self.t_train_start, self.t_batch_start = .0, .0

    def on_train_start(self, trainer, pl_module) -> None:
        # log train configs
        logger.info(
            f"WORLD_SIZE={trainer.world_size}; "
            f"NUM_NODES={trainer.num_nodes}; "
            f"GPUS_PER_NODE={trainer.num_devices}; "
            f"STEPS_PER_EPOCH={trainer.num_training_batches}."
        )
        # upload training task config
        self.upload_to_cloud("config.yaml")
        # upload log file
        self.upload_to_cloud(self.log_file)
        self.t_train_start = time.monotonic()

    def on_train_batch_start(self, trainer, module, batch, batch_idx, unused=0):
        if batch_idx % trainer.log_every_n_steps == 1:
            self.t_batch_start = time.monotonic()

    def on_train_batch_end(
            self, trainer, module, outputs, batch, batch_idx, unused=0) -> None:
        if batch_idx % trainer.log_every_n_steps == 1:
            metrics = deepcopy(trainer.callback_metrics)
            msg = "; ".join([f"{k}={v.item():.4f}" for k, v in metrics.items()
                             if k.startswith("train")])
            zfill_batch = len(str(trainer.estimated_stepping_batches))
            time_elapsed = datetime.timedelta(
                seconds=int(time.monotonic() - self.t_train_start))
            time_remained = datetime.timedelta(
                seconds=int(
                    (time.monotonic() - self.t_batch_start) *
                    (trainer.estimated_stepping_batches - trainer.global_step)
                )
            )
            time_info = f"{time_elapsed} < {time_remained}"
            logger.info("[Steps {}/{}]: {} (Time: {})".format(
                str(trainer.global_step).zfill(zfill_batch),
                str(trainer.estimated_stepping_batches).zfill(zfill_batch),
                msg, time_info
            ))
            # upload log files
            self.upload_to_cloud(self.log_file)
            self.upload_to_cloud(self.get_tblog_dir(trainer.loggers))

    def on_validation_end(self, trainer, pl_module):
        # exit()
        zfill_batch = len(str(trainer.estimated_stepping_batches))
        metrics = deepcopy(trainer.callback_metrics)
        monitor_metric = metrics.get(self.monitor)
        if trainer.global_step > 200:
            if trainer.global_step % 5000 == 0:
                ckpt_name = f"steps_{str(trainer.global_step).zfill(zfill_batch)}.pth"
                self.model_perf.setdefault(ckpt_name, (ckpt_name, monitor_metric))
                if self.save_ckpt:
                    trainer.save_checkpoint(ckpt_name, weights_only=True)
        return
        if trainer.sanity_checking:
            return
        metrics = deepcopy(trainer.callback_metrics)
        metrics = {k: v.item() for k, v in metrics.items() if k.startswith("val")}
        if len(metrics) < 1:
            logger.warning("There are no metrics for validation!")
        zfill_batch = len(str(trainer.estimated_stepping_batches))
        msg = "; ".join([f"{k}={v:.6f}" for k, v in metrics.items()])
        logger.info("[Evaluation] [Steps={}/{}]: {}".format(
            str(trainer.global_step).zfill(zfill_batch),
            str(trainer.estimated_stepping_batches).zfill(zfill_batch), msg
        ))
        # upload log file
        if self.monitor not in metrics:
            raise KeyError(f"Metric `{self.monitor}` not in callback metrics: {metrics.keys()}")
        monitor_metric = metrics.get(self.monitor)
        self.upload_to_cloud(self.log_file)
        self.upload_to_cloud(self.get_tblog_dir(trainer.loggers))
        # compute model performance
        ckpt_name = f"steps_{str(trainer.global_step).zfill(zfill_batch)}.pth"
        self.model_perf.setdefault(ckpt_name, (ckpt_name, monitor_metric))
        # update best model information
        best_model = self.monitor_fn(self.model_perf, key=lambda x: self.model_perf.get(x)[-1])
        self.model_perf["best_model"] = (
            self.model_perf[best_model][0], self.model_perf[best_model][-1])
        perf_file = f"performances_{self.monitor}.json"
        with open(perf_file, "w") as f:
            json.dump(self.model_perf, f, indent=2)
        self.upload_to_cloud(perf_file)
        # save to checkpoint and upload to remote server
        if self.save_ckpt:
            trainer.save_checkpoint(ckpt_name, weights_only=True)
            if self.upload_to_cloud(ckpt_name):
                Path(ckpt_name).unlink()
        if self.remote_dir:
            logger.info(f"Detailed results are saved in: {self.remote_dir}")
        # upload output samples
        samples_dir = Path(f"samples_s{str(trainer.global_step).zfill(zfill_batch)}")
        self.upload_to_cloud(samples_dir.as_posix())

    @staticmethod
    def get_tblog_dir(loggers):
        tb_loggers = [x for x in loggers if isinstance(x, TensorBoardLogger)]
        return tb_loggers[0].log_dir if tb_loggers else ""

    @rank_zero_only
    def upload_to_cloud(self, local_file):
        if ops.mox_valid and self.remote_dir and os.path.exists(local_file):
            remote_path = os.path.join(
                self.remote_dir, os.path.basename(os.path.normpath(local_file)))
            ops.copy(local_file, remote_path)
            return remote_path
        else:
            return None
