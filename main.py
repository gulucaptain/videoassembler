import logging
import os

import hydra
from hydra.utils import instantiate
from lightning_fabric.utilities.seed import seed_everything
from omegaconf import OmegaConf

from data import download_data, download_model
from utils import instantiate_multi, save_config, dist_envs, enable_logger

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="example", version_base=None)
def main(config):
    # 1. initialize trainer (should be done at the first place)
    trainer_callbacks = instantiate_multi(config, "callbacks")
    trainer_loggers = instantiate_multi(config, "loggers")
    trainer = instantiate(
        config.trainer, callbacks=trainer_callbacks, logger=trainer_loggers,
    )
    # 2. save envs and configs
    save_config(config)
    seed_everything(config.get("seed", 42), workers=True)
    dist_envs.init_envs(trainer)
    # 3. enable some loggers
    if "log_master" in config.callbacks and config.callbacks.log_master is not None:
        enable_logger("callbacks.log_master", config.callbacks.log_master.log_file)
    for log_name in (
            "__main__", "data.dataloader", "data.datasets", "data.data_downloader",
            "model.model", "components.interpolation.module", "components.interpolation.pipeline"
    ):
        enable_logger(log_name)
    # 4. build model
    remote_paths = ["pretrained_model_path", "ckpt_path", "temporal_vae_path"]
    for remote_path in remote_paths:
        if hasattr(config.model, remote_path):
            rpath = getattr(config.model, remote_path)
            setattr(config.model, remote_path, download_model(rpath))
    model = instantiate(config.model)
    # [DEBUG] show whole config
    if os.environ.get("DEBUG_ON", None):
        logger.info(OmegaConf.to_yaml(config))
    # 5. starting training/testing
    evaluator = config.get("evaluator", None)
    if evaluator is None or evaluator == "pl_validate":
        # config.data = download_data(config.data)
        datamodule = instantiate(config.data)
        run_fn = trainer.fit if evaluator is None else trainer.validate
        run_fn(model=model, datamodule=datamodule)
    else:
        model.setup(stage="test")
        evaluator = instantiate(config.evaluator)
        evaluator(model)
    logger.info("All finished.")


if __name__ == "__main__":
    main()
