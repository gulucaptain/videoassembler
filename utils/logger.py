import logging

from .dist import dist_envs


def enable_logger(logger_name, log_file=None, log_level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.disabled = False
    if len(logger.handlers) > 0:
        return
    logger.propagate = False
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if not dist_envs.is_initialized:
        raise ValueError(f"Distributed environments are not initialized!")

    # only global_rank=0 will add a FileHandler
    if dist_envs.global_rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        handlers.append(file_handler)

    # only rank=0 for each node will print logs
    log_level = log_level if dist_envs.local_rank == 0 else logging.ERROR

    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    logger.setLevel(log_level)
