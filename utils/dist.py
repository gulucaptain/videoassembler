import logging

logger = logging.getLogger(__name__)


def ensure_env_init(fn):
    def wrapped_fn(obj):
        if not obj.is_initialized:
            logger.error("Distributed environments are not initialized!")
        return fn(obj)

    return wrapped_fn


class DistEnvs:
    """Environments for distributed training."""

    def __init__(self):
        self._world_size = 1
        self._num_nodes = 1
        self._node_rank = 0
        self._global_rank = 0
        self._local_rank = 0
        self.is_initialized = False

    def init_envs(self, trainer):
        """Distributed Environments need be initialized once and only once."""
        if self.is_initialized:
            logger.warning("Distributed environments are repeatedly initialized!")
        self.is_initialized = True
        self._world_size = trainer.world_size
        self._num_nodes = trainer.num_nodes
        self._node_rank = trainer.node_rank
        self._global_rank = trainer.global_rank
        self._local_rank = trainer.local_rank

    @property
    @ensure_env_init
    def world_size(self):
        return self._world_size

    @property
    @ensure_env_init
    def num_nodes(self):
        return self._num_nodes

    @property
    @ensure_env_init
    def node_rank(self):
        return self._node_rank

    @property
    @ensure_env_init
    def global_rank(self):
        return self._global_rank

    @property
    @ensure_env_init
    def local_rank(self):
        return self._local_rank


# singleton by module
dist_envs = DistEnvs()
