import importlib
import os
import shutil
from distutils.dir_util import copy_tree


class FileOps:
    """
    Unified file operations for both s3 and local paths
    """

    def __init__(self):
        if importlib.util.find_spec("moxing"):
            import moxing as mox
            self.mox_valid = True
            self.mox = mox.file
        else:
            self.mox_valid = False
            self.mox = None

    @property
    def open(self):
        if self.mox_valid:
            return self.mox.File
        else:
            return open

    @property
    def exists(self):
        if self.mox_valid:
            return self.mox.exists
        else:
            return os.path.exists

    @property
    def listdir(self):
        if self.mox_valid:
            return self.mox.list_directory
        else:
            return os.listdir

    @property
    def isdir(self):
        if self.mox_valid:
            return self.mox.is_directory
        else:
            return os.path.isdir

    @property
    def makedirs(self):
        if self.mox_valid:
            return self.mox.make_dirs
        else:
            return os.makedirs

    @property
    def copy_dir(self):
        if self.mox_valid:
            return self.mox.copy_parallel
        else:
            return copy_tree

    @property
    def copy_file(self):
        if self.mox_valid:
            return self.mox.copy
        else:
            return shutil.copy

    def copy(self, src, dst, *args, **kwargs):
        if not self.exists(src):
            raise IOError('Source file {} does not exist.'.format(src))
        if self.isdir(src):
            self.copy_dir(src, dst, *args, **kwargs)
        else:
            self.copy_file(src, dst, *args, **kwargs)

    def mkdir_or_exist(self, path):
        if not self.exists(path):
            self.makedirs(path)

    @property
    def remove(self):
        if self.mox_valid:
            return self.mox.remove
        else:
            return os.remove


ops = FileOps()
