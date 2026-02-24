import logging

from . import _backend, constants, utils
from .core import get_sampling_rate as _get_sampling_rate
from .core import ts
from .elements import *
from .errors import LeTalkerError
from .function_generators import *
from .sim import sim, sim_kinematic, sim_vf  # , sim_dual, sim_dual_kinematic

logger = logging.getLogger("letalker")
logger.addHandler(logging.NullHandler())


__version__ = "0.1.0"


def __getattr__(name):
    if name == "fs":
        return _get_sampling_rate()
    raise AttributeError()
