from .core import get_sampling_rate as _get_sampling_rate

from .sim import sim, sim_kinematic #, sim_dual, sim_dual_kinematic
from .elements import *
from .function_generators import *
from . import constants
from .errors import LeTalkerError
from . import utils


import logging

logger = logging.getLogger("letalker")
logger.addHandler(logging.NullHandler())


__version__ = "0.1.0"

def __getattr__(name):
    if name == "fs":
        return _get_sampling_rate()
    raise AttributeError()
