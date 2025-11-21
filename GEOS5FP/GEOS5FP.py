from .constants import *
from .exceptions import *
from .GEOS5FP_granule import *
from .GEOS5FP_connection import *

# Optional import for point queries
try:
    from .GEOS5FP_point import *
except ImportError:
    pass

__author__ = 'Gregory H. Halverson'

GEOS5FP = GEOS5FPConnection
