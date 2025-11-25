import os
import warnings

# Suppress NumExpr threading messages
os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count() or 1)
os.environ['NUMEXPR_NUM_THREADS'] = str(os.cpu_count() or 1)
warnings.filterwarnings('ignore', message='.*NumExpr.*')

from .GEOS5FP import *
from .version import __version__
from .validate_GEOS5FP_NetCDF_file import (
    validate_GEOS5FP_NetCDF_file,
    validate_GEOS5FP_directory,
    is_valid_GEOS5FP_file,
    quick_validate,
    safe_validate_GEOS5FP_NetCDF_file,
    GEOS5FPValidationError,
    GEOS5FPValidationResult,
    get_validation_summary
)

__author__ = "Gregory H. Halverson"
