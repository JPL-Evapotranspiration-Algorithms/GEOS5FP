from os.path import join, dirname, abspath
from matplotlib.colors import LinearSegmentedColormap
import csv

DEFAULT_READ_TIMEOUT = 60
DEFAULT_RETRIES = 3

# DEFAULT_WORKING_DIRECTORY removed
DEFAULT_DOWNLOAD_DIRECTORY = join("~", "data", "GEOS5FP")
DEFAULT_USE_HTTP_LISTING = False
DEFAULT_COARSE_CELL_SIZE_METERS = 27440

SM_CMAP = LinearSegmentedColormap.from_list("SM", [
    "#f6e8c3",
    "#d8b365",
    "#99894a",
    "#2d6779",
    "#6bdfd2",
    "#1839c5"
])

NDVI_CMAP = LinearSegmentedColormap.from_list(
    name="LAI",
    colors=[
        "#000000",
        "#745d1a",
        "#e1dea2",
        "#45ff01",
        "#325e32"
    ]
)

DEFAULT_UPSAMPLING = "mean"
DEFAULT_DOWNSAMPLING = "lanczos"

# GEOS-5 FP Variable Mappings
# Load variable mappings from CSV file
def _load_variables():
    """Load GEOS-5 FP variable mappings from embedded CSV file."""
    variables_csv = join(dirname(abspath(__file__)), "variables.csv")
    variables_dict = {}
    
    with open(variables_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            variable_name = row['variable_name']
            description = row['description']
            product = row['product']
            variable = row['variable']
            variables_dict[variable_name] = (description, product, variable)
    
    return variables_dict

GEOS5FP_VARIABLES = _load_variables()

# Variable transformations (applied after data retrieval)
# Maps variable names to transformation functions
VARIABLE_TRANSFORMATIONS = {
    'Ta_C': lambda x: x - 273.15,  # Convert Kelvin to Celsius
}

# Computed/derived variables that require calling methods
# These cannot be queried directly but must be computed from other variables
COMPUTED_VARIABLES = {
    'RH',      # Relative Humidity (computed from Q, PS, Ta)
    'Ta_C',    # Temperature in Celsius (computed from Ta_K)
    'Ea_Pa',   # Actual Vapor Pressure (computed from RH and SVP)
    'SVP_Pa',  # Saturated Vapor Pressure (computed from Ta)
    'VPD_kPa', # Vapor Pressure Deficit (computed from SVP and Ea)
    'Td_K',    # Dew Point Temperature (computed from Ta and RH)
    'wind_speed_mps', # Wind speed in m/s (computed from U2M and V2M)
    'PAR_proportion', # PAR albedo fraction (computed from ALBVISDR / ALBEDO)
    'NIR_proportion', # NIR albedo fraction (computed from ALBNIRDR / ALBEDO)
}
