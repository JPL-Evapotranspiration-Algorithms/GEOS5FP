from os.path import join
from matplotlib.colors import LinearSegmentedColormap

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
DEFAULT_DOWNSAMPLING = "cubic"
