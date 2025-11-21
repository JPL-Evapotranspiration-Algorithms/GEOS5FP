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

# GEOS-5 FP Variable Mappings
# Maps variable name to (description, product, variable)
GEOS5FP_VARIABLES = {
    "SM": ("top layer soil moisture", "tavg1_2d_lnd_Nx", "SFMC"),
    "SFMC": ("top layer soil moisture", "tavg1_2d_lnd_Nx", "SFMC"),
    "LAI": ("leaf area index", "tavg1_2d_lnd_Nx", "LAI"),
    "LHLAND": ("latent heat flux land", "tavg1_2d_lnd_Nx", "LHLAND"),
    "EFLUX": ("total latent energy flux", "tavg1_2d_flx_Nx", "EFLUX"),
    "PARDR": ("PARDR", "tavg1_2d_lnd_Nx", "PARDR"),
    "PARDF": ("PARDF", "tavg1_2d_lnd_Nx", "PARDF"),
    "AOT": ("AOT", "tavg3_2d_aer_Nx", "TOTEXTTAU"),
    "COT": ("COT", "tavg1_2d_rad_Nx", "TAUTOT"),
    "Ts": ("Ts", "tavg1_2d_slv_Nx", "TS"),
    "Ts_K": ("Ts", "tavg1_2d_slv_Nx", "TS"),
    "Ta": ("Ta", "tavg1_2d_slv_Nx", "T2M"),
    "Ta_K": ("Ta", "tavg1_2d_slv_Nx", "T2M"),
    "Tmin": ("Tmin", "inst3_2d_asm_Nx", "T2MMIN"),
    "Tmin_K": ("Tmin", "inst3_2d_asm_Nx", "T2MMIN"),
    "PS": ("surface pressure", "tavg1_2d_slv_Nx", "PS"),
    "Q": ("Q", "tavg1_2d_slv_Nx", "QV2M"),
    "vapor_kgsqm": ("vapor_gccm", "inst3_2d_asm_Nx", "TQV"),
    "vapor_gccm": ("vapor_gccm", "inst3_2d_asm_Nx", "TQV"),
    "ozone_dobson": ("ozone_cm", "inst3_2d_asm_Nx", "TO3"),
    "ozone_cm": ("ozone_cm", "inst3_2d_asm_Nx", "TO3"),
    "U2M": ("U2M", "inst3_2d_asm_Nx", "U2M"),
    "V2M": ("V2M", "inst3_2d_asm_Nx", "V2M"),
    "CO2SC": ("CO2SC", "tavg3_2d_chm_Nx", "CO2SC"),
    "SWin": ("SWin", "tavg1_2d_rad_Nx", "SWGNT"),
    "SWTDN": ("SWTDN", "tavg1_2d_rad_Nx", "SWTDN"),
    "ALBVISDR": ("ALBVISDR", "tavg1_2d_rad_Nx", "ALBVISDR"),
    "ALBVISDF": ("ALBVISDF", "tavg1_2d_rad_Nx", "ALBVISDF"),
    "ALBNIRDF": ("ALBNIRDF", "tavg1_2d_rad_Nx", "ALBNIRDF"),
    "ALBNIRDR": ("ALBNIRDR", "tavg1_2d_rad_Nx", "ALBNIRDR"),
    "ALBEDO": ("ALBEDO", "tavg1_2d_rad_Nx", "ALBEDO"),
}
