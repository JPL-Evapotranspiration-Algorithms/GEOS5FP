from os.path import dirname, join
import logging
import sys

from ECOv002_calval_tables import load_times_locations
from GEOS5FP import GEOS5FP

GEOS5FP_INPUTS = [
    "Ta_C",            # Air temperature (°C)
    "RH",              # Relative humidity (fraction)
    "ALBEDO",          # Surface albedo
    "ALBVISDR",        # Visible direct beam albedo
    "ALBNIRDR",        # Near-infrared direct beam albedo
    "albedo_visible",  # Visible broadband albedo (computed from ALBVISDR and ALBVISDF)
    "albedo_NIR",      # NIR broadband albedo (computed from ALBNIRDR and ALBNIRDF)
]

logger = logging.getLogger(__name__)

def generate_BESSJPL_GEOS5FP_inputs(
        filename: str = None,
        update_package_data: bool = True) -> None:
    logger.info("Generating BESSJPL GEOS-5 FP input table:")

    for item in GEOS5FP_INPUTS:
        logger.info(f"  - {item}")

    # Load sample times and locations
    targets_df = load_times_locations()

    # Create GEOS5FP connection
    GEOS5FP_connection = GEOS5FP()

    # Query for BESSJPL GEOS5FP input variables
    # Note: Using verbose=True provides reliable progress updates every few seconds
    # Progress bar mode (verbose=False) can experience hanging issues with remote OPeNDAP queries
    results_df = GEOS5FP_connection.query(
        target_variables=GEOS5FP_INPUTS,
        targets_df=targets_df,
        verbose=True
    )

    if update_package_data:
        if filename is None:
            filename = join(dirname(__file__), "ECOv002_calval_BESSJPL_GEOS5FP_inputs.csv")

        results_df.to_csv(filename, index=False)

    return results_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_BESSJPL_GEOS5FP_inputs()
