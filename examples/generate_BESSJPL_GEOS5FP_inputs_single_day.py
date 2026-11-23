from os.path import dirname, join
import logging
import sys

from check_distribution import check_distribution

from ECOv002_calval_tables import load_calval_table
from GEOS5FP import GEOS5FP

GEOS5FP_INPUTS = [
    "Ca",
    "wind_speed_mps"
]

logger = logging.getLogger(__name__)

def generate_BESSJPL_GEOS5FP_inputs_single_day(
        filename: str = None,
        update_package_data: bool = True) -> None:
    """
    Generate BESSJPL GEOS-5 FP inputs for the first date in the input table.
    
    Args:
        filename: Output CSV filename (optional)
        update_package_data: Whether to save results to CSV
    
    Returns:
        DataFrame with results for the first day
    """
    # Load sample times and locations
    full_df = load_calval_table()
    
    # Convert time_UTC to datetime if it's not already
    if 'time_UTC' not in full_df.columns:
        raise ValueError("time_UTC column not found in the calval table")
    
    full_df['time_UTC'] = full_df['time_UTC'].astype(str)
    
    # Get the first date from the table
    first_datetime = full_df['time_UTC'].iloc[0]
    date_str = first_datetime.split(' ')[0]  # Extract YYYY-MM-DD
    
    logger.info(f"Generating BESSJPL GEOS-5 FP input table for first date: {date_str}:")

    for item in GEOS5FP_INPUTS:
        logger.info(f"  - {item}")
    
    # Filter for the first date
    targets_df = full_df[full_df['time_UTC'].str.startswith(date_str)].copy()
    
    if len(targets_df) == 0:
        logger.warning(f"No data found for date {date_str}")
        return None
    
    logger.info(f"Found {len(targets_df)} records for {date_str}")

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
    
    for column in results_df.columns:
        try:
            check_distribution(results_df[column], column)
        except Exception as e:
            continue

    if update_package_data:
        if filename is None:
            filename = join(dirname(__file__), f"ECOv002_calval_BESSJPL_GEOS5FP_inputs_{date_str}.csv")

        results_df.to_csv(filename, index=False)
        logger.info(f"Results saved to {filename}")

    return results_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_BESSJPL_GEOS5FP_inputs_single_day()
