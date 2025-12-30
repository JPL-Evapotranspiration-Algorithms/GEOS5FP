import logging
from datetime import datetime, timedelta
from shapely.geometry import MultiPoint
from GEOS5FP.GEOS5FP_connection import GEOS5FPConnection

# Clear existing handlers and reconfigure logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Explicitly set the logger level

def debug_query_with_multipoint():
    logger.debug("Starting debug query with MultiPoint geometry.")
    # Initialize the connection
    conn = GEOS5FPConnection()

    # Define the MultiPoint geometry
    multipoint = MultiPoint([
        (-118.25, 34.05),  # Los Angeles
        (-74.0, 40.7)      # New York
    ])

    # Define the time range
    end_time = datetime(2024, 11, 15)
    start_time = end_time - timedelta(days=7)

    # Define the target variables
    target_variables = ["T2M", "PS"]

    # Perform the query
    try:
        result = conn.query(
            target_variables=target_variables,
            time_range=(start_time, end_time),
            dataset="tavg1_2d_slv_Nx",
            geometry=multipoint,
            verbose=True
        )

        # Log the result
        logger.info("Query result:")
        logger.info(result)
    except Exception as e:
        logger.error(f"Query failed: {e}")

if __name__ == "__main__":
    debug_query_with_multipoint()