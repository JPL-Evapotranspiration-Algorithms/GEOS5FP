from typing import Union, List, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm.notebook import tqdm
from rasters import Raster, RasterGeometry
from shapely.geometry import Point, MultiPoint
from dateutil import parser
import logging

from .constants import *
from .geometry_utils import is_point_geometry, extract_points
from .get_variable_info import get_variable_info

logger = logging.getLogger(__name__)

# Optional import for point queries via OPeNDAP
try:
    from .GEOS5FP_point import query_geos5fp_point
    HAS_OPENDAP_SUPPORT = True
except ImportError:
    HAS_OPENDAP_SUPPORT = False
    query_geos5fp_point = None

def query(
        target_variables: Union[str, List[str]] = None,
        targets_df: Union[pd.DataFrame, gpd.GeoDataFrame] = None,
        time_UTC: Union[datetime, str, List[datetime], List[str], pd.Series] = None,
        time_range: Tuple[Union[datetime, str], Union[datetime, str]] = None,
        dataset: str = None,
        geometry: Union[RasterGeometry, Point, MultiPoint, List, gpd.GeoSeries] = None,
        resampling: str = None,
        lat: Union[float, List[float], pd.Series] = None,
        lon: Union[float, List[float], pd.Series] = None,
        dropna: bool = True,
        temporal_interpolation: str = TEMPORAL_INTERPOLATION,
        variable_name: Union[str, List[str]] = None,
        verbose: bool = False,
        **kwargs) -> Union[np.ndarray, Raster, gpd.GeoDataFrame]:
    """
    General-purpose query method that can retrieve any variable from any dataset.
    
    This method provides a flexible interface for retrieving GEOS-5 FP data, supporting both:
    - Raster queries (for spatial data over a region)
    - Point queries (for time-series at specific coordinates using OPeNDAP)
    - Batch spatio-temporal queries (multiple points at different times)
    
    Parameters
    ----------
    target_variables : str or list of str
        Name(s) of the variable(s) to retrieve. Can be either:
        - A single variable name (str): "Ta_K", "SM", "SWin", etc.
        - A list of variable names (List[str]): ["Ta_K", "SM", "SWin"]
        Each can be either:
        - A predefined variable name from GEOS5FP_VARIABLES (e.g., "Ta_K", "SM", "SWin")
        - A raw GEOS-5 FP variable name (e.g., "T2M", "SFMC", "SWGDN")
        When multiple variables are requested with point geometry, each variable becomes a column.
    targets_df : DataFrame or GeoDataFrame, optional
        Input table containing 'time_UTC' and 'geometry' columns. When provided,
        the target variables will be queried for each row and added as new columns
        to this table, which is then returned. This is useful for generating
        validation tables or adding GEOS-5 FP data to existing datasets.
    time_UTC : datetime, str, list, or Series, optional
        Date/time in UTC. Can be:
        - Single datetime or str for raster queries or single point query
        - List of datetimes or Series for batch spatio-temporal queries
        Required if time_range not provided.
    time_range : tuple of (start, end), optional
        Time range (start_time, end_time) for time-series point queries.
        Use this with lat/lon for efficient multi-timestep queries at same location.
    dataset : str, optional
        GEOS-5 FP product/dataset name (e.g., "tavg1_2d_slv_Nx", "inst3_2d_asm_Nx").
        If not provided, will be looked up from GEOS5FP_VARIABLES.
        Required when using raw variable names or when querying multiple variables from same dataset.
    geometry : RasterGeometry or Point or MultiPoint, optional
        Target geometry. Can be:
        - RasterGeometry for spatial raster queries
        - shapely Point or MultiPoint for time-series queries
        - None for native GEOS-5 FP resolution
    resampling : str, optional
        Resampling method when reprojecting to target geometry (e.g., "nearest", "bilinear")
    lat : float, optional
        Latitude for point query (alternative to providing Point geometry)
    lon : float, optional
        Longitude for point query (alternative to providing Point geometry)
    dropna : bool, default=True
        Whether to drop NaN values from point query results
    temporal_interpolation : str, default="nearest"
        Method for handling different temporal resolutions when querying multiple variables:
        - "nearest": Use nearest neighbor in time for each variable independently
        - "interpolate": Linear interpolation between observations before and after target time
        This parameter is only used for multi-variable queries at a single time point.
    verbose : bool, optional
        Control logging verbosity. If None (default), uses the connection's verbose setting.
        - True: Detailed console logging with progress information
        - False: Single TQDM progress bar for notebook-friendly display
    **kwargs : dict
        Additional keyword arguments passed to interpolation or query functions
    
    Returns
    -------
    np.ndarray, Raster, or gpd.GeoDataFrame
        - np.ndarray: For single-variable point queries (Point or MultiPoint geometry), 
          returns a 1D array of values matching the query points
        - Raster: For spatial data queries at a single time (single variable only)
        - gpd.GeoDataFrame: For multi-variable point queries, batch queries with targets_df, 
          or point time-series, with geometry and variable column(s)
    
    Examples
    --------
    # Example 1: Single-point time-series using OPeNDAP (fast!)
    >>> from datetime import datetime, timedelta
    >>> conn = GEOS5FPConnection()
    >>> end_time = datetime(2024, 11, 15)
    >>> start_time = end_time - timedelta(days=7)
    >>> df = conn.query(
    ...     "T2M",
    ...     time_range=(start_time, end_time),
    ...     dataset="tavg1_2d_slv_Nx",
    ...     lat=34.05,
    ...     lon=-118.25
    ... )
    
    # Example 2: Multiple variables in point query
    >>> df = conn.query(
    ...     ["T2M", "PS", "QV2M"],
    ...     time_range=(start_time, end_time),
    ...     dataset="tavg1_2d_slv_Nx",
    ...     lat=34.05,
    ...     lon=-118.25
    ... )
    
    # Example 3: Raster query at single time
    >>> from rasters import RasterGeometry
    >>> geometry = RasterGeometry.open("target_area.tif")
    >>> raster = conn.query(
    ...     "T2M",
    ...     time_UTC="2024-11-15 12:00:00",
    ...     dataset="tavg1_2d_slv_Nx",
    ...     geometry=geometry
    ... )
    
    # Example 4: Use predefined variable name
    >>> df = conn.query(
    ...     "Ta_K",
    ...     time_range=(start_time, end_time),
    ...     lat=34.05,
    ...     lon=-118.25
    ... )
    
    # Example 5: Query multiple predefined variables
    >>> df = conn.query(
    ...     target_variables=["Ta_K", "SM", "SWin"],
    ...     time_range=(start_time, end_time),
    ...     lat=40.7,
    ...     lon=-74.0
    ... )
    
    # Example 6: Generate table with target variables
    >>> import geopandas as gpd
    >>> targets = gpd.GeoDataFrame({
    ...     'time_UTC': [datetime(2024, 11, 15, 12), datetime(2024, 11, 15, 13)],
    ...     'geometry': [Point(-118.25, 34.05), Point(-74.0, 40.7)]
    ... })
    >>> result = conn.query(
    ...     target_variables=["Ta_C", "RH"],
    ...     targets_df=targets
    ... )
    
    Notes
    -----
    - For time-series point queries, this method uses OPeNDAP which is much faster
        than iterating through individual timesteps
    - When both time_UTC and time_range are provided, time_range takes precedence
        for point queries
    - Point queries require xarray and netCDF4 to be installed
    - Multiple variables can only be queried simultaneously for point geometries
    - Raster queries only support single variables
    """
    if verbose is None:
        verbose = VERBOSE
    
    # Handle backward compatibility: variable_name is deprecated, use target_variables
    if target_variables is None and variable_name is not None:
        target_variables = variable_name
    elif target_variables is None:
        raise ValueError("target_variables parameter is required")
    
    # Handle targets_df parameter
    if targets_df is not None:
        # Validate targets_df has required columns
        if 'time_UTC' not in targets_df.columns:
            raise ValueError("targets_df must contain 'time_UTC' column")
        if 'geometry' not in targets_df.columns:
            raise ValueError("targets_df must contain 'geometry' column")
        
        # Extract time_UTC and geometry from targets_df
        time_UTC = targets_df['time_UTC']
        geometry = targets_df['geometry']
    
    # Normalize target_variables to list
    if isinstance(target_variables, str):
        variable_names = [target_variables]
        single_variable = True
    else:
        variable_names = target_variables
        single_variable = False
    
    # Validate inputs
    if time_UTC is None and time_range is None:
        raise ValueError("Either time_UTC or time_range must be provided")
    
    # Check for vectorized batch query (lists of times and geometries)
    is_batch_query = False
    if time_UTC is not None and time_range is None:
        # Check if time_UTC is a list/Series/tuple/array
        if isinstance(time_UTC, (list, tuple, pd.Series, np.ndarray)):
            is_batch_query = True
        # Check if lat/lon are lists/Series
        elif lat is not None and lon is not None:
            if isinstance(lat, (list, pd.Series)) or isinstance(lon, (list, pd.Series)):
                is_batch_query = True
        # Check if geometry is a GeoSeries
        elif geometry is not None and isinstance(geometry, gpd.GeoSeries):
            is_batch_query = True
    
    # Handle vectorized batch queries
    if is_batch_query:
        if not HAS_OPENDAP_SUPPORT:
            raise ImportError(
                "Point query support requires xarray and netCDF4. "
                "Install with: conda install -c conda-forge xarray netcdf4"
            )
        
        # Convert inputs to lists
        if isinstance(time_UTC, pd.Series):
            times = time_UTC.tolist()
        elif isinstance(time_UTC, (list, tuple)):
            times = list(time_UTC)
        elif isinstance(time_UTC, np.ndarray):
            times = time_UTC.tolist()
        else:
            raise ValueError("For batch queries, time_UTC must be a list, tuple, Series, or array")
        
        # Get geometries
        if geometry is not None:
            if isinstance(geometry, gpd.GeoSeries):
                geometries = geometry.tolist()
            elif isinstance(geometry, list):
                geometries = geometry
            elif is_point_geometry(geometry):
                # Single geometry - broadcast to all times
                geometries = [geometry] * len(times)
            else:
                raise ValueError("For batch queries with geometry, must provide GeoSeries, list, or single Point/MultiPoint")
        elif lat is not None and lon is not None:
            # Convert lat/lon to Point geometries
            if isinstance(lat, pd.Series):
                lats = lat.tolist()
            else:
                lats = lat if isinstance(lat, list) else [lat]
            
            if isinstance(lon, pd.Series):
                lons = lon.tolist()
            else:
                lons = lon if isinstance(lon, list) else [lon]
            
            if len(lats) != len(lons):
                raise ValueError("lat and lon must have the same length")
            
            geometries = [Point(lon_val, lat_val) for lon_val, lat_val in zip(lons, lats)]
        else:
            raise ValueError("For batch queries, must provide geometry or lat/lon")
        
        # Validate lengths match
        if len(times) != len(geometries):
            raise ValueError(
                f"Number of times ({len(times)}) must match number of geometries ({len(geometries)})"
            )
        
        # Parse time strings to datetime objects
        parsed_times = []
        for t in times:
            if isinstance(t, str):
                parsed_times.append(parser.parse(t))
            else:
                parsed_times.append(t)
        
        # Optimize queries: group by unique coordinates, then cluster times
        # to avoid querying excessively long time ranges
        
        # Build mapping of (lat, lon) -> list of (index, time, geometry)
        coord_to_records = {}
        for idx, (time_val, geom) in enumerate(zip(parsed_times, geometries)):
            # Extract point coordinates
            # Handle rasters geometry wrapper (similar to geometry_utils.extract_points)
            if hasattr(geom, 'geometry'):
                unwrapped_geom = geom.geometry
            else:
                unwrapped_geom = geom
            
            if isinstance(unwrapped_geom, Point):
                pt_lon, pt_lat = unwrapped_geom.x, unwrapped_geom.y
            elif isinstance(unwrapped_geom, MultiPoint):
                # Use first point
                pt_lon, pt_lat = unwrapped_geom.geoms[0].x, unwrapped_geom.geoms[0].y
            else:
                raise ValueError(f"Unsupported geometry type (must be Point or MultiPoint): {type(geom)}")

            # Round to avoid floating point issues
            coord_key = (round(pt_lat, 6), round(pt_lon, 6))
            
            if coord_key not in coord_to_records:
                coord_to_records[coord_key] = []
            
            coord_to_records[coord_key].append({
                'index': idx,
                'time': time_val,
                'geometry': geom
            })
        
        # Import cluster_times utility
        from GEOS5FP.cluster_times import cluster_times
        
        # Count total queries needed - this is an estimate before dataset grouping
        # The actual count will be calculated after grouping variables by dataset
        estimated_batches_per_coord = {}
        for coord_key, records in coord_to_records.items():
            clusters = cluster_times(records, max_days_per_query=1)
            estimated_batches_per_coord[coord_key] = len(clusters)
        
        if verbose:
            logger.info(
                f"Processing {len(parsed_times)} spatio-temporal records at "
                f"{len(coord_to_records)} unique coordinates..."
            )
        
        # Initialize timing for ETA tracking
        from time import time as current_time
        query_start_time = current_time()
        completed_batches = 0
        
        logger.info("DEBUG: About to initialize results_by_index...")
        
        # Initialize results dictionary indexed by original record index
        results_by_index = {i: {'time_UTC': t, 'geometry': g} 
                            for i, (t, g) in enumerate(zip(parsed_times, geometries))}
        
        logger.info(f"DEBUG: Initialized results_by_index with {len(results_by_index)} entries")
        logger.info(f"DEBUG: About to group variables by dataset for {len(variable_names)} variables...")
        
        # Group variables by dataset for efficient multi-variable queries
        from collections import defaultdict
        dataset_to_variables = defaultdict(list)
        computed_variables = []  # Track computed variables separately
        
        # Map computed variables to their dependencies
        computed_dependencies = {
            'RH': ['Q', 'PS', 'Ta'],  # Need these to compute RH
            'Ta_C': ['Ta'],  # Need Ta to convert to Celsius
            'wind_speed_mps': ['U2M', 'V2M'],  # Need wind components to compute speed
            'albedo_visible': ['ALBEDO', 'ALBVISDR'],  # Need total albedo and visible direct albedo
            'albedo_NIR': ['ALBEDO', 'ALBNIRDR'],  # Need total albedo and NIR direct albedo
        }
        
        for var_name in variable_names:
            logger.info(f"DEBUG: Processing variable '{var_name}'...")
            # Check if this is a computed variable
            if var_name in COMPUTED_VARIABLES:
                computed_variables.append(var_name)
                # Add dependencies to query list if not already present
                if var_name in computed_dependencies:
                    for dep in computed_dependencies[var_name]:
                        if dep not in variable_names and dep not in [v[0] for v in sum(dataset_to_variables.values(), [])]:
                            # Add dependency variable
                            if dep in GEOS5FP_VARIABLES:
                                _, dep_dataset, dep_raw_variable = get_variable_info(dep)
                                dep_opendap = dep_raw_variable.lower()
                                dataset_to_variables[dep_dataset].append((dep, dep_opendap))
                continue
            
            # Determine dataset for this variable
            var_dataset = dataset
            if var_dataset is None:
                if var_name in GEOS5FP_VARIABLES:
                    _, var_dataset, raw_variable = get_variable_info(var_name)
                else:
                    raise ValueError(
                        f"Dataset must be specified when using raw variable name '{var_name}'. "
                        f"Known variables: {list(GEOS5FP_VARIABLES.keys())}. "
                        f"Computed variables: {list(COMPUTED_VARIABLES)}"
                    )
            else:
                # Use provided dataset, determine variable
                if var_name in GEOS5FP_VARIABLES:
                    _, _, raw_variable = get_variable_info(var_name)
                else:
                    raw_variable = var_name
            
            variable_opendap = raw_variable.lower()
            dataset_to_variables[var_dataset].append((var_name, variable_opendap))
        
        logger.info(f"DEBUG: Finished grouping variables. About to import query_geos5fp_point_multi...")
        
        # Import multi-variable query function
        from GEOS5FP.GEOS5FP_point import query_geos5fp_point_multi
        
        # Calculate total query batches based on datasets and coordinates
        total_query_batches = 0
        for var_dataset in dataset_to_variables.keys():
            for coord_key in coord_to_records.keys():
                total_query_batches += estimated_batches_per_coord[coord_key]
        
        logger.info(f"DEBUG: Import successful. About to process {len(dataset_to_variables)} datasets...")
        logger.info(f"DEBUG: Datasets to process: {list(dataset_to_variables.keys())}")
        logger.info(f"DEBUG: Total query batches across all datasets: {total_query_batches}")
        logger.info(f"DEBUG: Starting dataset iteration loop...")
        
        # Initialize progress bar (disabled for now due to freezing issues)
        pbar = None
        
        # Process each dataset (querying all its variables together)
        for var_dataset, var_list in dataset_to_variables.items():
            logger.info(f"DEBUG: Entered loop, processing dataset: {var_dataset}")
            var_names_in_dataset = [v[0] for v in var_list]
            var_opendap_names = [v[1] for v in var_list]
            
            if verbose:
                if len(var_list) == 1:
                    logger.info(
                        f"Querying {var_names_in_dataset[0]} from {var_dataset} "
                        f"at {len(coord_to_records)} coordinates..."
                    )
                else:
                    logger.info(
                        f"Querying {len(var_list)} variables ({', '.join(var_names_in_dataset)}) "
                        f"from {var_dataset} at {len(coord_to_records)} coordinates..."
                    )
            
            # Query each unique coordinate with time clustering
            for coord_idx, (coord_key, records) in enumerate(coord_to_records.items(), 1):
                pt_lat, pt_lon = coord_key
                
                # Cluster times to keep queries manageable (1 day windows for reliability)
                time_clusters = cluster_times(records, max_days_per_query=1)
                
                if verbose:
                    logger.info(
                        f"  Coordinate {coord_idx}/{len(coord_to_records)}: "
                        f"({pt_lat:.4f}, {pt_lon:.4f}) - "
                        f"{len(records)} records in {len(time_clusters)} time clusters"
                    )
                
                # Query each time cluster
                for cluster_idx, cluster in enumerate(time_clusters, 1):
                    # Get time range for this cluster
                    times_in_cluster = [r['time'] for r in cluster]
                    min_time = min(times_in_cluster)
                    max_time = max(times_in_cluster)
                    time_span_days = (max_time - min_time).total_seconds() / 86400
                    
                    # Add buffer to ensure we get data
                    time_range_start = min_time - timedelta(hours=2)
                    time_range_end = max_time + timedelta(hours=2)
                    
                    # Calculate and display ETA (using completed_batches as the counter)
                    elapsed_time = current_time() - query_start_time
                    if verbose:
                        if completed_batches > 0:
                            avg_time_per_batch = elapsed_time / completed_batches
                            remaining_batches = total_query_batches - completed_batches
                            eta_seconds = avg_time_per_batch * remaining_batches
                            
                            # Format time as human-readable
                            if eta_seconds < 60:
                                eta_str = f"{eta_seconds:.0f}s"
                            elif eta_seconds < 3600:
                                eta_str = f"{eta_seconds/60:.1f}m"
                            else:
                                eta_str = f"{eta_seconds/3600:.1f}h"
                            
                            # Format elapsed time
                            if elapsed_time < 60:
                                elapsed_str = f"{elapsed_time:.0f}s"
                            elif elapsed_time < 3600:
                                elapsed_str = f"{elapsed_time/60:.1f}m"
                            else:
                                elapsed_str = f"{elapsed_time/3600:.1f}h"
                            
                            logger.info(
                                f"    Batch {completed_batches + 1}/{total_query_batches}: "
                                f"cluster {cluster_idx}/{len(time_clusters)} - "
                                f"{len(cluster)} records, {time_span_days:.1f} days "
                                f"({min_time.date()} to {max_time.date()}) - "
                                f"Elapsed: {elapsed_str}, ETA: {eta_str}"
                            )
                        else:
                            logger.info(
                                f"    Batch {completed_batches + 1}/{total_query_batches}: "
                                f"cluster {cluster_idx}/{len(time_clusters)} - "
                                f"{len(cluster)} records, {time_span_days:.1f} days "
                                f"({min_time.date()} to {max_time.date()})"
                            )
                    
                    # Update progress bar description to show current batch
                    if pbar is not None:
                        pbar.set_description(
                            # f"Querying batch {batch_num}/{total_query_batches} "
                            f"GEOS-5 FP {var_dataset} "
                            f"({pt_lat:.2f}, {pt_lon:.2f})"
                        )
                        pbar.refresh()
                    
                    try:
                        # Query all variables for this dataset in a single request
                        if verbose:
                            logger.info(
                                f"      Starting query for {len(var_opendap_names)} variable(s) "
                                f"at ({pt_lat:.4f}, {pt_lon:.4f})..."
                            )
                        
                        if len(var_opendap_names) == 1:
                            # Single variable - use original function
                            from GEOS5FP.GEOS5FP_point import query_geos5fp_point
                            result = query_geos5fp_point(
                                dataset=var_dataset,
                                variable=var_opendap_names[0],
                                lat=pt_lat,
                                lon=pt_lon,
                                time_range=(time_range_start, time_range_end),
                                dropna=dropna
                            )
                        else:
                            # Multiple variables - use multi-variable query
                            result = query_geos5fp_point_multi(
                                dataset=var_dataset,
                                variables=var_opendap_names,
                                lat=pt_lat,
                                lon=pt_lon,
                                time_range=(time_range_start, time_range_end),
                                dropna=dropna
                            )
                        
                        if verbose:
                            logger.info(f"      Query completed, retrieved {len(result.df)} time steps")
                        
                        if len(result.df) == 0:
                            logger.warning(
                                f"No data found for ({pt_lat}, {pt_lon}) "
                                f"in time range {time_range_start.date()} to {time_range_end.date()}"
                            )
                            # Set all records in this cluster to None for all variables
                            for record in cluster:
                                for var_name in var_names_in_dataset:
                                    results_by_index[record['index']][var_name] = None
                        else:
                            # Extract values for each needed time in this cluster
                            for record in cluster:
                                target_time = record['time']
                                
                                # Find closest available time
                                time_diffs = abs(result.df.index - target_time)
                                closest_idx = time_diffs.argmin()
                                
                                # Extract all variables
                                for var_name, var_opendap in zip(var_names_in_dataset, var_opendap_names):
                                    value = result.df.iloc[closest_idx][var_opendap]
                                    # Apply transformation if defined for this variable
                                    if var_name in VARIABLE_TRANSFORMATIONS:
                                        value = VARIABLE_TRANSFORMATIONS[var_name](value)
                                    results_by_index[record['index']][var_name] = value
                        
                        # Update completed batches counter for ETA
                        completed_batches += 1
                        
                        # Update progress bar if not verbose
                        if pbar is not None:
                            pbar.update(1)
                    except Exception as e:
                        if verbose:
                            logger.warning(
                                f"Failed to query variables at ({pt_lat}, {pt_lon}): {e}"
                            )
                        # Set all records in this cluster to None for all variables
                        for record in cluster:
                            for var_name in var_names_in_dataset:
                                results_by_index[record['index']][var_name] = None
                        
                        # Update completed batches counter even for failures
                        completed_batches += 1
                        
                        # Update progress bar if not verbose
                        if pbar is not None:
                            pbar.update(1)
        
        # Close progress bar if used
        if pbar is not None:
            pbar.close()
        
        # Convert results dictionary to list in original order
        all_dfs = [results_by_index[i] for i in range(len(parsed_times))]
        
        # Create DataFrame from results
        result_df = pd.DataFrame(all_dfs)
        
        # Set time as index
        result_df = result_df.set_index('time_UTC')
        
        # Move geometry to end
        if 'geometry' in result_df.columns:
            # Unwrap rasters geometry objects to shapely for GeoPandas compatibility
            result_df['geometry'] = result_df['geometry'].apply(
                lambda g: g.geometry if hasattr(g, 'geometry') else g
            )
            cols = [c for c in result_df.columns if c != 'geometry']
            cols.append('geometry')
            result_df = result_df[cols]
        
        # Convert to GeoDataFrame
        result_gdf = gpd.GeoDataFrame(result_df, geometry='geometry', crs='EPSG:4326')
        
        # Compute derived/computed variables
        if computed_variables:
            if verbose:
                logger.info(f"Computing derived variables: {', '.join(computed_variables)}")
            for var_name in computed_variables:
                try:
                    # Call the appropriate method for each computed variable
                    if var_name == 'RH':
                        # Compute RH from Q, PS, and Ta (for SVP)
                        if 'Q' in result_gdf.columns and 'PS' in result_gdf.columns and 'Ta' in result_gdf.columns:
                            # Import RH calculation utility
                            from GEOS5FP.calculate_RH import calculate_RH
                            
                            # Get the base variables
                            Q = result_gdf['Q'].values
                            PS = result_gdf['PS'].values
                            Ta_K = result_gdf['Ta'].values
                            
                            # Calculate RH
                            RH = calculate_RH(Q, PS, Ta_K)
                            
                            result_gdf['RH'] = RH
                        else:
                            logger.warning("Cannot compute RH: missing required variables (Q, PS, Ta)")
                            result_gdf['RH'] = None
                            
                    elif var_name == 'Ta_C':
                        # Convert Ta from Kelvin to Celsius
                        if 'Ta' in result_gdf.columns:
                            result_gdf['Ta_C'] = result_gdf['Ta'] - 273.15
                        else:
                            logger.warning("Cannot compute Ta_C: missing Ta")
                            result_gdf['Ta_C'] = None
                    
                    elif var_name == 'Tmin_C':
                        # Convert Tmin from Kelvin to Celsius
                        if 'Tmin' in result_gdf.columns:
                            result_gdf['Tmin_C'] = result_gdf['Tmin'] - 273.15
                        else:
                            logger.warning("Cannot compute Tmin_C: missing Tmin")
                            result_gdf['Tmin_C'] = None
                    
                    elif var_name == 'wind_speed_mps':
                        # Compute wind speed from U2M and V2M components
                        if 'U2M' in result_gdf.columns and 'V2M' in result_gdf.columns:
                            U2M = result_gdf['U2M'].values
                            V2M = result_gdf['V2M'].values
                            result_gdf['wind_speed_mps'] = np.sqrt(U2M**2 + V2M**2)
                        else:
                            logger.warning("Cannot compute wind_speed_mps: missing required variables (U2M, V2M)")
                            result_gdf['wind_speed_mps'] = None
                    
                    elif var_name == 'albedo_visible':
                        # Compute visible albedo scaling factor (ALBEDO / ALBVISDR)
                        if 'ALBEDO' in result_gdf.columns and 'ALBVISDR' in result_gdf.columns:
                            ALBEDO = result_gdf['ALBEDO'].values
                            ALBVISDR = result_gdf['ALBVISDR'].values
                            result_gdf['albedo_visible'] = ALBEDO / ALBVISDR
                        else:
                            logger.warning("Cannot compute albedo_visible: missing required variables (ALBEDO, ALBVISDR)")
                            result_gdf['albedo_visible'] = None
                    
                    elif var_name == 'albedo_NIR':
                        # Compute NIR albedo scaling factor (ALBEDO / ALBNIRDR)
                        if 'ALBEDO' in result_gdf.columns and 'ALBNIRDR' in result_gdf.columns:
                            ALBEDO = result_gdf['ALBEDO'].values
                            ALBNIRDR = result_gdf['ALBNIRDR'].values
                            result_gdf['albedo_NIR'] = ALBEDO / ALBNIRDR
                        else:
                            logger.warning("Cannot compute albedo_NIR: missing required variables (ALBEDO, ALBNIRDR)")
                            result_gdf['albedo_NIR'] = None
                    # Add more computed variables as needed
                except Exception as e:
                    logger.warning(f"Failed to compute {var_name}: {e}")
                    result_gdf[var_name] = None
            
            # Remove dependency columns that weren't originally requested
            cols_to_keep = ['geometry'] + variable_names
            cols_to_drop = [col for col in result_gdf.columns if col not in cols_to_keep]
            if cols_to_drop:
                result_gdf = result_gdf.drop(columns=cols_to_drop)
        
        # Ensure geometry column is at the end
        if 'geometry' in result_gdf.columns:
            cols = [c for c in result_gdf.columns if c != 'geometry']
            cols.append('geometry')
            result_gdf = result_gdf[cols]
        
        # Handle targets_df: merge results back into original table
        if targets_df is not None:
            # Drop time_UTC and geometry from result_gdf as they're already in targets_df
            result_cols = [c for c in result_gdf.columns if c not in ['time_UTC', 'geometry']]
            result_data = result_gdf[result_cols].reset_index(drop=True)
            
            # Add variable columns to targets_df
            for col in result_cols:
                targets_df[col] = result_data[col].values
            
            # Ensure geometry column is at the end
            if 'geometry' in targets_df.columns:
                cols = [c for c in targets_df.columns if c != 'geometry']
                cols.append('geometry')
                targets_df = targets_df[cols]
            
            return targets_df
        
        # For single-variable queries, return numpy array of values instead of GeoDataFrame
        if single_variable and len(variable_names) == 1:
            var_name = variable_names[0]
            if var_name in result_gdf.columns:
                return result_gdf[var_name].values
        
        return result_gdf
    
    # Create Point geometry from lat/lon if provided
    if lat is not None and lon is not None:
        if geometry is not None:
            raise ValueError("Cannot specify both geometry and lat/lon")
        geometry = Point(lon, lat)
    
    # Determine if this is a point query with time range
    is_point_time_series = (
        time_range is not None and 
        (is_point_geometry(geometry) or (lat is not None and lon is not None))
    )
    
    # Check if multiple variables requested for non-point query
    if not single_variable and not is_point_geometry(geometry):
        raise ValueError("Multiple variables can only be queried for point geometries")
    
    # Handle point time-series queries with OPeNDAP
    if is_point_time_series:
        if not HAS_OPENDAP_SUPPORT:
            raise ImportError(
                "Point query support requires xarray and netCDF4. "
                "Install with: conda install -c conda-forge xarray netcdf4"
            )
        
        # Extract point coordinates
        if geometry is not None:
            points = extract_points(geometry)
        else:
            points = [(lat, lon)]
        
        # Process each variable
        all_variable_dfs = []
        
        for var_name in variable_names:
            # Determine dataset for this variable
            var_dataset = dataset
            if var_dataset is None:
                if var_name in GEOS5FP_VARIABLES:
                    _, var_dataset, raw_variable = get_variable_info(var_name)
                else:
                    raise ValueError(
                        f"Dataset must be specified when using raw variable name '{var_name}'. "
                        f"Known variables: {list(GEOS5FP_VARIABLES.keys())}"
                    )
            else:
                # Use provided dataset, determine variable
                if var_name in GEOS5FP_VARIABLES:
                    _, _, raw_variable = get_variable_info(var_name)
                else:
                    raw_variable = var_name
            
            # Convert variable name to lowercase for OPeNDAP
            variable_opendap = raw_variable.lower()
            
            if verbose:
                logger.info(
                    f"retrieving {var_name} time-series "
                    f"from GEOS-5 FP {var_dataset} {raw_variable} "
                    f"for time range {time_range[0]} to {time_range[1]}"
                )
            
            # Determine if we should use temporal chunking
            # Chunk if: single point AND time range > 7 days
            time_span = (time_range[1] - time_range[0]).days
            use_temporal_chunking = len(points) == 1 and time_span > 7
            chunk_size_days = 7  # Chunk into 7-day periods
            
            if use_temporal_chunking:
                # Generate temporal chunks
                temporal_chunks = []
                current_chunk_start = time_range[0]
                while current_chunk_start < time_range[1]:
                    chunk_end = min(
                        current_chunk_start + timedelta(days=chunk_size_days),
                        time_range[1]
                    )
                    temporal_chunks.append((current_chunk_start, chunk_end))
                    current_chunk_start = chunk_end
                
                total_queries = len(temporal_chunks)
                query_unit = "chunk"
                if verbose:
                    logger.info(f"  Using temporal chunking: {total_queries} chunks of ~{chunk_size_days} days")
            else:
                # No temporal chunking - use spatial points
                temporal_chunks = [time_range]
                total_queries = len(points)
                query_unit = "point"
            
            # Initialize timing for ETA tracking
            from time import time as current_time
            query_start_time = current_time()
            completed_queries = 0
            
            # Create progress bar if not verbose
            # TODO: Fix progress bar freezing issues and re-enable
            pbar = None
            if not verbose:
                pbar = tqdm(
                    total=total_queries,
                    desc=f"Querying {var_name}",
                    unit=query_unit,
                    disable=True  # Disabled due to freezing issues
                )
            
            # Query data
            dfs = []
            
            if use_temporal_chunking:
                # Temporal chunking: iterate through time chunks for single point
                pt_lat, pt_lon = points[0]
                
                for chunk_idx, (chunk_start, chunk_end) in enumerate(temporal_chunks, 1):
                    # Calculate and display ETA
                    if verbose:
                        if completed_queries > 0:
                            elapsed_time = current_time() - query_start_time
                            avg_time_per_query = elapsed_time / completed_queries
                            remaining_queries = total_queries - completed_queries
                            eta_seconds = avg_time_per_query * remaining_queries
                            
                            # Format time as human-readable
                            if eta_seconds < 60:
                                eta_str = f"{eta_seconds:.0f}s"
                            elif eta_seconds < 3600:
                                eta_str = f"{eta_seconds/60:.1f}m"
                            else:
                                eta_str = f"{eta_seconds/3600:.1f}h"
                            
                            # Format elapsed time
                            if elapsed_time < 60:
                                elapsed_str = f"{elapsed_time:.0f}s"
                            elif elapsed_time < 3600:
                                elapsed_str = f"{elapsed_time/60:.1f}m"
                            else:
                                elapsed_str = f"{elapsed_time/3600:.1f}h"
                            
                            logger.info(
                                f"  Chunk {chunk_idx}/{total_queries} "
                                f"({chunk_start.date()} to {chunk_end.date()}) - "
                                f"Elapsed: {elapsed_str}, ETA: {eta_str}"
                            )
                        else:
                            logger.info(
                                f"  Chunk {chunk_idx}/{total_queries} "
                                f"({chunk_start.date()} to {chunk_end.date()})"
                            )
                    
                    try:
                        # Import locally to avoid scope issues
                        from GEOS5FP.GEOS5FP_point import query_geos5fp_point
                        
                        result = query_geos5fp_point(
                            dataset=var_dataset,
                            variable=var_opendap,
                            lat=pt_lat,
                            lon=pt_lon,
                            time_range=(chunk_start, chunk_end),
                            dropna=dropna
                        )
                        
                        if len(result.df) > 0:
                            df_chunk = result.df.copy()
                            
                            # Rename from OPeNDAP variable name to requested variable name
                            if variable_opendap in df_chunk.columns:
                                df_chunk = df_chunk.rename(columns={variable_opendap: var_name})
                            
                            # Add geometry column at the end
                            df_chunk['geometry'] = Point(pt_lon, pt_lat)
                            
                            dfs.append(df_chunk)
                        
                        # Update completed queries counter for ETA
                        completed_queries += 1
                        
                        # Update progress bar if not verbose
                        if pbar is not None:
                            pbar.update(1)
                    except Exception as e:
                        if verbose:
                            logger.warning(
                                f"Failed to query chunk {chunk_start.date()} to {chunk_end.date()} "
                                f"for {var_name}: {e}"
                            )
                        # Update completed queries counter even for failures
                        completed_queries += 1
                        
                        # Update progress bar if not verbose
                        if pbar is not None:
                            pbar.update(1)
            else:
                # No temporal chunking - iterate through spatial points
                for point_idx, (pt_lat, pt_lon) in enumerate(points, 1):
                    # Calculate and display ETA
                    if verbose:
                        if completed_queries > 0:
                            elapsed_time = current_time() - query_start_time
                            avg_time_per_query = elapsed_time / completed_queries
                            remaining_queries = total_queries - completed_queries
                            eta_seconds = avg_time_per_query * remaining_queries
                            
                            # Format time as human-readable
                            if eta_seconds < 60:
                                eta_str = f"{eta_seconds:.0f}s"
                            elif eta_seconds < 3600:
                                eta_str = f"{eta_seconds/60:.1f}m"
                            else:
                                eta_str = f"{eta_seconds/3600:.1f}h"
                            
                            # Format elapsed time
                            if elapsed_time < 60:
                                elapsed_str = f"{elapsed_time:.0f}s"
                            elif elapsed_time < 3600:
                                elapsed_str = f"{elapsed_time/60:.1f}m"
                            else:
                                elapsed_str = f"{elapsed_time/3600:.1f}h"
                            
                            logger.info(
                                f"  Point {point_idx}/{total_queries} ({pt_lat:.4f}, {pt_lon:.4f}) - "
                                f"Elapsed: {elapsed_str}, ETA: {eta_str}"
                            )
                        else:
                            logger.info(f"  Point {point_idx}/{total_queries} ({pt_lat:.4f}, {pt_lon:.4f})")
                    
                    try:
                        # Import locally to avoid scope issues
                        from GEOS5FP.GEOS5FP_point import query_geos5fp_point
                        
                        result = query_geos5fp_point(
                            dataset=var_dataset,
                            variable=var_opendap,
                            lat=pt_lat,
                            lon=pt_lon,
                            time_range=time_range,
                            dropna=dropna
                        )
                        
                        if len(result.df) == 0:
                            logger.warning(f"No data found for point ({pt_lat}, {pt_lon})")
                            continue
                        
                        df_point = result.df.copy()
                        
                        # Rename from OPeNDAP variable name to requested variable name
                        if variable_opendap in df_point.columns:
                            df_point = df_point.rename(columns={variable_opendap: var_name})
                        
                        # Add geometry column at the end
                        df_point['geometry'] = Point(pt_lon, pt_lat)
                        
                        dfs.append(df_point)
                        
                        # Update completed queries counter for ETA
                        completed_queries += 1
                        
                        # Update progress bar if not verbose
                        if pbar is not None:
                            pbar.update(1)
                    except Exception as e:
                        if verbose:
                            logger.warning(f"Failed to query point ({pt_lat}, {pt_lon}) for {var_name}: {e}")
                        # Update completed queries counter even for failures
                        completed_queries += 1
                        
                        # Update progress bar if not verbose
                        if pbar is not None:
                            pbar.update(1)
            
            # Close progress bar if used
            if pbar is not None:
                pbar.close()
            
            if not dfs:
                if verbose:
                    logger.warning(f"No successful point queries for variable {var_name}")
                continue
            
            # Concatenate all DataFrames for this variable
            if len(dfs) > 1:
                df_combined = pd.concat(dfs, axis=0).sort_index()
            else:
                df_combined = dfs[0]
            
            all_variable_dfs.append(df_combined)
        
        # Check if we got any results
        if not all_variable_dfs:
            if verbose:
                logger.warning("No successful point queries for any variable")
            # Return empty numpy array for single-variable queries, empty GeoDataFrame otherwise
            if single_variable:
                return np.array([])
            return gpd.GeoDataFrame()
        
        # Merge all variable DataFrames
        if len(all_variable_dfs) == 1:
            result_df = all_variable_dfs[0]
        else:
            # When using temporal interpolation, all variables should be at exact target time
            # For nearest, they may be at slightly different times, so we merge with tolerance
            
            # Start with first variable's dataframe
            result_df = all_variable_dfs[0].copy()
            
            # Merge each subsequent variable
            for var_df in all_variable_dfs[1:]:
                # Get the variable column name (exclude geometry)
                var_cols = [c for c in var_df.columns if c != 'geometry']
                
                if temporal_interpolation == "interpolate":
                    # Exact match on time index (all should be at target time_UTC)
                    result_df = result_df.merge(
                        var_df[var_cols],
                        left_index=True,
                        right_index=True,
                        how='outer'
                    )
                else:
                    # Nearest neighbor - use merge_asof to handle small time differences
                    result_df = pd.merge_asof(
                        result_df.sort_index(),
                        var_df[var_cols].sort_index(),
                        left_index=True,
                        right_index=True,
                        direction='nearest',
                        tolerance=pd.Timedelta(hours=2)
                    )
        
        # Convert to GeoDataFrame
        result_gdf = gpd.GeoDataFrame(result_df, geometry='geometry', crs='EPSG:4326')
        
        # For single-variable queries, return numpy array of values instead of GeoDataFrame
        if single_variable and len(variable_names) == 1:
            var_name = variable_names[0]
            if var_name in result_gdf.columns:
                return result_gdf[var_name].values
        
        return result_gdf
    
    # Handle single-time point queries (not time-series, not batch)
    # This is the case where: single time_UTC, Point geometry, no time_range
    if time_UTC is not None and is_point_geometry(geometry) and time_range is None:
        if not HAS_OPENDAP_SUPPORT:
            raise ImportError(
                "Point query support requires xarray and netCDF4. "
                "Install with: conda install -c conda-forge xarray netCDF4"
            )
        
        # Parse time if string
        if isinstance(time_UTC, str):
            target_time = parser.parse(time_UTC)
        else:
            target_time = time_UTC
        
        # Create a narrow time range around the target time (±2 hours)
        time_range_start = target_time - timedelta(hours=2)
        time_range_end = target_time + timedelta(hours=2)
        
        # Extract point coordinates
        # Handle rasters geometry wrapper
        if hasattr(geometry, 'geometry'):
            unwrapped_geom = geometry.geometry
        else:
            unwrapped_geom = geometry
        
        if isinstance(unwrapped_geom, Point):
            pt_lon, pt_lat = unwrapped_geom.x, unwrapped_geom.y
        elif isinstance(unwrapped_geom, MultiPoint):
            pt_lon, pt_lat = unwrapped_geom.geoms[0].x, unwrapped_geom.geoms[0].y
        else:
            raise ValueError(f"Unsupported point geometry: {type(geometry)}")
        
        # Query each variable
        all_variable_dfs = []
        
        for var_name in variable_names:
            # Determine dataset for this variable
            var_dataset = dataset
            if var_dataset is None:
                if var_name in GEOS5FP_VARIABLES:
                    _, var_dataset, raw_variable = get_variable_info(var_name)
                else:
                    raise ValueError(
                        f"Dataset must be specified when using raw variable name '{var_name}'. "
                        f"Known variables: {list(GEOS5FP_VARIABLES.keys())}"
                    )
            else:
                # Use provided dataset, determine variable
                if var_name in GEOS5FP_VARIABLES:
                    _, _, raw_variable = get_variable_info(var_name)
                else:
                    raw_variable = var_name
            
            # Convert variable name to lowercase for OPeNDAP
            variable_opendap = raw_variable.lower()
            
            if verbose:
                logger.info(
                    f"Retrieving {var_name} from GEOS-5 FP {var_dataset} {raw_variable} "
                    f"at ({pt_lat}, {pt_lon}) for time {target_time}"
                )
            
            try:
                from GEOS5FP.GEOS5FP_point import query_geos5fp_point
                
                result = query_geos5fp_point(
                    dataset=var_dataset,
                    variable=variable_opendap,
                    lat=pt_lat,
                    lon=pt_lon,
                    time_range=(time_range_start, time_range_end),
                    dropna=dropna
                )
                
                if len(result.df) == 0:
                    logger.warning(f"No data found for point ({pt_lat}, {pt_lon}) at time {target_time}")
                    # Return empty numpy array for single-variable queries, empty GeoDataFrame otherwise
                    if single_variable:
                        return np.array([])
                    return gpd.GeoDataFrame()
                
                # Find closest time to target
                time_diffs = abs(result.df.index - target_time)
                closest_idx = time_diffs.argmin()
                
                # Extract value at closest time
                df_point = result.df.iloc[[closest_idx]].copy()
                
                # Rename from OPeNDAP variable name to requested variable name
                if variable_opendap in df_point.columns:
                    df_point = df_point.rename(columns={variable_opendap: var_name})
                
                # Add geometry column
                df_point['geometry'] = Point(pt_lon, pt_lat)
                
                all_variable_dfs.append(df_point)
            except Exception as e:
                logger.warning(f"Failed to query {var_name} at ({pt_lat}, {pt_lon}): {e}")
                # Return empty numpy array for single-variable queries, empty GeoDataFrame otherwise
                if single_variable:
                    return np.array([])
                return gpd.GeoDataFrame()
        
        # Merge all variable DataFrames
        if not all_variable_dfs:
            if single_variable:
                return np.array([])
            return gpd.GeoDataFrame()
        
        if len(all_variable_dfs) == 1:
            result_df = all_variable_dfs[0]
        else:
            # Merge on time index
            result_df = all_variable_dfs[0].copy()
            for var_df in all_variable_dfs[1:]:
                var_cols = [c for c in var_df.columns if c != 'geometry']
                result_df = result_df.merge(
                    var_df[var_cols],
                    left_index=True,
                    right_index=True,
                    how='outer'
                )
        
        # Convert to GeoDataFrame
        result_gdf = gpd.GeoDataFrame(result_df, geometry='geometry', crs='EPSG:4326')
        
        # For single-variable queries, return numpy array of values instead of GeoDataFrame
        if single_variable and len(variable_names) == 1:
            var_name = variable_names[0]
            if var_name in result_gdf.columns:
                return result_gdf[var_name].values
        
        return result_gdf
    
    # If we get here, we need to handle raster queries or other cases
    # For now, raise an error to indicate this path is not yet fully implemented
    raise NotImplementedError(
        "Non-time-series queries for raster geometries are not yet fully implemented in this version. "
        "Please use time_range for point queries or provide proper implementation for raster queries."
    )
