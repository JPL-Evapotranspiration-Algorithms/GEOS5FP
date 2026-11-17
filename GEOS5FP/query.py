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
        self=None,
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
    Supports both bound method calls (conn.query(...)) and standalone calls.
    """
    # Support calling either as conn.query(target_variables, ...) or query(target_variables, ...)
    if self is not None and not hasattr(self, "variable"):
        # Called as standalone function where first positional arg was target_variables
        if target_variables is not None and targets_df is None and not isinstance(target_variables, (str, list)):
            targets_df = target_variables
        target_variables = self
        self = None

    if verbose is None:
        verbose = VERBOSE

    # Handle backward compatibility: variable_name is deprecated, use target_variables
    if target_variables is None and variable_name is not None:
        target_variables = variable_name
    elif target_variables is None:
        raise ValueError("target_variables parameter is required")

    # Handle targets_df parameter
    if targets_df is not None:
        if 'time_UTC' not in targets_df.columns:
            raise ValueError("targets_df must contain 'time_UTC' column")
        if 'geometry' not in targets_df.columns:
            raise ValueError("targets_df must contain 'geometry' column")

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
        if isinstance(time_UTC, (list, tuple, pd.Series, np.ndarray)):
            is_batch_query = True
        elif lat is not None and lon is not None:
            if isinstance(lat, (list, pd.Series)) or isinstance(lon, (list, pd.Series)):
                is_batch_query = True
        elif geometry is not None and isinstance(geometry, gpd.GeoSeries):
            is_batch_query = True

    # Handle vectorized batch queries
    if is_batch_query:
        if not HAS_OPENDAP_SUPPORT:
            raise ImportError(
                "Point query support requires xarray and netCDF4. "
                "Install with: conda install -c conda-forge xarray netcdf4"
            )

        if isinstance(time_UTC, pd.Series):
            times = time_UTC.tolist()
        elif isinstance(time_UTC, (list, tuple)):
            times = list(time_UTC)
        elif isinstance(time_UTC, np.ndarray):
            times = time_UTC.tolist()
        else:
            raise ValueError("For batch queries, time_UTC must be a list, tuple, Series, or array")

        if geometry is not None:
            if isinstance(geometry, gpd.GeoSeries):
                geometries = geometry.tolist()
            elif isinstance(geometry, list):
                geometries = geometry
            elif is_point_geometry(geometry):
                geometries = [geometry] * len(times)
            else:
                raise ValueError("For batch queries with geometry, must provide GeoSeries, list, or single Point/MultiPoint")
        elif lat is not None and lon is not None:
            lats = lat.tolist() if isinstance(lat, pd.Series) else (lat if isinstance(lat, list) else [lat])
            lons = lon.tolist() if isinstance(lon, pd.Series) else (lon if isinstance(lon, list) else [lon])

            if len(lats) != len(lons):
                raise ValueError("lat and lon must have the same length")

            geometries = [Point(lon_val, lat_val) for lon_val, lat_val in zip(lons, lats)]
        else:
            raise ValueError("For batch queries, must provide geometry or lat/lon")

        if len(times) != len(geometries):
            raise ValueError(
                f"Number of times ({len(times)}) must match number of geometries ({len(geometries)})"
            )

        parsed_times = [parser.parse(t) if isinstance(t, str) else t for t in times]

        coord_to_records = {}
        for idx, (time_val, geom) in enumerate(zip(parsed_times, geometries)):
            unwrapped_geom = geom.geometry if hasattr(geom, 'geometry') else geom

            if isinstance(unwrapped_geom, Point):
                pt_lon, pt_lat = unwrapped_geom.x, unwrapped_geom.y
            elif isinstance(unwrapped_geom, MultiPoint):
                pt_lon, pt_lat = unwrapped_geom.geoms[0].x, unwrapped_geom.geoms[0].y
            else:
                raise ValueError(f"Unsupported geometry type (must be Point or MultiPoint): {type(geom)}")

            coord_key = (round(pt_lat, 6), round(pt_lon, 6))
            coord_to_records.setdefault(coord_key, []).append({
                'index': idx,
                'time': time_val,
                'geometry': geom
            })

        from GEOS5FP.cluster_times import cluster_times

        estimated_batches_per_coord = {
            coord_key: len(cluster_times(records, max_days_per_query=1))
            for coord_key, records in coord_to_records.items()
        }

        if verbose:
            logger.info(
                f"Processing {len(parsed_times)} spatio-temporal records at "
                f"{len(coord_to_records)} unique coordinates..."
            )

        from time import time as current_time
        query_start_time = current_time()
        completed_batches = 0

        results_by_index = {
            i: {'time_UTC': t, 'geometry': g}
            for i, (t, g) in enumerate(zip(parsed_times, geometries))
        }

        from collections import defaultdict
        dataset_to_variables = defaultdict(list)
        computed_variables = []

        computed_dependencies = {
            'RH': ['Q', 'PS', 'Ta'],
            'Ta_C': ['Ta'],
            'Tmin_C': ['Tmin'],
            'wind_speed_mps': ['U2M', 'V2M'],
            'albedo_visible': ['ALBVISDR'],
            'albedo_NIR': ['ALBNIRDR'],
        }

        for var_name in variable_names:
            if var_name in COMPUTED_VARIABLES:
                computed_variables.append(var_name)
                if var_name in computed_dependencies:
                    for dep in computed_dependencies[var_name]:
                        existing = [v[0] for v in sum(dataset_to_variables.values(), [])]
                        if dep not in variable_names and dep not in existing:
                            if dep in GEOS5FP_VARIABLES:
                                _, dep_dataset, dep_raw_variable = get_variable_info(dep)
                                dataset_to_variables[dep_dataset].append((dep, dep_raw_variable.lower()))
                continue

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
                if var_name in GEOS5FP_VARIABLES:
                    _, _, raw_variable = get_variable_info(var_name)
                else:
                    raw_variable = var_name

            dataset_to_variables[var_dataset].append((var_name, raw_variable.lower()))

        from GEOS5FP.GEOS5FP_point import query_geos5fp_point_multi

        total_query_batches = sum(
            estimated_batches_per_coord[coord_key]
            for _ in dataset_to_variables.keys()
            for coord_key in coord_to_records.keys()
        )

        for var_dataset, var_list in dataset_to_variables.items():
            var_names_in_dataset = [v[0] for v in var_list]
            var_opendap_names = [v[1] for v in var_list]

            for coord_idx, (coord_key, records) in enumerate(coord_to_records.items(), 1):
                pt_lat, pt_lon = coord_key
                time_clusters = cluster_times(records, max_days_per_query=1)

                for cluster_idx, cluster in enumerate(time_clusters, 1):
                    times_in_cluster = [r['time'] for r in cluster]
                    min_time = min(times_in_cluster)
                    max_time = max(times_in_cluster)
                    time_range_start = min_time - timedelta(hours=2)
                    time_range_end = max_time + timedelta(hours=2)

                    try:
                        if len(var_opendap_names) == 1:
                            result = query_geos5fp_point(
                                dataset=var_dataset,
                                variable=var_opendap_names[0],
                                lat=pt_lat,
                                lon=pt_lon,
                                time_range=(time_range_start, time_range_end),
                                dropna=dropna
                            )
                        else:
                            result = query_geos5fp_point_multi(
                                dataset=var_dataset,
                                variables=var_opendap_names,
                                lat=pt_lat,
                                lon=pt_lon,
                                time_range=(time_range_start, time_range_end),
                                dropna=dropna
                            )

                        if len(result.df) == 0:
                            for record in cluster:
                                for var_name in var_names_in_dataset:
                                    results_by_index[record['index']][var_name] = None
                        else:
                            for record in cluster:
                                target_time = record['time']
                                time_diffs = abs(result.df.index - target_time)
                                closest_idx = time_diffs.argmin()

                                for var_name, var_opendap in zip(var_names_in_dataset, var_opendap_names):
                                    value = result.df.iloc[closest_idx][var_opendap]
                                    if var_name in VARIABLE_TRANSFORMATIONS:
                                        value = VARIABLE_TRANSFORMATIONS[var_name](value)
                                    results_by_index[record['index']][var_name] = value

                        completed_batches += 1
                    except Exception as e:
                        if verbose:
                            logger.warning(f"Failed to query variables at ({pt_lat}, {pt_lon}): {e}")
                        for record in cluster:
                            for var_name in var_names_in_dataset:
                                results_by_index[record['index']][var_name] = None
                        completed_batches += 1

        all_dfs = [results_by_index[i] for i in range(len(parsed_times))]
        result_df = pd.DataFrame(all_dfs).set_index('time_UTC')

        if 'geometry' in result_df.columns:
            result_df['geometry'] = result_df['geometry'].apply(
                lambda g: g.geometry if hasattr(g, 'geometry') else g
            )
            cols = [c for c in result_df.columns if c != 'geometry'] + ['geometry']
            result_df = result_df[cols]

        result_gdf = gpd.GeoDataFrame(result_df, geometry='geometry', crs='EPSG:4326')

        if computed_variables:
            for var_name in computed_variables:
                try:
                    if var_name == 'RH':
                        if {'Q', 'PS', 'Ta'}.issubset(result_gdf.columns):
                            from GEOS5FP.calculate_RH import calculate_RH
                            result_gdf['RH'] = calculate_RH(
                                result_gdf['Q'].values,
                                result_gdf['PS'].values,
                                result_gdf['Ta'].values
                            )
                        else:
                            result_gdf['RH'] = None
                    elif var_name == 'Ta_C':
                        result_gdf['Ta_C'] = result_gdf['Ta'] - 273.15 if 'Ta' in result_gdf.columns else None
                    elif var_name == 'Tmin_C':
                        result_gdf['Tmin_C'] = result_gdf['Tmin'] - 273.15 if 'Tmin' in result_gdf.columns else None
                    elif var_name == 'wind_speed_mps':
                        if {'U2M', 'V2M'}.issubset(result_gdf.columns):
                            result_gdf['wind_speed_mps'] = np.sqrt(
                                result_gdf['U2M'].values ** 2 + result_gdf['V2M'].values ** 2
                            )
                        else:
                            result_gdf['wind_speed_mps'] = None
                    elif var_name == 'albedo_visible':
                        result_gdf['albedo_visible'] = (
                            result_gdf['ALBVISDR'].values if 'ALBVISDR' in result_gdf.columns else None
                        )
                    elif var_name == 'albedo_NIR':
                        result_gdf['albedo_NIR'] = (
                            result_gdf['ALBNIRDR'].values if 'ALBNIRDR' in result_gdf.columns else None
                        )
                except Exception as e:
                    logger.warning(f"Failed to compute {var_name}: {e}")
                    result_gdf[var_name] = None

            cols_to_keep = ['geometry'] + variable_names
            cols_to_drop = [col for col in result_gdf.columns if col not in cols_to_keep]
            if cols_to_drop:
                result_gdf = result_gdf.drop(columns=cols_to_drop)

        if 'geometry' in result_gdf.columns:
            cols = [c for c in result_gdf.columns if c != 'geometry'] + ['geometry']
            result_gdf = result_gdf[cols]

        if targets_df is not None:
            result_cols = [c for c in result_gdf.columns if c not in ['time_UTC', 'geometry']]
            result_data = result_gdf[result_cols].reset_index(drop=True)
            for col in result_cols:
                targets_df[col] = result_data[col].values
            if 'geometry' in targets_df.columns:
                cols = [c for c in targets_df.columns if c != 'geometry'] + ['geometry']
                targets_df = targets_df[cols]
            return targets_df

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

    is_point_time_series = (
        time_range is not None and
        (is_point_geometry(geometry) or (lat is not None and lon is not None))
    )

    if not single_variable and not is_point_geometry(geometry):
        raise ValueError("Multiple variables can only be queried for point geometries")

    # Handle point time-series queries with OPeNDAP
    if is_point_time_series:
        if not HAS_OPENDAP_SUPPORT:
            raise ImportError(
                "Point query support requires xarray and netCDF4. "
                "Install with: conda install -c conda-forge xarray netcdf4"
            )

        points = extract_points(geometry) if geometry is not None else [(lat, lon)]
        all_variable_dfs = []

        for var_name in variable_names:
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
                if var_name in GEOS5FP_VARIABLES:
                    _, _, raw_variable = get_variable_info(var_name)
                else:
                    raw_variable = var_name

            variable_opendap = raw_variable.lower()
            time_span = (time_range[1] - time_range[0]).days
            use_temporal_chunking = len(points) == 1 and time_span > 7
            chunk_size_days = 7

            if use_temporal_chunking:
                temporal_chunks = []
                current_chunk_start = time_range[0]
                while current_chunk_start < time_range[1]:
                    chunk_end = min(current_chunk_start + timedelta(days=chunk_size_days), time_range[1])
                    temporal_chunks.append((current_chunk_start, chunk_end))
                    current_chunk_start = chunk_end
            else:
                temporal_chunks = [time_range]

            dfs = []
            if use_temporal_chunking:
                pt_lat, pt_lon = points[0]
                for chunk_start, chunk_end in temporal_chunks:
                    try:
                        result = query_geos5fp_point(
                            dataset=var_dataset,
                            variable=variable_opendap,
                            lat=pt_lat,
                            lon=pt_lon,
                            time_range=(chunk_start, chunk_end),
                            dropna=dropna
                        )
                        if len(result.df) > 0:
                            df_chunk = result.df.copy()
                            if variable_opendap in df_chunk.columns:
                                df_chunk = df_chunk.rename(columns={variable_opendap: var_name})
                            df_chunk['geometry'] = Point(pt_lon, pt_lat)
                            dfs.append(df_chunk)
                    except Exception as e:
                        if verbose:
                            logger.warning(f"Failed to query chunk {chunk_start.date()} to {chunk_end.date()} for {var_name}: {e}")
            else:
                for pt_lat, pt_lon in points:
                    try:
                        result = query_geos5fp_point(
                            dataset=var_dataset,
                            variable=variable_opendap,
                            lat=pt_lat,
                            lon=pt_lon,
                            time_range=time_range,
                            dropna=dropna
                        )
                        if len(result.df) == 0:
                            continue
                        df_point = result.df.copy()
                        if variable_opendap in df_point.columns:
                            df_point = df_point.rename(columns={variable_opendap: var_name})
                        df_point['geometry'] = Point(pt_lon, pt_lat)
                        dfs.append(df_point)
                    except Exception as e:
                        if verbose:
                            logger.warning(f"Failed to query point ({pt_lat}, {pt_lon}) for {var_name}: {e}")

            if not dfs:
                continue

            df_combined = pd.concat(dfs, axis=0).sort_index() if len(dfs) > 1 else dfs[0]
            all_variable_dfs.append(df_combined)

        if not all_variable_dfs:
            return np.array([]) if single_variable else gpd.GeoDataFrame()

        result_df = all_variable_dfs[0].copy()
        for var_df in all_variable_dfs[1:]:
            var_cols = [c for c in var_df.columns if c != 'geometry']
            if temporal_interpolation == "interpolate":
                result_df = result_df.merge(var_df[var_cols], left_index=True, right_index=True, how='outer')
            else:
                result_df = pd.merge_asof(
                    result_df.sort_index(),
                    var_df[var_cols].sort_index(),
                    left_index=True,
                    right_index=True,
                    direction='nearest',
                    tolerance=pd.Timedelta(hours=2)
                )

        result_gdf = gpd.GeoDataFrame(result_df, geometry='geometry', crs='EPSG:4326')
        if single_variable and len(variable_names) == 1 and variable_names[0] in result_gdf.columns:
            return result_gdf[variable_names[0]].values

        return result_gdf

    # Handle single-time point queries (Point or MultiPoint, no time_range)
    if time_UTC is not None and is_point_geometry(geometry) and time_range is None:
        if not HAS_OPENDAP_SUPPORT:
            raise ImportError(
                "Point query support requires xarray and netCDF4. "
                "Install with: conda install -c conda-forge xarray netcdf4"
            )

        target_time = parser.parse(time_UTC) if isinstance(time_UTC, str) else time_UTC
        time_range_start = target_time - timedelta(hours=2)
        time_range_end = target_time + timedelta(hours=2)
        points = extract_points(geometry)

        all_variable_dfs = []
        for var_name in variable_names:
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
                if var_name in GEOS5FP_VARIABLES:
                    _, _, raw_variable = get_variable_info(var_name)
                else:
                    raw_variable = var_name

            variable_opendap = raw_variable.lower()
            point_dfs = []

            for pt_lat, pt_lon in points:
                try:
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
                        continue

                    time_diffs = abs(result.df.index - target_time)
                    closest_idx = time_diffs.argmin()
                    df_point = result.df.iloc[[closest_idx]].copy()

                    if variable_opendap in df_point.columns:
                        df_point = df_point.rename(columns={variable_opendap: var_name})

                    df_point['geometry'] = Point(pt_lon, pt_lat)
                    point_dfs.append(df_point)
                except Exception as e:
                    logger.warning(f"Failed to query {var_name} at ({pt_lat}, {pt_lon}): {e}")

            if not point_dfs:
                return np.array([]) if single_variable else gpd.GeoDataFrame()

            all_variable_dfs.append(pd.concat(point_dfs, axis=0))

        if not all_variable_dfs:
            return np.array([]) if single_variable else gpd.GeoDataFrame()

        result_df = all_variable_dfs[0].copy()
        for var_df in all_variable_dfs[1:]:
            var_cols = [c for c in var_df.columns if c != 'geometry']
            result_df = result_df.merge(var_df[var_cols], left_index=True, right_index=True, how='outer')

        result_gdf = gpd.GeoDataFrame(result_df, geometry='geometry', crs='EPSG:4326')
        if single_variable and len(variable_names) == 1 and variable_names[0] in result_gdf.columns:
            return result_gdf[variable_names[0]].values

        return result_gdf

    # Single-time raster / non-point snapshot routing
    if time_UTC is not None and time_range is None and single_variable:
        var_name = variable_names[0]
        if self is not None:
            if var_name in GEOS5FP_VARIABLES:
                return self.variable(
                    variable_name=var_name,
                    time_UTC=time_UTC,
                    geometry=geometry,
                    resampling=resampling,
                    **kwargs
                )
            elif dataset is not None:
                parsed_time = parser.parse(time_UTC) if isinstance(time_UTC, str) else time_UTC
                return self.interpolate(
                    time_UTC=parsed_time,
                    product=dataset,
                    variable=var_name,
                    geometry=geometry,
                    resampling=resampling,
                    **kwargs
                )
            elif var_name in COMPUTED_VARIABLES and hasattr(self, var_name):
                return getattr(self, var_name)(
                    time_UTC=time_UTC,
                    geometry=geometry,
                    resampling=resampling,
                    **kwargs
                )

    raise NotImplementedError(
        "Non-time-series queries for raster geometries require an active GEOS5FPConnection instance "
        "and a known variable name or explicit dataset."
    )
