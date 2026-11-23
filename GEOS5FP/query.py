from typing import Union, List, Tuple, Dict, Any
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import geopandas as gpd
from rasters import Raster, RasterGeometry
from shapely.geometry import Point, MultiPoint
from dateutil import parser
import logging

from .constants import *
from .geometry_utils import is_point_geometry, extract_points
from .get_variable_info import get_variable_info
from .calculate_RH import calculate_RH

logger = logging.getLogger(__name__)

# Optional import for point queries via OPeNDAP
try:
    from .GEOS5FP_point import query_geos5fp_point, query_geos5fp_point_multi
    HAS_OPENDAP_SUPPORT = True
except ImportError:
    HAS_OPENDAP_SUPPORT = False
    query_geos5fp_point = None
    query_geos5fp_point_multi = None


COMPUTED_DEPENDENCIES: Dict[str, List[str]] = {
    'RH': ['Q', 'PS', 'Ta'],
    'Ta_C': ['Ta'],
    'Tmin_C': ['Tmin'],
    'SVP_Pa': ['Ta'],
    'Ea_Pa': ['Q', 'PS', 'Ta'],
    'VPD_kPa': ['Q', 'PS', 'Ta'],
    'Td_K': ['Q', 'PS', 'Ta'],
    'Td_C': ['Q', 'PS', 'Ta'],
    'wind_speed_mps': ['U2M', 'V2M'],
    'PAR_proportion': ['ALBVISDR', 'ALBEDO'],
    'NIR_proportion': ['ALBNIRDR', 'ALBEDO'],
    'PAR_albedo': ['ALBVISDR', 'ALBVISDF'],
    'NIR_albedo': ['ALBNIRDR', 'ALBNIRDF'],
    'albedo_visible': ['ALBVISDR'],
    'albedo_NIR': ['ALBNIRDR'],
}


def _expand_requested_variables(variable_names: List[str]) -> Tuple[List[str], List[str]]:
    """
    Separate computed variables from base variables and expand all transitive
    base dependencies required to compute them.
    """
    computed_vars: List[str] = []
    base_vars: List[str] = []

    def _add_var(var_name: str):
        if var_name in COMPUTED_VARIABLES:
            if var_name not in computed_vars:
                computed_vars.append(var_name)
            for dep in COMPUTED_DEPENDENCIES.get(var_name, []):
                _add_var(dep)
        else:
            if var_name not in base_vars:
                base_vars.append(var_name)

    for v in variable_names:
        _add_var(v)

    return base_vars, computed_vars


def _evaluate_computed_variable(var_name: str, data: Union[pd.DataFrame, gpd.GeoDataFrame]) -> Any:
    """Evaluate a computed variable column on a DataFrame/GeoDataFrame containing base columns."""
    def _col(name: str) -> np.ndarray:
        return np.asarray(data[name].values, dtype=np.float64)

    if var_name == 'Ta_C':
        if 'Ta' not in data.columns:
            return None
        return _col('Ta') - 273.15

    if var_name == 'Tmin_C':
        if 'Tmin' not in data.columns:
            return None
        return _col('Tmin') - 273.15

    if var_name == 'SVP_Pa':
        if 'Ta' not in data.columns:
            return None
        ta_c = _col('Ta') - 273.15
        return 0.6108 * np.exp((17.27 * ta_c) / (ta_c + 237.3)) * 1000.0

    if var_name == 'RH':
        if not {'Q', 'PS', 'Ta'}.issubset(data.columns):
            return None
        return calculate_RH(_col('Q'), _col('PS'), _col('Ta'))

    if var_name == 'Ea_Pa':
        if not {'Q', 'PS', 'Ta'}.issubset(data.columns):
            return None
        rh = _evaluate_computed_variable('RH', data)
        svp = _evaluate_computed_variable('SVP_Pa', data)
        return rh * svp

    if var_name == 'VPD_kPa':
        if not {'Q', 'PS', 'Ta'}.issubset(data.columns):
            return None
        svp = _evaluate_computed_variable('SVP_Pa', data)
        ea = _evaluate_computed_variable('Ea_Pa', data)
        return np.clip(svp - ea, 0.0, None) / 1000.0

    if var_name == 'Td_K':
        if not {'Q', 'PS', 'Ta'}.issubset(data.columns):
            return None
        ta_k = _col('Ta')
        rh = _evaluate_computed_variable('RH', data)
        return ta_k - (100.0 - rh * 100.0) / 5.0

    if var_name == 'Td_C':
        td_k = _evaluate_computed_variable('Td_K', data)
        if td_k is None:
            return None
        return td_k - 273.15

    if var_name == 'wind_speed_mps':
        if not {'U2M', 'V2M'}.issubset(data.columns):
            return None
        return np.sqrt(_col('U2M') ** 2 + _col('V2M') ** 2)

    if var_name == 'PAR_proportion':
        if not {'ALBVISDR', 'ALBEDO'}.issubset(data.columns):
            return None
        albedo = _col('ALBEDO')
        return np.where(albedo != 0, _col('ALBVISDR') / albedo, np.nan)

    if var_name == 'NIR_proportion':
        if not {'ALBNIRDR', 'ALBEDO'}.issubset(data.columns):
            return None
        albedo = _col('ALBEDO')
        return np.where(albedo != 0, _col('ALBNIRDR') / albedo, np.nan)

    if var_name == 'PAR_albedo':
        if not {'ALBVISDR', 'ALBVISDF'}.issubset(data.columns):
            return None
        return _col('ALBVISDR') + _col('ALBVISDF')

    if var_name == 'NIR_albedo':
        if not {'ALBNIRDR', 'ALBNIRDF'}.issubset(data.columns):
            return None
        return _col('ALBNIRDR') + _col('ALBNIRDF')

    if var_name == 'albedo_visible':
        if 'ALBVISDR' not in data.columns:
            return None
        return _col('ALBVISDR')

    if var_name == 'albedo_NIR':
        if 'ALBNIRDR' not in data.columns:
            return None
        return _col('ALBNIRDR')

    return None


def _apply_computed_columns(
    df: Union[pd.DataFrame, gpd.GeoDataFrame],
    requested_variables: List[str],
    computed_variables: List[str],
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Compute requested derived variables and prune intermediate dependency columns."""
    if not computed_variables:
        return df

    for var_name in computed_variables:
        try:
            df[var_name] = _evaluate_computed_variable(var_name, df)
        except Exception as e:
            logger.warning(f"Failed to compute {var_name}: {e}")
            df[var_name] = None

    cols_to_keep = {'geometry', 'point_index'} | set(requested_variables)
    cols_to_drop = [col for col in df.columns if col not in cols_to_keep]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df


def _interpolate_series_at_time(
    df: pd.DataFrame,
    column: str,
    target_time: datetime,
    mode: str = TEMPORAL_INTERPOLATION,
) -> float:
    """
    Extract a scalar value from a time-indexed DataFrame at target_time using either
    linear temporal interpolation between bracketing timestamps or nearest-neighbor lookup.
    """
    if len(df) == 0 or column not in df.columns:
        return np.nan

    series = df[column].dropna()
    if len(series) == 0:
        return np.nan

    target_ts = pd.Timestamp(target_time)
    if target_ts.tzinfo is not None:
        target_ts = target_ts.tz_convert("UTC").tz_localize(None)

    series = series.sort_index()
    times = series.index
    values = np.asarray(series.values, dtype=np.float64)

    if mode != "interpolate" or len(times) == 1:
        closest_idx = np.abs(times - target_ts).argmin()
        return float(values[closest_idx])

    # Exact timestamp match
    exact_mask = times == target_ts
    if np.any(exact_mask):
        return float(values[np.where(exact_mask)[0][0]])

    # Find bracketing indices
    pos = int(times.searchsorted(target_ts))
    if pos == 0:
        return float(values[0])
    if pos >= len(times):
        return float(values[-1])

    t_before = times[pos - 1]
    t_after = times[pos]
    v_before = values[pos - 1]
    v_after = values[pos]

    total_span = (t_after - t_before).total_seconds()
    if total_span <= 0:
        return float(v_before)

    fraction = (target_ts - t_before).total_seconds() / total_span
    return float(v_before + (v_after - v_before) * fraction)


def _resolve_single_raster_variable(
    conn,
    var_name: str,
    time_UTC: Union[datetime, str],
    geometry: RasterGeometry = None,
    dataset: str = None,
    resampling: str = None,
    **kwargs,
) -> Raster:
    """Retrieve a single Raster for a known, raw, or computed variable."""
    if var_name in COMPUTED_VARIABLES and hasattr(conn, var_name):
        return getattr(conn, var_name)(
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling,
        )
    if var_name in GEOS5FP_VARIABLES:
        return conn.variable(
            variable_name=var_name,
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling,
            **kwargs,
        )
    if dataset is not None:
        parsed_time = parser.parse(time_UTC) if isinstance(time_UTC, str) else time_UTC
        return conn.interpolate(
            time_UTC=parsed_time,
            product=dataset,
            variable=var_name,
            geometry=geometry,
            resampling=resampling,
            **kwargs,
        )
    raise ValueError(
        f"Dataset must be specified when using raw variable name '{var_name}'. "
        f"Known variables: {list(GEOS5FP_VARIABLES.keys())}. "
        f"Computed variables: {list(COMPUTED_VARIABLES)}"
    )


def _fallback_single_point_via_interpolate(
    conn,
    var_name: str,
    dataset: str,
    target_time: datetime,
    pt_lat: float,
    pt_lon: float,
    **kwargs,
) -> float:
    """Retrieve a single scalar point value via HTTP granule interpolation when OPeNDAP fails."""
    if dataset is None:
        if var_name in GEOS5FP_VARIABLES:
            _, var_dataset, raw_variable = get_variable_info(var_name)
        else:
            raise ValueError(f"Dataset must be specified for raw variable '{var_name}'")
    else:
        var_dataset = dataset
        if var_name in GEOS5FP_VARIABLES:
            _, _, raw_variable = get_variable_info(var_name)
        else:
            raw_variable = var_name

    interpolated = conn.interpolate(
        time_UTC=target_time,
        product=var_dataset,
        variable=raw_variable,
        geometry=Point(pt_lon, pt_lat),
        resampling="nearest",
        **kwargs,
    )
    val = float(np.asarray(interpolated).reshape(-1)[0])
    if not np.isnan(val) and var_name in VARIABLE_TRANSFORMATIONS:
        val = float(VARIABLE_TRANSFORMATIONS[var_name](val))
    return val


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
        variables: Union[str, List[str]] = None,
        return_dict: bool = False,
        verbose: bool = False,
        **kwargs) -> Union[np.ndarray, Raster, Dict[str, Raster], gpd.GeoDataFrame]:
    """
    General-purpose query method that can retrieve any variable from any dataset.
    Supports both bound method calls (conn.query(...)) and standalone calls.
    """
    # Support calling either as conn.query(target_variables, ...) or query(target_variables, ...)
    if self is not None and not hasattr(self, "variable"):
        if (
            target_variables is not None
            and targets_df is None
            and not isinstance(target_variables, (str, list, tuple, set, np.ndarray, pd.Series))
        ):
            targets_df = target_variables
        target_variables = self
        self = None

    if verbose is None:
        verbose = VERBOSE

    # Handle backward compatibility and aliases: variable_name / variables -> target_variables
    if target_variables is None:
        if variable_name is not None:
            target_variables = variable_name
        elif variables is not None:
            target_variables = variables
        else:
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
    elif isinstance(target_variables, (list, tuple, set, np.ndarray, pd.Series)):
        variable_names = list(target_variables)
        if len(variable_names) == 0:
            raise ValueError("At least one variable must be specified")
        single_variable = len(variable_names) == 1
    else:
        raise TypeError(f"Unsupported type for target_variables: {type(target_variables)}")

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
            unwrapped_input_geom = geometry.geometry if hasattr(geometry, "geometry") else geometry
            if isinstance(geometry, gpd.GeoSeries):
                geometries = geometry.tolist()
            elif isinstance(geometry, list):
                geometries = geometry
            elif isinstance(unwrapped_input_geom, MultiPoint):
                pts = list(unwrapped_input_geom.geoms)
                if len(pts) == len(times):
                    geometries = pts
                elif len(pts) == 1:
                    geometries = [pts[0]] * len(times)
                else:
                    raise ValueError(
                        f"Number of points in MultiPoint ({len(pts)}) must match number of times ({len(times)})"
                    )
            elif is_point_geometry(geometry):
                geometries = [geometry] * len(times)
            else:
                raise ValueError("For batch queries with geometry, must provide GeoSeries, list, or Point/MultiPoint")
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

        if verbose:
            logger.info(
                f"Processing {len(parsed_times)} spatio-temporal records at "
                f"{len(coord_to_records)} unique coordinates..."
            )

        results_by_index = {
            i: {'time_UTC': t, 'geometry': g}
            for i, (t, g) in enumerate(zip(parsed_times, geometries))
        }

        base_variables, computed_variables = _expand_requested_variables(variable_names)
        dataset_to_variables = defaultdict(list)

        for var_name in base_variables:
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

        for var_dataset, var_list in dataset_to_variables.items():
            var_names_in_dataset = [v[0] for v in var_list]
            var_opendap_names = [v[1] for v in var_list]

            for coord_key, records in coord_to_records.items():
                pt_lat, pt_lon = coord_key
                time_clusters = cluster_times(records, max_days_per_query=1)

                for cluster in time_clusters:
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
                                for var_name, var_opendap in zip(var_names_in_dataset, var_opendap_names):
                                    value = _interpolate_series_at_time(
                                        result.df,
                                        var_opendap,
                                        target_time,
                                        mode=temporal_interpolation,
                                    )
                                    if not np.isnan(value) and var_name in VARIABLE_TRANSFORMATIONS:
                                        value = VARIABLE_TRANSFORMATIONS[var_name](value)
                                    results_by_index[record['index']][var_name] = None if np.isnan(value) else value

                    except Exception as e:
                        if verbose:
                            logger.warning(f"Failed to query variables at ({pt_lat}, {pt_lon}): {e}")
                        for record in cluster:
                            for var_name in var_names_in_dataset:
                                results_by_index[record['index']][var_name] = None

        all_dfs = [results_by_index[i] for i in range(len(parsed_times))]
        result_df = pd.DataFrame(all_dfs).set_index('time_UTC')

        if 'geometry' in result_df.columns:
            result_df['geometry'] = result_df['geometry'].apply(
                lambda g: g.geometry if hasattr(g, 'geometry') else g
            )
            cols = [c for c in result_df.columns if c != 'geometry'] + ['geometry']
            result_df = result_df[cols]

        result_gdf = gpd.GeoDataFrame(result_df, geometry='geometry', crs='EPSG:4326')
        result_gdf = _apply_computed_columns(result_gdf, variable_names, computed_variables)

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

    # Handle point time-series queries with OPeNDAP
    if is_point_time_series:
        if not HAS_OPENDAP_SUPPORT:
            raise ImportError(
                "Point query support requires xarray and netCDF4. "
                "Install with: conda install -c conda-forge xarray netcdf4"
            )

        points = extract_points(geometry) if geometry is not None else [(lat, lon)]
        multi_point = len(points) > 1
        base_variables, computed_variables = _expand_requested_variables(variable_names)
        all_variable_dfs = []

        for var_name in base_variables:
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
                            if var_name in VARIABLE_TRANSFORMATIONS:
                                df_chunk[var_name] = VARIABLE_TRANSFORMATIONS[var_name](df_chunk[var_name])
                            df_chunk['geometry'] = Point(pt_lon, pt_lat)
                            dfs.append(df_chunk)
                    except Exception as e:
                        if verbose:
                            logger.warning(f"Failed to query chunk {chunk_start.date()} to {chunk_end.date()} for {var_name}: {e}")
            else:
                for pt_idx, (pt_lat, pt_lon) in enumerate(points):
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
                        if var_name in VARIABLE_TRANSFORMATIONS:
                            df_point[var_name] = VARIABLE_TRANSFORMATIONS[var_name](df_point[var_name])
                        if multi_point:
                            df_point['point_index'] = pt_idx
                        df_point['geometry'] = Point(pt_lon, pt_lat)
                        dfs.append(df_point)
                    except Exception as e:
                        if verbose:
                            logger.warning(f"Failed to query point ({pt_lat}, {pt_lon}) for {var_name}: {e}")

            if not dfs:
                continue

            if len(dfs) > 1:
                df_combined = pd.concat(dfs, axis=0)
                if multi_point and 'point_index' in df_combined.columns:
                    df_combined = df_combined.reset_index().sort_values(['point_index', 'time']).set_index('time')
                else:
                    df_combined = df_combined.sort_index()
            else:
                df_combined = dfs[0]

            all_variable_dfs.append(df_combined)

        if not all_variable_dfs:
            return np.array([]) if (single_variable and not multi_point) else gpd.GeoDataFrame()

        result_df = all_variable_dfs[0].copy()
        for var_df in all_variable_dfs[1:]:
            var_cols = [c for c in var_df.columns if c not in ('geometry', 'point_index')]
            if multi_point and 'point_index' in result_df.columns and 'point_index' in var_df.columns:
                merged_parts = []
                for pt_idx, grp_left in result_df.groupby('point_index', sort=True):
                    grp_right = var_df[var_df['point_index'] == pt_idx][var_cols]
                    if temporal_interpolation == "interpolate":
                        part = grp_left.merge(grp_right, left_index=True, right_index=True, how='outer')
                        part['point_index'] = pt_idx
                        part['geometry'] = grp_left['geometry'].iloc[0]
                        for col in var_cols:
                            if col in part.columns:
                                part[col] = part[col].interpolate(method='time')
                    else:
                        part = pd.merge_asof(
                            grp_left.sort_index(),
                            grp_right.sort_index(),
                            left_index=True,
                            right_index=True,
                            direction='nearest',
                            tolerance=pd.Timedelta(hours=2)
                        )
                    merged_parts.append(part)
                result_df = pd.concat(merged_parts, axis=0)
            elif temporal_interpolation == "interpolate":
                result_df = result_df.merge(var_df[var_cols], left_index=True, right_index=True, how='outer')
                for col in var_cols:
                    if col in result_df.columns:
                        result_df[col] = result_df[col].interpolate(method='time')
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
        result_gdf = _apply_computed_columns(result_gdf, variable_names, computed_variables)

        if single_variable and not multi_point and len(variable_names) == 1 and variable_names[0] in result_gdf.columns:
            return result_gdf[variable_names[0]].values

        return result_gdf

    # Handle single-time point queries (Point or MultiPoint, no time_range)
    if time_UTC is not None and is_point_geometry(geometry) and time_range is None:
        allow_fallback = bool(self is not None and getattr(self, "allow_fallback", False))
        if not HAS_OPENDAP_SUPPORT and not allow_fallback:
            raise ImportError(
                "Point query support requires xarray and netCDF4. "
                "Install with: conda install -c conda-forge xarray netcdf4"
            )

        target_time = parser.parse(time_UTC) if isinstance(time_UTC, str) else time_UTC
        time_range_start = target_time - timedelta(hours=2)
        time_range_end = target_time + timedelta(hours=2)
        points = extract_points(geometry)
        base_variables, computed_variables = _expand_requested_variables(variable_names)

        all_variable_dfs = []
        for var_name in base_variables:
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
                value = np.nan
                opendap_succeeded = False

                if HAS_OPENDAP_SUPPORT:
                    try:
                        result = query_geos5fp_point(
                            dataset=var_dataset,
                            variable=variable_opendap,
                            lat=pt_lat,
                            lon=pt_lon,
                            time_range=(time_range_start, time_range_end),
                            dropna=dropna
                        )
                        if len(result.df) > 0:
                            value = _interpolate_series_at_time(
                                result.df,
                                variable_opendap,
                                target_time,
                                mode=temporal_interpolation,
                            )
                            if not np.isnan(value) and var_name in VARIABLE_TRANSFORMATIONS:
                                value = VARIABLE_TRANSFORMATIONS[var_name](value)
                            opendap_succeeded = not np.isnan(value)
                        else:
                            logger.warning(f"No data found for point ({pt_lat}, {pt_lon}) at time {target_time}")
                    except Exception as e:
                        logger.warning(f"Failed to query {var_name} at ({pt_lat}, {pt_lon}): {e}")

                if not opendap_succeeded and allow_fallback:
                    try:
                        value = _fallback_single_point_via_interpolate(
                            conn=self,
                            var_name=var_name,
                            dataset=var_dataset,
                            target_time=target_time,
                            pt_lat=pt_lat,
                            pt_lon=pt_lon,
                            **kwargs,
                        )
                        opendap_succeeded = not np.isnan(value)
                    except Exception as fallback_err:
                        logger.warning(f"Fallback interpolation failed for {var_name} at ({pt_lat}, {pt_lon}): {fallback_err}")

                if not opendap_succeeded:
                    continue

                df_point = pd.DataFrame(
                    {var_name: [value], 'geometry': [Point(pt_lon, pt_lat)]},
                    index=pd.DatetimeIndex([target_time], name="time"),
                )
                point_dfs.append(df_point)

            if not point_dfs:
                return np.array([]) if single_variable else gpd.GeoDataFrame()

            all_variable_dfs.append(pd.concat(point_dfs, axis=0))

        if not all_variable_dfs:
            return np.array([]) if single_variable else gpd.GeoDataFrame()

        result_df = all_variable_dfs[0].copy()
        for var_df in all_variable_dfs[1:]:
            var_cols = [c for c in var_df.columns if c != 'geometry']
            result_df[var_cols] = var_df[var_cols].values

        result_gdf = gpd.GeoDataFrame(result_df, geometry='geometry', crs='EPSG:4326')
        result_gdf = _apply_computed_columns(result_gdf, variable_names, computed_variables)

        if single_variable and len(variable_names) == 1 and variable_names[0] in result_gdf.columns:
            return result_gdf[variable_names[0]].values

        return result_gdf

    # Raster / non-point snapshot routing
    if time_UTC is not None and time_range is None:
        conn = self
        if conn is None:
            from .GEOS5FP_connection import GEOS5FPConnection
            conn = GEOS5FPConnection()

        if single_variable and not return_dict:
            return _resolve_single_raster_variable(
                conn=conn,
                var_name=variable_names[0],
                time_UTC=time_UTC,
                geometry=geometry,
                dataset=dataset,
                resampling=resampling,
                **kwargs,
            )
        else:
            return {
                var_name: _resolve_single_raster_variable(
                    conn=conn,
                    var_name=var_name,
                    time_UTC=time_UTC,
                    geometry=geometry,
                    dataset=dataset,
                    resampling=resampling,
                    **kwargs,
                )
                for var_name in variable_names
            }

    raise NotImplementedError(
        "Time-series queries for raster geometries are not supported; pass a Point/MultiPoint geometry or single time_UTC."
    )
