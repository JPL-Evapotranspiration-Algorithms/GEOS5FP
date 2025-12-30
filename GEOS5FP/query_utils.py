from typing import Union, List, Tuple
from datetime import datetime
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
from GEOS5FP.cluster_times import cluster_times
from GEOS5FP.GEOS5FP_point import query_geos5fp_point, query_geos5fp_point_multi
from GEOS5FP.calculate_RH import calculate_RH

# Define the extracted `query` function here
def query(
    target_variables: Union[str, List[str]] = None,
    targets_df: Union[pd.DataFrame, gpd.GeoDataFrame] = None,
    time_UTC: Union[datetime, str, List[datetime], List[str], pd.Series] = None,
    time_range: Tuple[Union[datetime, str], Union[datetime, str]] = None,
    dataset: str = None,
    geometry: Union[Point, MultiPoint, List, gpd.GeoSeries] = None,
    resampling: str = None,
    lat: Union[float, List[float], pd.Series] = None,
    lon: Union[float, List[float], pd.Series] = None,
    dropna: bool = True,
    temporal_interpolation: str = "interpolate",
    variable_name: Union[str, List[str]] = None,
    verbose: bool = False,
    **kwargs
):
    # Implementation of the query function
    pass