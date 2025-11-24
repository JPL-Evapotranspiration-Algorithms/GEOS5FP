"""
Utility functions for loading and working with GEOS-5 FP spatio-temporal data
"""

import pandas as pd
import geopandas as gpd
from shapely import wkt


def load_spatiotemporal_csv(filepath: str, time_column: str = 'time_UTC') -> gpd.GeoDataFrame:
    """
    Load a CSV file with WKT geometry into a GeoDataFrame.
    
    This function handles CSV files that have a 'geometry' column in WKT format
    (e.g., "POINT (lon lat)") and converts them to proper geometry objects.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file
    time_column : str, optional
        Name of the time column to parse as datetime (default: 'time_UTC')
    
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with parsed geometry and datetime
    
    Examples
    --------
    >>> gdf = load_spatiotemporal_csv('spatio_temporal.csv')
    >>> print(gdf.head())
    
    >>> # Access coordinates
    >>> lat = gdf.iloc[0].geometry.y
    >>> lon = gdf.iloc[0].geometry.x
    """
    # Load as regular DataFrame
    df = pd.read_csv(filepath)
    
    # Parse geometry column from WKT string to geometry objects
    if 'geometry' in df.columns:
        df['geometry'] = df['geometry'].apply(wkt.loads)
    
    # Parse time column if it exists
    if time_column in df.columns:
        df[time_column] = pd.to_datetime(df[time_column])
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    
    return gdf


def extract_lat_lon(gdf: gpd.GeoDataFrame, 
                     lat_column: str = 'lat', 
                     lon_column: str = 'lon') -> gpd.GeoDataFrame:
    """
    Extract latitude and longitude from geometry column into separate columns.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with geometry column
    lat_column : str, optional
        Name for latitude column (default: 'lat')
    lon_column : str, optional
        Name for longitude column (default: 'lon')
    
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with added lat/lon columns
    
    Examples
    --------
    >>> gdf = load_spatiotemporal_csv('spatio_temporal.csv')
    >>> gdf = extract_lat_lon(gdf)
    >>> print(gdf[['lat', 'lon']].head())
    """
    gdf = gdf.copy()
    gdf[lat_column] = gdf.geometry.y
    gdf[lon_column] = gdf.geometry.x
    return gdf


def summary_stats(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for a spatio-temporal GeoDataFrame.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with location and time data
    
    Returns
    -------
    pd.DataFrame
        Summary statistics including spatial and temporal extent
    
    Examples
    --------
    >>> gdf = load_spatiotemporal_csv('spatio_temporal.csv')
    >>> stats = summary_stats(gdf)
    >>> print(stats)
    """
    stats = {}
    
    # Spatial extent
    stats['num_records'] = len(gdf)
    stats['min_lat'] = gdf.geometry.y.min()
    stats['max_lat'] = gdf.geometry.y.max()
    stats['min_lon'] = gdf.geometry.x.min()
    stats['max_lon'] = gdf.geometry.x.max()
    
    # Temporal extent
    if 'time_UTC' in gdf.columns:
        time_col = pd.to_datetime(gdf['time_UTC'])
        stats['start_time'] = time_col.min()
        stats['end_time'] = time_col.max()
        stats['time_range_days'] = (time_col.max() - time_col.min()).days
    
    # Unique locations
    if 'ID' in gdf.columns:
        stats['num_unique_locations'] = gdf['ID'].nunique()
    
    return pd.Series(stats)
