"""
Utility functions for working with geometry objects in GEOS-5 FP queries.
"""

from typing import List, Tuple
from shapely.geometry import Point, MultiPoint


def is_point_geometry(geometry) -> bool:
    """
    Check if geometry is a point or multipoint.
    
    Parameters
    ----------
    geometry : object
        Geometry to check (can be shapely Point/MultiPoint or rasters Point/MultiPoint)
    
    Returns
    -------
    bool
        True if point geometry, False otherwise
    
    Examples
    --------
    >>> from shapely.geometry import Point, Polygon
    >>> is_point_geometry(Point(0, 0))
    True
    >>> is_point_geometry(Polygon([(0,0), (1,0), (1,1), (0,1)]))
    False
    """
    if geometry is None:
        return False
    
    # Check shapely types
    if isinstance(geometry, (Point, MultiPoint)):
        return True
    
    # Check if it's a rasters geometry with point type
    if hasattr(geometry, 'geometry') and isinstance(geometry.geometry, (Point, MultiPoint)):
        return True
    
    # Check string representation
    geom_type = str(type(geometry).__name__).lower()
    if 'point' in geom_type:
        return True
        
    return False


def extract_points(geometry) -> List[Tuple[float, float]]:
    """
    Extract (lat, lon) coordinates from point geometry.
    
    Parameters
    ----------
    geometry : Point or MultiPoint
        Point or MultiPoint geometry (shapely or rasters)
    
    Returns
    -------
    list of tuple
        List of (lat, lon) tuples
    
    Raises
    ------
    ValueError
        If geometry type is not supported for point extraction
    
    Examples
    --------
    >>> from shapely.geometry import Point, MultiPoint
    >>> extract_points(Point(-118.25, 34.05))
    [(34.05, -118.25)]
    >>> extract_points(MultiPoint([(-118, 34), (-117, 33)]))
    [(34.0, -118.0), (33.0, -117.0)]
    
    Notes
    -----
    Shapely Point stores coordinates as (lon, lat) but this function
    returns them as (lat, lon) tuples for consistency with geographic conventions.
    """
    points = []
    
    # Handle rasters geometry wrapper
    if hasattr(geometry, 'geometry'):
        geom = geometry.geometry
    else:
        geom = geometry
    
    if isinstance(geom, Point):
        # Single point: (lon, lat) in shapely, return as (lat, lon)
        points.append((geom.y, geom.x))
    elif isinstance(geom, MultiPoint):
        # Multiple points
        for pt in geom.geoms:
            points.append((pt.y, pt.x))
    else:
        raise ValueError(f"Unsupported geometry type for point extraction: {type(geom)}")
    
    return points
