"""
Utility for clustering time records to optimize query batching.
"""
import pandas as pd

def cluster_times(records, max_days_per_query=30):
    """
    Cluster records by time to keep queries under max_days_per_query duration.
    
    Parameters
    ----------
    records : list of dict
        List of record dictionaries, each containing at minimum a 'time' key
        with a datetime value.
    max_days_per_query : int, default=30
        Maximum number of days to include in a single query batch.
    
    Returns
    -------
    list of list of dict
        List of record clusters, where each cluster is a list of records
        that can be queried together within the time constraint.
    
    Notes
    -----
    This function helps optimize batch queries by grouping records at the same
    coordinate into time clusters that don't exceed the specified maximum duration.
    This prevents excessively long time range queries that could timeout or
    return too much data.
    """
    if not records:
        return []
    
    # Ensure all times are datetime objects (convert from Timestamp if needed)
    for record in records:
        if isinstance(record['time'], pd.Timestamp):
            record['time'] = record['time'].to_pydatetime()
    
    # Sort by time
    sorted_records = sorted(records, key=lambda r: r['time'])
    
    clusters = []
    current_cluster = [sorted_records[0]]
    
    for record in sorted_records[1:]:
        cluster_start = current_cluster[0]['time']
        cluster_end = current_cluster[-1]['time']
        record_time = record['time']
        
        # Check if adding this record would exceed max duration
        potential_span = (max(cluster_end, record_time) - 
                        min(cluster_start, record_time))
        
        if potential_span.total_seconds() / 86400 <= max_days_per_query:
            # Add to current cluster
            current_cluster.append(record)
        else:
            # Start new cluster
            clusters.append(current_cluster)
            current_cluster = [record]
    
    # Add final cluster
    if current_cluster:
        clusters.append(current_cluster)
    
    return clusters
