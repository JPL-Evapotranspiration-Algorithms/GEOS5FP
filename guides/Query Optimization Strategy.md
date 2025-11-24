# Query Optimization Strategy

## Problem

When querying GEOS-5 FP data for many spatio-temporal records (e.g., flux tower observations), naive approaches make too many network requests:

**Naive approach**: Query each record individually
- 1,000 records × 3 variables = **3,000 OPeNDAP queries**
- Each query has network overhead (~1-2 seconds)
- Total time: 50-100 minutes

## Solution: Two-Level Optimization

### Level 1: Group by Unique Coordinates

**Insight**: Many records often share the same location (e.g., flux tower time-series at same site)

**Strategy**: 
- Identify unique (lat, lon) coordinates
- Query each coordinate once, not once per record

**Example**:
- 1,000 records at 50 unique sites
- Reduction: 1,000 → 50 queries per variable (95% reduction!)

### Level 2: Time Clustering

**Problem with Level 1 only**: 
- If one site has observations spanning 2 years, you'd query 2 years of hourly data
- That's ~17,000 timesteps when you only need 10 specific times!
- Slow download, wastes bandwidth, server strain

**Solution**: Cluster times at each coordinate
- Group nearby times together (default: max 30 days per cluster)
- Query each cluster separately
- Extract only the specific times needed

**Example**:
```
Site A observations:
  - 2019-06-15
  - 2019-06-20  
  - 2019-06-25  } Cluster 1: June 15-25 (10 days)
  - 2019-08-10  } Cluster 2: Aug 10 (single day)
  - 2021-04-01  } Cluster 3: Apr 1 (single day)

Instead of querying June 2019 - April 2021 (almost 2 years!),
make 3 shorter queries totaling 12 days
```

## Algorithm

```python
for each variable:
    for each unique coordinate:
        # Cluster times at this coordinate
        clusters = cluster_times(records_at_coordinate, max_days=30)
        
        for each cluster:
            # Query time range covering this cluster
            min_time = min(cluster_times)
            max_time = max(cluster_times)
            time_range = (min_time - 2hrs, max_time + 2hrs)
            
            # Single OPeNDAP query
            result = query_opendap(lat, lon, time_range)
            
            # Extract specific times from result
            for record in cluster:
                value = result.get_closest_time(record.time)
                store_result(record, value)
```

## Time Clustering Algorithm

```python
def cluster_times(records, max_days_per_query=30):
    """
    Group records by time, keeping each cluster under max_days_per_query.
    
    Sorts records by time, then walks through creating clusters.
    Starts new cluster when adding next record would exceed max duration.
    """
    sorted_records = sorted(records, key=lambda r: r.time)
    
    clusters = []
    current_cluster = [sorted_records[0]]
    
    for record in sorted_records[1:]:
        cluster_span = max(current_cluster + [record]).time - 
                      min(current_cluster + [record]).time
        
        if cluster_span <= max_days_per_query:
            current_cluster.append(record)
        else:
            clusters.append(current_cluster)
            current_cluster = [record]
    
    clusters.append(current_cluster)
    return clusters
```

## Performance Comparison

### Example Dataset: 1,065 flux tower observations

| Approach | Queries | Reduction | Time Estimate |
|----------|---------|-----------|---------------|
| **Naive (row-by-row)** | 3,195 | baseline | 90 minutes |
| **Coordinate grouping only** | 300 | 91% | 8 minutes |
| **With time clustering** | 450 | 86% | 12 minutes |

**Note**: Time clustering makes slightly more queries than coordinate-only, but:
- Avoids massive multi-year queries
- More predictable performance
- Respects server resources
- Better error handling (smaller blast radius)

### Example: 20 records over 2 years at 6 sites

**Dataset characteristics**:
- Total records: 20
- Unique coordinates: 6
- Time span: 2019-06-23 to 2021-06-20 (730 days)
- Some sites have 7 records spanning 75 days
- Some sites have single records

**Query counts**:
- Naive: 20 × 3 vars = **60 queries**
- Coordinate-only: 6 × 3 vars = **18 queries** (but some query 75+ days!)
- Time clustered: 8 batches × 3 vars = **24 queries** (all ≤ 30 days)

**Console output**:
```
[INFO] Processing 20 spatio-temporal records at 6 unique coordinates (8 query batches)...
[INFO]   Coordinate 4/6: (41.7727, -80.6313) - 5 records in 2 time clusters
[INFO]     Batch 4/8: cluster 1/2 - 4 records, 14.8 days (2019-06-23 to 2019-07-08)
[INFO]     Batch 5/8: cluster 2/2 - 1 records, 0.0 days (2019-08-25 to 2019-08-25)
```

Shows that coordinate 4 had 5 records, but they were split into:
- Cluster 1: 4 records in mid-June to July (14.8 days)
- Cluster 2: 1 record in late August (isolated)

This avoids querying the entire June-August span when there's a 1.5 month gap.

## Tuning Parameters

### `max_days_per_query` (default: 30)

Controls maximum time span for each query cluster.

**Lower values (e.g., 7 days)**:
- ✅ Smaller, faster queries
- ✅ Less data transfer
- ❌ More queries (more network overhead)
- **Best for**: Sparse observations across long periods

**Higher values (e.g., 90 days)**:
- ✅ Fewer queries
- ❌ Larger data transfers per query
- ❌ Slower individual queries
- **Best for**: Dense observations in continuous periods

**Default 30 days** balances:
- GEOS-5 FP data is hourly (720 timesteps/month)
- Typical observation frequency (weekly to daily)
- Network overhead vs. transfer time

### When to adjust

```python
# For sparse data (annual observations):
max_days_per_query = 7  # Keep queries very short

# For dense time-series (hourly observations):
max_days_per_query = 90  # Fewer, larger queries

# Modify in GEOS5FP_connection.py line ~1750:
clusters = cluster_times(records, max_days_per_query=30)  # Change this value
```

## Benefits Summary

✅ **Efficiency**: 85-95% reduction in queries vs. naive approach  
✅ **Predictable**: Each query bounded to reasonable time span  
✅ **Scalable**: Works for 10 records or 10,000 records  
✅ **Transparent**: Detailed logging shows what's happening  
✅ **Robust**: Individual failures don't cascade  
✅ **Respectful**: Doesn't overwhelm NASA's servers with huge requests  

## Implementation Notes

- Coordinate rounding to 6 decimal places (~0.1 meter precision)
- Time buffer of ±2 hours around query range (ensures data availability)
- Closest-time matching for each record (handles slight time misalignments)
- Results preserved in original record order
- Geometry preserved with each result
- Handles missing data gracefully (NaN values)

## Future Enhancements

Potential improvements:
1. **Parallel queries**: Use threading for concurrent requests
2. **Adaptive clustering**: Adjust `max_days_per_query` based on data density
3. **Caching**: Store results for repeated coordinate-time pairs
4. **Smart prefetching**: Anticipate likely next queries
5. **Progress bar**: Add visual progress indicator (tqdm)

The current implementation prioritizes correctness and transparency over maximum speed, making it suitable for research and operational use.
