# ETA Display Feature for Query Operations

## Overview

The `.query()` method now displays a running ETA (Estimated Time of Arrival/completion) each time it waits on a server request. This provides users with real-time feedback on query progress and expected completion time.

## Changes Made

### 1. Batch Spatio-Temporal Queries

When processing multiple points at different times, the query method now:
- Tracks the total number of query batches
- Calculates elapsed time and average time per batch
- Displays remaining estimated time before each server request
- Shows progress as: `Batch X/Y - Elapsed: Xs, ETA: Ys`

**Location**: Lines ~1800-1950 in `GEOS5FP_connection.py`

### 2. Time-Series Point Queries

For time-series queries at multiple points (single variable, time range):
- Tracks total number of points to query
- Calculates ETA based on completed points
- Displays progress for each point with elapsed time and ETA
- Shows: `Point X/Y (lat, lon) - Elapsed: Xs, ETA: Ys`

**Location**: Lines ~2155-2225 in `GEOS5FP_connection.py`

### 3. Multi-Variable Point Queries

For queries of multiple variables at a single time:
- Tracks total queries across all variables and points
- Shows variable progress: `retrieving varname (X/Y)`
- Displays point progress with ETA for each server request
- Shows: `Point X/Y (lat, lon) - Elapsed: Xs, ETA: Ys`

**Location**: Lines ~2410-2550 in `GEOS5FP_connection.py`

## Features

### Time Formatting
- Times under 60 seconds: displayed as seconds (e.g., `45s`)
- Times 60s to 3600s: displayed as minutes (e.g., `2.5m`)
- Times over 3600s: displayed as hours (e.g., `1.2h`)

### ETA Calculation
- Based on average time per completed request
- Only shown after first request completes (no ETA on first request)
- Updates dynamically as more requests complete
- Accounts for both successful and failed requests

### Progress Information
Each ETA display includes:
- Current position in the sequence (e.g., "Batch 3/10" or "Point 2/5")
- Coordinate information for point queries
- Elapsed time since query started
- Estimated time remaining to completion

## Example Output

### Batch Query Output
```
Processing 50 spatio-temporal records at 10 unique coordinates (15 query batches)...
Querying T2M from tavg1_2d_slv_Nx at 10 coordinates (15 batches)...
  Coordinate 1/10: (34.0500, -118.2500) - 5 records in 2 time clusters
    Batch 1/15: cluster 1/2 - 3 records, 2.0 days (2024-11-01 to 2024-11-03)
    Batch 2/15: cluster 2/2 - 2 records, 1.0 days (2024-11-15 to 2024-11-16) - Elapsed: 3.2s, ETA: 42s
  Coordinate 2/10: (35.0000, -119.0000) - 5 records in 1 time clusters
    Batch 3/15: cluster 1/1 - 5 records, 15.0 days (2024-11-01 to 2024-11-16) - Elapsed: 6.5s, ETA: 39s
...
```

### Time-Series Query Output
```
retrieving T2M time-series from GEOS-5 FP tavg1_2d_slv_Nx T2M for time range 2024-11-12 00:00:00 to 2024-11-15 00:00:00
  Point 1/5 (34.0500, -118.2500)
  Point 2/5 (35.0000, -119.0000) - Elapsed: 2.1s, ETA: 6s
  Point 3/5 (36.0000, -120.0000) - Elapsed: 4.3s, ETA: 5s
...
```

### Multi-Variable Query Output
```
retrieving T2M (1/3) from GEOS-5 FP tavg1_2d_slv_Nx at 3 point location(s)
  Point 1/3 (34.0500, -118.2500)
  Point 2/3 (37.7700, -122.4200) - Elapsed: 1.8s, ETA: 4s
  Point 3/3 (40.7100, -74.0000) - Elapsed: 3.7s, ETA: 2s
retrieving PS (2/3) from GEOS-5 FP tavg1_2d_slv_Nx at 3 point location(s)
  Point 1/3 (34.0500, -118.2500) - Elapsed: 5.5s, ETA: 6s
...
```

## Benefits

1. **User Feedback**: Users know the query is progressing and haven't experienced a hang
2. **Time Estimation**: Users can estimate how long large queries will take
3. **Planning**: Users can decide whether to wait or optimize their query
4. **Debugging**: Helps identify slow individual requests
5. **Transparency**: Makes it clear when the system is waiting on server responses vs processing data

## Testing

A test script has been created at `examples/test_eta_display.py` that demonstrates:
- Batch spatio-temporal queries with ETA
- Time-series queries with multiple points
- Multi-variable point queries

Run it with:
```bash
python examples/test_eta_display.py
```

## Technical Details

### Implementation
- Uses Python's `time.time()` for high-precision timing
- Calculates running average of completed requests
- Updates counter after both successful and failed requests
- No external dependencies required

### Performance Impact
- Minimal overhead (microseconds per update)
- Only logging operations added
- No network or disk I/O introduced
- ETA calculation is simple arithmetic

### Compatibility
- Fully backward compatible
- No changes to method signatures
- No changes to return values
- Existing code continues to work unchanged
