from GEOS5FP import GEOS5FP
from sentinel_tiles import sentinel_tiles

geos5fp = GEOS5FP()

geometry = sentinel_tiles.grid("11SPS")

timestamp = "2025-02-22 12:00:00"

Ta_C = geos5fp.Ta_C(time_UTC=timestamp, geometry=geometry)
print("Ta_C:", Ta_C)

SM = geos5fp.SM(time_UTC=timestamp, geometry=geometry)
print("SM:", SM)



