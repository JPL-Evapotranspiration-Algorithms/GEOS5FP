from GEOS5FP import GEOS5FP
from rasters import Point

geos5fp = GEOS5FP()
AOT = geos5fp.AOT(time_UTC="2021-05-29 21:34:33", geometry=Point(-110.0522, 31.7438))
print("AOT:", AOT)
