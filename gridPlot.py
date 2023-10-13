from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import math

map = Basemap()

map.drawcoastlines()
lonarr = []
latarr = []

latRange = (0, 15)
lonRange = (90, 105)
scale = .5




for lat in range(math.floor(latRange[0] / scale), math.ceil(latRange[1] / scale)):
    for lon in range(math.floor(lonRange[0] / scale), math.ceil(lonRange[1] / scale)):
        n = lon * scale
        t = lat * scale

        if not map.is_land(n, t):
            lonarr.append(n)
            latarr.append(t)
            

lons, lats = map(lonarr, latarr)

map.scatter(lons, lats, marker = 'o', color='r', zorder=1, s=1)


plt.show()
plt.savefig('test.png')