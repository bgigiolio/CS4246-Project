from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

map = Basemap()

map.drawcoastlines()

#latitude (-90, 90)
#longitude (-180, 180)

x = [-100, -70, -30, 0, 30, 70, 100]
y = [20, -10, -30, -35, -30, -10, 20]

map.plot(x, y, latlon=True)
print(map.is_land(100, 20)) #land
print(map.is_land(0, -35)) #ocean
plt.show()
