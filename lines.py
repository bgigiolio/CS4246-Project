from __future__ import annotations
from enum import Enum
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

def plotActions(map: Basemap, start: tuple[float, float], actions: list[int] = [], granularity: float = 0.5):
    print(len(actions))
    print(granularity)

    prevPoint = start
    for i in actions:
        if i == 0:
            start = (start[0] + granularity, start[1])    
        elif i == 1:
            start = (start[0] , start[1] + granularity)
        elif i == 2:
            start = (start[0] - granularity, start[1])
        elif i == 3:
            start = (start[0], start[1] - granularity)
        else:
            raise NotImplementedError("invalid action argument " +  str(i))
        
        map.plot([prevPoint[0], start[0]], [prevPoint[1], start[1]], color="b")
        prevPoint = start

plotActions(map, (-50,-50), [1,1,0,0,0], 5)

plt.show()