from __future__ import annotations
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

map = Basemap()

map.drawcoastlines()

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
        
        map.plot([prevPoint[0], start[0]], [prevPoint[1], start[1]], color="b", latlon=True)
        prevPoint = start

plotActions(map, (-50,-50), [1,1,0,0,0], 5)

plt.show()