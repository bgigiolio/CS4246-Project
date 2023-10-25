from __future__ import annotations
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

map = Basemap()

map.drawcoastlines()

def moveState(curr_state: tuple[float, float], action: int, granularity: float):
    new_state = curr_state
    if action == 0:
        new_state = (curr_state[0] + granularity, curr_state[1])    
    elif action == 1:
        new_state = (curr_state[0] , curr_state[1] + granularity)
    elif action == 2:
        new_state = (curr_state[0] - granularity, curr_state[1])
    elif action == 3:
        new_state = (curr_state[0], curr_state[1] - granularity)
    else:
        raise NotImplementedError("invalid action argument " +  str(action))
    
    return new_state
        

def plotActionsList(map: Basemap, start: tuple[float, float], actions: list[int] = [], granularity: float = 0.5):
    print(len(actions))
    print(granularity)

    prevPoint = start
    for i in actions:
        start = moveState(start, i, granularity)
        map.plot([prevPoint[0], start[0]], [prevPoint[1], start[1]], color="b", latlon=True)
        prevPoint = start

def plotActions(map: Basemap, start: tuple[float, float], end: tuple[float, float],
                 policyFunction: list[int] = [], coords: dict[int, tuple[float, float]] = {},  granularity: float = 0.5):
    num_policy = len(policyFunction)
    num_states = len(coords)
    print(num_policy)
    print(num_states)
    print(granularity)

    coords_to_index = {j : i for i, j in coords.items()}

    prevPoint = start
    for i in num_states:
        if prevPoint == end:
            break
        curr_index = coords_to_index[prevPoint]
        action = policyFunction[curr_index]
        start = moveState(start, action, granularity)
        map.plot([prevPoint[0], start[0]], [prevPoint[1], start[1]], color="b", latlon=True)
        prevPoint = start

plotActionsList(map, (-50,-50), [1,1,0,0,0], 5)

plt.show()