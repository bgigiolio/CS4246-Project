from __future__ import annotations
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import json

latitude = (-12, -8)
longitude = (86, 90)
map = Basemap(llcrnrlon=longitude[0], llcrnrlat=latitude[0], urcrnrlon=longitude[1], urcrnrlat=latitude[1])

map.drawcoastlines()

def moveState(curr_state: tuple[float, float], action: int, granularity: float):
    new_state = curr_state
    print(action)
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
    print(new_state)
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

    coords_to_index = {(j[0], j[1]) : i for i, j in coords.items()}
    print(coords_to_index)
    prevPoint = start
    for i in range(num_states):
        print(end)
        if prevPoint == end:
            break
        curr_index = coords_to_index[prevPoint]
        action = policyFunction[int(curr_index)]
        start = moveState(start, action, granularity)
        map.plot([prevPoint[0], start[0]], [prevPoint[1], start[1]], color="b", latlon=True)
        prevPoint = start

def mapUtility(map: Basemap, value_policy: dict[int, float]={}, index_to_coords: dict[int, tuple[float,float]] = {}):
    print(len(value_policy))
    print(len(index_to_coords))
    values = value_policy.values()
    max_value = max(values)
    min_value = min(values)

    for k, v in value_policy.items():
        coord = index_to_coords[k]
        curr_alpha = 0
        curr_color = "b"
        if v >=0:
            curr_alpha = v/max_value
            curr_color = "g"

        else:
            curr_alpha = v/min_value
            curr_color = "r"

        map.scatter(coord[0], coord[1], s=100, marker='s', color=curr_color, latlon=True, alpha=curr_alpha)
        

if __name__ == "__main__":
    f1 = open('riskMaps/(-12, -8)_(86, 90)_1/JSON.json')
    f2 = open('results/(86, 90)_(-12, -8)_1_VI/JSON.json')
    actions = json.load(f2)
    data = json.load(f1)
    # print(actions['policy'][1364])
    # plotActionsList(map, (-50,-70), [1,1,0,0,0], 5)
    plotActions(map, (86, -12), end=(float(data['goal'][0]),float(data['goal'][1])), coords=data['indexToCoord'], policyFunction=actions['policy'], granularity=1)
    plt.show()
    # print(actions.keys())
    # mapUtility(map, {0:1, 1:-1, 2:-0.5, 3:0.5}, {0:[-50,-70], 1:[-50.5,-70], 2:[-51, -70], 3:[-51.5, -70]})

    # plt.show()