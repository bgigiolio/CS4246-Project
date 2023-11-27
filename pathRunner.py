import random
actions = {0: "right", 1: "up", 2: "left", 3: "down"}
def runPath(policy: list, start: tuple[float, float], goal: tuple[float, float], coordToIndex: dict, scale: float) -> dict:
    curr = (round(start[0], 4), round(start[1], 4))
    distance = 0
    path = []
    pathState = []
    history = []
    while curr != goal:
        try:
            currState = coordToIndex[curr[0]][curr[1]]
        except:
            currState = history.pop()
        pathState.append(currState)
        path.append(curr)
        choice = policy[currState]
        if currState in history:
            # print("Coords:")
            # print(path)
            # print("States:")
            # print(pathState)
            # c = 0
            # print("Path Followed: ")
            # for state in pathState:
            #     print(f"{path[c]}: {actions[policy[state]]}")
            #     c += 1
            # raise Exception("Path caught in loop!")
            c = random.randint(0, 1)
            if choice == 0 or choice == 2:
                if c == 0 and round((curr[1] + scale), 4) in coordToIndex[curr[0]].keys():
                    choice = 1
                else:
                    choice = 3
            else:
                if c == 0 and curr[1] in coordToIndex[round(curr[0]+ scale, 4)].keys():
                    choice = 0
                else:
                    choice = 2
        history.append(currState)
        if choice == 0:
            curr = (round(curr[0]+ scale, 4), curr[1])
        if choice == 1:
            curr = (curr[0], round((curr[1] + scale), 4))
        if choice == 2:
            curr = (round(curr[0] - scale, 4), curr[1])
        if choice == 3:
            curr = (curr[0], round((curr[1] - scale), 4))
        distance += scale
    return{"path": path, "pathState": pathState, "distance": distance}

def coordToPolicy(indexToCoord, policy) -> dict:
    d = {}
    for i in range(len(policy)):
        d[indexToCoord[i]] = policy[i]
    return d
    
    
