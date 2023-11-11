
actions = {0: "right", 1: "up", 2: "left", 3: "down"}
def runPath(policy: list, start: tuple[float, float], goal: tuple[float, float], coordToIndex: dict, scale: float) -> dict:
    curr = start
    distance = 0
    path = []
    pathState = []
    history = []
    while curr != goal:
        currState = coordToIndex[curr[0]][curr[1]]
        pathState.append(currState)
        path.append(curr)
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
            return{"path": path, "pathState": pathState, "distance": distance}
        history.append(currState)
        if policy[currState] == 0:
            curr = (curr[0]+ scale, curr[1])
        if policy[currState] == 1:
            curr = (curr[0], curr[1] + scale)
        if policy[currState] == 2:
            curr = (curr[0] - scale, curr[1])
        if policy[currState] == 3:
            curr = (curr[0], curr[1] - scale)
        distance += scale
    return{"path": path, "pathState": pathState, "distance": distance}

def coordToPolicy(indexToCoord, policy) -> dict:
    d = {}
    for i in range(len(policy)):
        d[indexToCoord[i]] = policy[i]
    return d
    
    
