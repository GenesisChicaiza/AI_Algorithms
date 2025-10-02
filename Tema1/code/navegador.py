from problems import Problem



def __init__(self, initial, goal=None):
    grid = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
    ]

    self.initial = initial
    self.goal = goal


def actions(self, state):
    listadeactions = []
    for x in self.grid:
        for y in x:
            listadeactions.append((x,y))
    return listadeactions

def result(self,state,action):
    x,y = state
    match action:
        case "u":
            state = (x,y+1)
        case "d":
            state = (x,y-1)
        case "l":
            state = (x-1,y)
        case "r":
            state = (x+1,y)

    if state not in actions(state):
        state=x,y
        raise ValueError("retrasao, eres mas gil q geminais")
    
    return state




