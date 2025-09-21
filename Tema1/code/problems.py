import random
import tkinter as tk

import numpy as np


def is_in(elt, seq):
    """Similar to (elt in seq), but compares with 'is', not '=='."""
    return any(x is elt for x in seq)


def distance(a, b):
    """The distance between two (x, y) points."""
    xA, yA = a
    xB, yB = b
    return np.hypot((xA - xB), (yA - yB))


class Problem:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return any(x is state for x in self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


class MazeProblem(Problem):
    def __init__(self, width, height, initial, goal=None):
        super().__init__(initial, goal)
        self.width = width
        self.height = height
        self.maze = self.generate_maze()

    def generate_maze(self):
        # 1s represent walls and 0s represent paths
        maze = [[0 for _ in range(self.width)] for _ in range(self.height)]

        # Randomly add walls to the maze
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) != self.initial and (i, j) != self.goal:
                    maze[i][j] = 1 if random.random() < 0.3 else 0

        return maze

    def actions(self, state):
        actions = []
        x, y = state

        if x > 0 and self.maze[x - 1][y] == 0:  # Up
            actions.append("Up")
        if x < self.height - 1 and self.maze[x + 1][y] == 0:  # Down
            actions.append("Down")
        if y > 0 and self.maze[x][y - 1] == 0:  # Left
            actions.append("Left")
        if y < self.width - 1 and self.maze[x][y + 1] == 0:  # Right
            actions.append("Right")

        return actions

    def result(self, state, action):
        x, y = state
        if action == "Up":
            return (x - 1, y)
        elif action == "Down":
            return (x + 1, y)
        elif action == "Left":
            return (x, y - 1)
        elif action == "Right":
            return (x, y + 1)

    def path_cost(self, c, state1, action, state2):
        return c + 1


class MazeVisualizer(tk.Tk):
    def __init__(self, problem, delay=100):
        super().__init__()
        self.problem = problem
        self.delay = delay
        self.previous_path = None
        self.title("Maze Problem")
        self.canvas = tk.Canvas(
            self, width=problem.width * 20, height=problem.height * 20
        )
        self.canvas.pack()
        self.draw_maze()
        self.after(500)

    def draw_maze(self):
        for i in range(self.problem.height):
            for j in range(self.problem.width):
                color = "white" if self.problem.maze[i][j] == 0 else "black"
                self.canvas.create_rectangle(
                    j * 20, i * 20, (j + 1) * 20, (i + 1) * 20, fill=color
                )

        self.canvas.create_rectangle(
            self.problem.initial[1] * 20,
            self.problem.initial[0] * 20,
            (self.problem.initial[1] + 1) * 20,
            (self.problem.initial[0] + 1) * 20,
            fill="blue",
        )
        self.canvas.create_rectangle(
            self.problem.goal[1] * 20,
            self.problem.goal[0] * 20,
            (self.problem.goal[1] + 1) * 20,
            (self.problem.goal[0] + 1) * 20,
            fill="blue",
        )

    def draw_current_path(self, states_path):
        if self.previous_path:
            self._update_state(self.previous_path, "visited")
        self._update_state(states_path, "current")
        self.previous_path = states_path

    def draw_solution(self, states_path):
        if self.previous_path:
            self._update_state(self.previous_path, "visited")
        self._update_state(states_path, "solution")
        self.previous_path = states_path

    def _update_state(self, states_path, state_type):
        if state_type == "visited":
            color = "yellow"
        elif state_type == "current":
            color = "red"
        elif state_type == "solution":
            color = "green"
        else:
            raise ValueError("Invalid state_type")
        for state in states_path:
            self.canvas.create_rectangle(
                state[1] * 20,
                state[0] * 20,
                (state[1] + 1) * 20,
                (state[0] + 1) * 20,
                fill=color,
            )
        self.update_idletasks()
        self.after(self.delay)


class EightPuzzle(Problem):
    """The problem of sliding tiles numbered from 1 to 8 on a 3x3 board, where one of the
    squares is a blank. A state is represented as a tuple of length 9, where  element at
    index i represents the tile number  at index i (0 if it's an empty square)"""

    def __init__(self, initial=None, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
        """Define goal state and initialize a problem"""
        if initial is None:
            solvable = False
            max_attemps = 20
            for _ in range(max_attemps):
                # generate a random permutation of 0 to 8
                numbers = list(range(9))
                random.shuffle(numbers)
                initial = tuple(numbers)
                if self.check_solvability(initial):
                    solvable = True
                    break
        if not solvable:
            raise ValueError("Could not generate a solvable puzzle")

        super().__init__(initial, goal)

    def find_blank_square(self, state):
        """Return the index of the blank square in a given state"""
        return state.index(0)

    def actions(self, state):
        """Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment"""

        possible_actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        index_blank_square = self.find_blank_square(state)

        if index_blank_square % 3 == 0:
            possible_actions.remove("LEFT")
        if index_blank_square < 3:
            possible_actions.remove("UP")
        if index_blank_square % 3 == 2:
            possible_actions.remove("RIGHT")
        if index_blank_square > 5:
            possible_actions.remove("DOWN")

        return possible_actions

    def result(self, state, action):
        """Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state"""

        # blank is the index of the blank square
        blank = self.find_blank_square(state)
        new_state = list(state)

        delta = {"UP": -3, "DOWN": 3, "LEFT": -1, "RIGHT": 1}
        neighbor = blank + delta[action]
        new_state[blank], new_state[neighbor] = new_state[neighbor], new_state[blank]

        return tuple(new_state)

    def goal_test(self, state):
        """Given a state, return True if state is a goal state or False, otherwise"""

        return state == self.goal

    def check_solvability(self, state):
        """Checks if the given state is solvable"""

        inversion = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if (state[i] > state[j]) and state[i] != 0 and state[j] != 0:
                    inversion += 1

        return inversion % 2 == 0

    def h(self, node):
        """Return the heuristic value for a given state. Default heuristic function used is
        h(n) = number of misplaced tiles"""

        return sum(s != g for (s, g) in zip(node.state, self.goal))


class Graph:
    """A graph connects nodes (vertices) by edges (links). Each edge can also
    have a length associated with it. The constructor call is something like:
        g = Graph({'A': {'B': 1, 'C': 2})
    this makes a graph with 3 nodes, A, B, and C, with an edge of length 1 from
    A to B,  and an edge of length 2 from A to C. You can also do:
        g = Graph({'A': {'B': 1, 'C': 2}, directed=False)
    This makes an undirected graph, so inverse links are also added. The graph
    stays undirected; if you add more links with g.connect('B', 'C', 3), then
    inverse link is also added. You can use g.nodes() to get a list of nodes,
    g.get('A') to get a dict of links out of A, and g.get('A', 'B') to get the
    length of the link from A to B. 'Lengths' can actually be any object at
    all, and nodes can be any hashable object."""

    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    def make_undirected(self):
        """Make a digraph into an undirected graph by adding symmetric edges."""
        for a in list(self.graph_dict.keys()):
            for b, dist in self.graph_dict[a].items():
                self.connect1(b, a, dist)

    def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed:
            self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        """Add a link from A to B of given distance, in one direction only."""
        self.graph_dict.setdefault(A, {})[B] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):
        """Return a list of nodes in the graph."""
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)


def UndirectedGraph(graph_dict=None):
    """Build a Graph where every edge (including future ones) goes both ways."""
    return Graph(graph_dict=graph_dict, directed=False)


def RandomGraph(
    nodes=list(range(10)),
    min_links=2,
    width=400,
    height=300,
    curvature=lambda: random.uniform(1.1, 1.5),
):
    """Construct a random graph, with the specified nodes, and random links.
    The nodes are laid out randomly on a (width x height) rectangle.
    Then each node is connected to the min_links nearest neighbors.
    Because inverse links are added, some nodes will have more connections.
    The distance between nodes is the hypotenuse times curvature(),
    where curvature() defaults to a random number between 1.1 and 1.5."""
    g = UndirectedGraph()
    g.locations = {}
    # Build the cities
    for node in nodes:
        g.locations[node] = (random.randrange(width), random.randrange(height))
    # Build roads from each city to at least min_links nearest neighbors.
    for i in range(min_links):
        for node in nodes:
            if len(g.get(node)) < min_links:
                here = g.locations[node]

                def distance_to_node(n):
                    if n is node or g.get(node, n):
                        return np.inf
                    return distance(g.locations[n], here)

                neighbor = min(nodes, key=distance_to_node)
                d = distance(g.locations[neighbor], here) * curvature()
                g.connect(node, neighbor, int(d))
    return g


"""
Simplified road map of Romania
"""
romania_map = UndirectedGraph(
    dict(
        Arad=dict(Zerind=75, Sibiu=140, Timisoara=118),
        Bucharest=dict(Urziceni=85, Pitesti=101, Giurgiu=90, Fagaras=211),
        Craiova=dict(Drobeta=120, Rimnicu=146, Pitesti=138),
        Drobeta=dict(Mehadia=75),
        Eforie=dict(Hirsova=86),
        Fagaras=dict(Sibiu=99),
        Hirsova=dict(Urziceni=98),
        Iasi=dict(Vaslui=92, Neamt=87),
        Lugoj=dict(Timisoara=111, Mehadia=70),
        Oradea=dict(Zerind=71, Sibiu=151),
        Pitesti=dict(Rimnicu=97),
        Rimnicu=dict(Sibiu=80),
        Urziceni=dict(Vaslui=142),
    )
)
romania_map.locations = dict(
    Arad=(91, 492),
    Bucharest=(400, 327),
    Craiova=(253, 288),
    Drobeta=(165, 299),
    Eforie=(562, 293),
    Fagaras=(305, 449),
    Giurgiu=(375, 270),
    Hirsova=(534, 350),
    Iasi=(473, 506),
    Lugoj=(165, 379),
    Mehadia=(168, 339),
    Neamt=(406, 537),
    Oradea=(131, 571),
    Pitesti=(320, 368),
    Rimnicu=(233, 410),
    Sibiu=(207, 457),
    Timisoara=(94, 410),
    Urziceni=(456, 350),
    Vaslui=(509, 444),
    Zerind=(108, 531),
)

vacuum_world = Graph(
    dict(
        State_1=dict(Suck=["State_7", "State_5"], Right=["State_2"]),
        State_2=dict(Suck=["State_8", "State_4"], Left=["State_2"]),
        State_3=dict(Suck=["State_7"], Right=["State_4"]),
        State_4=dict(Suck=["State_4", "State_2"], Left=["State_3"]),
        State_5=dict(Suck=["State_5", "State_1"], Right=["State_6"]),
        State_6=dict(Suck=["State_8"], Left=["State_5"]),
        State_7=dict(Suck=["State_7", "State_3"], Right=["State_8"]),
        State_8=dict(Suck=["State_8", "State_6"], Left=["State_7"]),
    )
)

""" 
One-dimensional state space Graph
"""
one_dim_state_space = Graph(
    dict(
        State_1=dict(Right="State_2"),
        State_2=dict(Right="State_3", Left="State_1"),
        State_3=dict(Right="State_4", Left="State_2"),
        State_4=dict(Right="State_5", Left="State_3"),
        State_5=dict(Right="State_6", Left="State_4"),
        State_6=dict(Left="State_5"),
    )
)
one_dim_state_space.least_costs = dict(
    State_1=8, State_2=9, State_3=2, State_4=2, State_5=4, State_6=3
)

""" 
Principal states and territories of Australia
"""
australia_map = UndirectedGraph(
    dict(
        T=dict(),
        SA=dict(WA=1, NT=1, Q=1, NSW=1, V=1),
        NT=dict(WA=1, Q=1),
        NSW=dict(Q=1, V=1),
    )
)
australia_map.locations = dict(
    WA=(120, 24),
    NT=(135, 20),
    SA=(135, 30),
    Q=(145, 20),
    NSW=(145, 32),
    T=(145, 42),
    V=(145, 37),
)


class GraphProblem(Problem):
    """The problem of searching a graph from one node to another."""

    def __init__(self, initial, goal, graph):
        super().__init__(initial, goal)
        self.graph = graph

    def actions(self, A):
        """The actions at a graph node are just its neighbors."""
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or np.inf)

    def find_min_edge(self):
        """Find minimum value of edges."""
        m = np.inf
        for d in self.graph.graph_dict.values():
            local_min = min(d.values())
            m = min(m, local_min)

        return m

    def h(self, node):
        """h function is straight-line distance from a node's state to goal."""

        locs = getattr(self.graph, "locations", None)
        if locs:
            if isinstance(node, str):
                return int(distance(locs[node], locs[self.goal]))

            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return np.inf


class NQueensProblem(Problem):
    """The problem of placing N queens on an NxN board with none attacking
    each other. A state is represented as an N-element array, where
    a value of r in the c-th entry means there is a queen at column c,
    row r, and a value of -1 means that the c-th column has not been
    filled in yet. We fill in columns left to right.
    >>> depth_first_tree_search(NQueensProblem(8))
    <Node (7, 3, 0, 2, 5, 1, 6, 4)>
    """

    def __init__(self, N):
        super().__init__(tuple([-1] * N))
        self.N = N

    def actions(self, state):
        """In the leftmost empty column, try all non-conflicting rows."""
        if state[-1] != -1:
            return []  # All columns filled; no successors
        else:
            col = state.index(-1)
            return [
                row for row in range(self.N) if not self.conflicted(state, row, col)
            ]

    def result(self, state, row):
        """Place the next queen at the given row."""
        col = state.index(-1)
        new = list(state[:])
        new[col] = row
        return tuple(new)

    def conflicted(self, state, row, col):
        """Would placing a queen at (row, col) conflict with anything?"""
        return any(self.conflict(row, col, state[c], c) for c in range(col))

    def conflict(self, row1, col1, row2, col2):
        """Would putting two queens in (row1, col1) and (row2, col2) conflict?"""
        return (
            row1 == row2  # same row
            or col1 == col2  # same column
            or row1 - col1 == row2 - col2  # same \ diagonal
            or row1 + col1 == row2 + col2
        )  # same / diagonal

    def goal_test(self, state):
        """Check if all columns filled, no conflicts."""
        if state[-1] == -1:
            return False
        return not any(
            self.conflicted(state, state[col], col) for col in range(len(state))
        )

    def h(self, node):
        """Return number of conflicting queens for a given node"""
        num_conflicts = 0
        for r1, c1 in enumerate(node.state):
            for r2, c2 in enumerate(node.state):
                if (r1, c1) != (r2, c2):
                    num_conflicts += self.conflict(r1, c1, r2, c2)

        return num_conflicts


class InstrumentedProblem(Problem):
    """Delegates to a problem, and keeps statistics."""

    def __init__(self, problem):
        self.problem = problem
        self.succs = self.goal_tests = self.states = 0
        self.found = None

    def actions(self, state):
        self.succs += 1
        return self.problem.actions(state)

    def result(self, state, action):
        self.states += 1
        return self.problem.result(state, action)

    def goal_test(self, state):
        self.goal_tests += 1
        result = self.problem.goal_test(state)
        if result:
            self.found = state
        return result

    def path_cost(self, c, state1, action, state2):
        return self.problem.path_cost(c, state1, action, state2)

    def value(self, state):
        return self.problem.value(state)

    def __getattr__(self, attr):
        return getattr(self.problem, attr)

    def summary(self):
        return {
            "successors": self.succs,
            "goal_tests": self.goal_tests,
            "states": self.states,
            "found": self.found,
        }

    def __repr__(self):
        return "<{:4d}/{:4d}/{:4d}/{}>".format(
            self.succs, self.goal_tests, self.states, str(self.found)[:4]
        )
