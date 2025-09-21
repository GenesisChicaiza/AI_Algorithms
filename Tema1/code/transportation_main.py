import argparse
import sys

from algorithms import backtracking
from problems import Problem

# sys.setrecursionlimit(10**6)


class TransportationProblem(Problem):
    def __init__(self, n):
        self.n = n
        super().__init__(1, n)

    def actions(self, state):
        valid_actions = []
        if state + 1 <= self.n:
            valid_actions.append("walk")
        if 2 * state <= self.n:
            valid_actions.append("tram")
        return valid_actions

    def result(self, state, action):
        if action == "walk":
            return state + 1
        else:
            return 2 * state

    def path_cost(self, c, state1, action, state2):
        if action == "walk":
            return c + 1
        else:
            return c + 2


if __name__ == "__main__":
    # Parse from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of blocks")
    args = parser.parse_args()

    problem = TransportationProblem(args.n)
    node = backtracking(problem)
    print(node.path())
