# Create a MazeProblem and visualize the solution
import random
import threading

from algorithms import (
    astar_search,
    breadth_first_graph_search,
    depth_limited_search,
    uniform_cost_search,
)
from problems import MazeProblem, MazeVisualizer

random.seed(10)
problem = MazeProblem(20, 20, (9, 9), (19, 19))
visualizer = MazeVisualizer(problem)


def visualize(current_node):
    states_path = [node.state for node in current_node.path()]
    visualizer.draw_current_path(states_path)


def h(node):
    x, y = node.state
    return (19 - x) + (19 - y)


def run_algorithm(problem, visualize):
    # solution = breadth_first_graph_search(problem, visualize)
    # solution = uniform_cost_search(problem, visualize)
    solution = astar_search(problem, h=h, visualize=visualize)
    # solution = depth_limited_search(problem, limit=3, visualize=visualize)
    if solution is not None:
        if solution == "cutoff":
            print("CUTOFF! Max depth reached!")
        else:
            # If a solution is found, mark the path
            node = solution
            solution_states_path = []
            while node.parent:
                solution_states_path.append(node.state)
                node = node.parent
            visualizer.draw_solution(solution_states_path)
    else:
        print("No solution found!")


thread = threading.Thread(target=run_algorithm, args=(problem, visualize))
thread.start()

visualizer.mainloop()
