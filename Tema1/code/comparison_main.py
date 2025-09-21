import pandas as pd

from algorithms import (
    astar_search,
    breadth_first_graph_search,
    breadth_first_tree_search,
    depth_first_graph_search,
    depth_limited_search,
    iterative_deepening_search,
    uniform_cost_search,
)
from problems import (
    EightPuzzle,
    GraphProblem,
    InstrumentedProblem,
    MazeProblem,
    RandomGraph,
)


def compare_searchers(
    problems,
    searchers=[
        breadth_first_tree_search,
        breadth_first_graph_search,
        depth_first_graph_search,
        iterative_deepening_search,
        depth_limited_search,
        astar_search,
    ],
):
    def do(searcher, problem):
        p = InstrumentedProblem(problem)
        searcher(p)
        summary = p.summary()
        summary["problem"] = problem.__class__.__name__
        return summary

    table = []
    for s in searchers:
        result = pd.DataFrame([do(s, p) for p in problems])
        result["searcher"] = s.__name__
        table.append(result)
    table = pd.concat(table)
    return table


if __name__ == "__main__":
    problem_size = 3
    maze_problem = MazeProblem(
        problem_size, problem_size, (0, 0), (problem_size - 1, problem_size - 1)
    )
    maze_problem.h = lambda node: abs(node.state[0] - problem_size + 1) + abs(
        node.state[1] - problem_size + 1
    )
    nodes = list(range(problem_size))
    random_graph = RandomGraph(nodes)
    problems = [
        GraphProblem(0, nodes[-1], random_graph),
        maze_problem,
    ]

    print(compare_searchers(problems, searchers=[astar_search, uniform_cost_search]))
