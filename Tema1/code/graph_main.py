from algorithms import uniform_cost_search
from problems import GraphProblem, UndirectedGraph

graph = UndirectedGraph(
    dict(
        A=dict(B=1, C=100),
        B=dict(A=1, C=1, D=100),
        C=dict(A=100, B=1, D=1),
        D=dict(B=100, C=1),
    )
)

problem = GraphProblem("A", "D", graph)
print(uniform_cost_search(problem).path())






