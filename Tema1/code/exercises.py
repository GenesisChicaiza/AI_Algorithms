from problems import RandomGraph, GraphProblem


graph = RandomGraph(nodes=list(range(5)))
problem = GraphProblem(1, 5, graph)
print(graph.graph_dict)
