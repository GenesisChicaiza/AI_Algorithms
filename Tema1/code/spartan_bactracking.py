from problems import Problem, GraphProblem, Graph
from algorithms import Node

def spartan_backtracking(problem: Problem) -> Node:
    def backtrack(node: Node):
        print("exploring node: ", node.state)
        if problem.goal_test(node.state):
            return node
        else:
            for new_node in node.expand(problem):
                sol = backtrack(new_node)
                if sol is not None:
                    return sol
        return None

    initial_node = Node(problem.initial)
    sol = backtrack(initial_node)
    return sol



if __name__=="__main__":
    # Create a tree with depth 4. All costs are 1
    graph = Graph(dict(
        A=dict(B=1, C=1),
        B=dict(D=1, E=1),
        C=dict(F=1, G=1),
        D=dict(H=1, I=1),
        E=dict(J=1, K=1),
        F=dict(L=1, M=1),
        G=dict(N=1, O=1),
    ))
    problem = GraphProblem('A', 'J', graph)
    print(spartan_backtracking(problem).path())
