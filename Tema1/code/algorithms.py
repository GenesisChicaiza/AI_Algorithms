import functools
import sys
from collections import deque
from queue import PriorityQueue

from problems import Problem


def memoize(fn, slot=None, maxsize=32):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, use lru_cache for caching the values."""


    # convierte una f(x) en una versión caché: si lo llamas con los
    # mismos argumentos, devuelve el valor ya calculado sin repetir trabajo. 


    # cada obj tiene su propio valor memorizado
    # dos nodos con el mismo estado pero caminos distintos

    if slot:

        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val

    # memoriza por argumentos

    else:

        @functools.lru_cache(maxsize=maxsize)
        def memoized_fn(*args):
            return fn(*args)

    return memoized_fn


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    """  Explaining constructor: 
            state: estado, (ej: ciudad "A")
            parent: Node anterior de la ruta, Si es None, es el inicial
            action: acción usada para llegar desde parent.state a state. En la raiz suele ser None
            path_cost: coste acumulado hasta este nodo. 
    
    
     """



    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state 
        self.parent = parent 
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    # sirve para ver en consola en q estado estas

    def __repr__(self):
        return "<{}, {}, {}>".format(self.state, self.action, self.path_cost)


    # Cuando tienen la misma prioridad se usar para desempatar
    # self.state : nodo actual 
    # node.state es otro nodo con el q se está comparando

    def __lt__(self, node):
        return self.state < node.state


    """ Devuelve todos los hijos alcanzanbles en un paso desde este nodo:
             problem.actions(self.state): lista de acciones válidas
             self.child_node(problem, action) : crea un nodo hijo


    """

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [
            self.child_node(problem, action) for action in problem.actions(self.state)
        ]


""" Problem: problema q sabe como funcionan accionaes, estados y costes
    action: Acción específica q vamos aplicar al estado actual
    Todo lo de problem lo saco de la clase problem.

"""

    def child_node(self, problem, action):
        """Get next node. Figure 3.10 in AIMA"""
        next_state = problem.result(self.state, action)
        next_node = Node(
            next_state, # a cual llegamos
            self,  # estado padre en el q estamos
            action, # acción q usamos para llegar
            problem.path_cost(
                self.path_cost,  # coste hasta el padre
                self.state,    # estado anterior
                action,     # acción hecha
                next_state),    # estado nuevo
        )

        return next_node #  devuelve el nuevo obj Node, listo para añadirse a la frontera del algoritmo de búsqueda.




    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

        #  self.path() obtiene lista desde raiz hasta el nodo,
        #  pero [1: ] ignora el nodo raiz.

""" Basicamente esta creando una lista vacia para almacenar 
    lo q has visitado, te lo da invertido. Pero luego con el reversed,
    te lo pone en el orden correcto"""

#   self = nodo actual
#   

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        
        #   node = self     path_back, lista vacía donde se guarda el camino, pero en orden hacia atras.
        node, path_back = self, []
        while node:
            #   Primero añade el nodo actual
            #   Luego sube a su padre y lo añade y asi hasta la raiz
            #   Resultado de path_back está al reves, con reversed lo devuelvo con el orden correcto


            path_back.append(node)  # guarda el nodo actual en la lista
            node = node.parent   # subimos un nivel de árbol
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]


    #   define cuando dos nodos se consideran iguales
    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    # Permite q los Node se usen en estrcuturas como set
    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)


def backtracking(problem: Problem) -> Node:

    # guarda la mejor solución encontrada hasta el momento
    # Node = None, xq aun no hay ninguna solución

    best = {"cost": float("inf"), "node": None}



    def backtrack(node: Node):

    #   Si el nodo actual es un obj, y el coste es mejor q lo encontrado
    #   Actualiza best

        if problem.goal_test(node.state) and node.path_cost < best["cost"]:
            best["cost"] = node.path_cost
            best["node"] = node
        else:

        #   Si no es obj, no mejora, se expande el nodo -> genera todos sus hijos.   

            for new_node in node.expand(problem):
                backtrack(new_node)

    initial_node = Node(problem.initial) # se crea el primer nodo
    backtrack(initial_node) # regresa
    return best["node"]



def breadth_first_tree_search(problem: Problem) -> Node:
    """
    Search the shallowest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Repeats infinitely in case of loops.
    """
  """ Empieza con el nodo inicial en la frontera.

        Mientras haya nodos por explorar:

        Saca el primero de la cola.

        Si es objetivo, lo devuelve.

        Si no, mete sus hijos al final de la cola.

        Devuelve None si no hay solución.     """
        
    # se crea frontera, lista de nodos a explorar
    #  empieza con único nodo: el inicial (problem.initial)
    frontier = deque([Node(problem.initial)])  # FIFO queue

    while frontier:
        #   Sacar el primer nodo de la frontera ( al frente de la cola)
        node = frontier.popleft()
        #   Si el estado es la meta, devuelve el Nodo
        if problem.goal_test(node.state):
            return node

        #   Si no lo es, lo extiende
        frontier.extend(node.expand(problem))

        # Si no hay solución devuelve None
    return None


# TODO: visualize
def breadth_first_graph_search(problem: Problem, visualize=None) -> Node:
    """Search the shallowest nodes in the search tree first."""

    # Se crea el Nodo raiz, con el estado inicial

    node = Node(problem.initial)

    #    Si estado incial ya es meta, se devuelve
    if problem.goal_test(node.state):
        return node

    #   cola FIFO con el nodo inicial, lista de nodos por visitar

    frontier = deque([node])
    # conjunto vacío de estados ya explorados
    explored = set()
    
    # mientras la frontera no este vacía, seguimos
    while frontier:
        node = frontier.popleft()
        # se añade estado a explorados
        explored.add(node.state)
        # Si alguien pasa un f(x) visualize, se llama aqui pa mostrar el nodo en pantalla

        if visualize:
            visualize(node)


        # genera nodos hijos desde nodo actual
        """ node.expand(problem) → genera todos los nodos hijos desde el nodo actual.

        Para cada hijo:

            Si no está en explored y no está ya en la frontier:

            Comprueba si es meta (goal_test). Si lo es, devuelve la solución enseguida.

            Si no, lo añade a la frontera para explorarlo más adelante. """ 


        for child in node.expand(problem):


            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child
                frontier.append(child)
    return None

# CONTUNUE HEREEE

def depth_first_tree_search(problem):
    """
    Search the deepest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Repeats infinitely in case of loops.
    """

    frontier = [Node(problem.initial)]  # Stack

    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))
    return None


# TODO: visualzize
def depth_first_graph_search(problem, visualize=None):
    """
    Search the deepest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Does not get trapped by loops.
    If two paths reach a state, only use the first one.
    """
    frontier = [(Node(problem.initial))]  # Stack

    explored = set()
    while frontier:
        node = frontier.pop()
        if visualize:
            visualize(node)
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        frontier.extend(
            child
            for child in node.expand(problem)
            if child.state not in explored and child not in frontier
        )
    return None


def depth_limited_search(problem, limit=50, visualize=None):
    def recursive_dls(node, problem):
        if visualize:
            visualize(node)
        if problem.goal_test(node.state):
            return node
        elif node.depth >= limit:
            # instead of None to be able to differentiate between no solution (None) and cutoff
            return "cutoff"
        else:
            cutoff_occurred = False
            for child in node.expand(problem):
                result = recursive_dls(child, problem)
                if result == "cutoff":
                    cutoff_occurred = True
                elif result is not None:
                    return result
            return "cutoff" if cutoff_occurred else None

    return recursive_dls(Node(problem.initial), problem)


def iterative_deepening_search(problem):
    for depth in range(sys.maxsize):
        result = depth_limited_search(problem, depth)
        if result != "cutoff":
            return result
    return None


class SearchPriorityQueue:
    def __init__(self, priority_function):
        self.priority_function = priority_function
        self.pq = PriorityQueue()

    def put(self, item):
        priority = self.priority_function(item)
        self.pq.put((priority, item))

    def get(self):
        # Return only the item, not the priority
        return self.pq.get()[1]

    def empty(self):
        return self.pq.empty()

    def __contains__(self, key):
        """Return True if the key is in SearchPriorityQueue."""
        return any([item == key for _, item in self.pq.queue])

    def __getitem__(self, key):
        """Returns the first value associated with key in SearchPriorityQueue.
        Raises KeyError if key is not present."""
        for value, item in self.pq.queue:
            if item == key:
                return value
        raise KeyError(str(key) + " is not in the priority queue")

    def __delitem__(self, key):
        """Delete the first occurrence of key."""
        try:
            del self.pq.queue[[item == key for _, item in self.pq.queue].index(True)]
        except ValueError:
            raise KeyError(str(key) + " is not in the priority queue")
        # Rebuild the priority queue
        items = self.pq.queue.copy()
        self.pq = PriorityQueue()
        for priority, item in items:
            self.pq.put((priority, item))


# TODO:
def best_first_graph_search(problem, f, visualize=None):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    # Memoize should rely on the slot mechanism, and not on the lru_cache.
    # This is because nodes are hashed only based on the state. As a result,
    # the lru_cache will not be able to update the f values of the nodes ending
    # on the same state, but having a different path.
    f = memoize(f, "f")
    node = Node(problem.initial)

    frontier = SearchPriorityQueue(f)
    frontier.put(node)
    explored = set()
    while frontier:
        node = frontier.get()
        if visualize:
            visualize(node)
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.put(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.put(child)
    return None


def uniform_cost_search(problem, visualize=None):
    return best_first_graph_search(problem, lambda node: node.path_cost, visualize)


def astar_search(problem, h=None, visualize=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    # Memoize should rely on the slot mechanism, and not on the lru_cache.
    # See best_first_graph_search for more details.
    h = memoize(h or problem.h, "h")
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), visualize)
