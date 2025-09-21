import argparse
import sys
from dataclasses import dataclass

from algorithms import backtracking, breadth_first_graph_search
from problems import Problem

sys.setrecursionlimit(10**6)
from typing import Set


@dataclass
class FarmerProblemState:
    farmer_on_left: bool
    animals_on_left: Set[str]
    animals_on_right: Set[str]

    def __repr__(self) -> str:
        # Change "wolf" to "W" and "goat" to "G" and "cabbage" to "C"
        left_str = "".join([animal[0].upper() for animal in self.animals_on_left])
        right_str = "".join([animal[0].upper() for animal in self.animals_on_right])
        if self.farmer_on_left:
            left_str = "F" + left_str
        else:
            right_str = right_str + "F"
        return f"{left_str}||{right_str}"

    def __hash__(self) -> int:
        return hash(
            (
                self.farmer_on_left,
                frozenset(self.animals_on_left),
                frozenset(self.animals_on_right),
            )
        )


@dataclass
class FarmerAction:
    farmer_to_right: bool
    animal: str

    def __repr__(self) -> str:
        if len(self.animal) != 0:
            return f"{'>' if self.farmer_to_right else '<'}{self.animal[0].upper()}"
        else:
            return f"{'>' if self.farmer_to_right else '<'}"


class FarmerProblem(Problem):
    def __init__(self):
        initial_state = FarmerProblemState(
            True, set(["wolf", "goat", "cabbage"]), set([])
        )
        end_state = FarmerProblemState(False, set([]), set(["wolf", "goat", "cabbage"]))
        super().__init__(initial_state, end_state)

    def _is_valid_state(self, animals):
        if "wolf" in animals and "goat" in animals:
            return False
        if "goat" in animals and "cabbage" in animals:
            return False
        return True

    def actions(self, state):
        if state.farmer_on_left:
            available_animals = state.animals_on_left.union(set([""]))
            return [
                FarmerAction(True, animal)
                for animal in available_animals
                if self._is_valid_state(state.animals_on_left.difference(set([animal])))
            ]
        else:
            available_animals = state.animals_on_right.union(set([""]))
            return [
                FarmerAction(False, animal)
                for animal in available_animals
                if self._is_valid_state(
                    state.animals_on_right.difference(set([animal]))
                )
            ]

    def result(self, state, action):
        if action.farmer_to_right:
            if action.animal == "":
                new_state = FarmerProblemState(
                    False, state.animals_on_left, state.animals_on_right
                )
            else:
                new_state = FarmerProblemState(
                    False,
                    state.animals_on_left.difference(set([action.animal])),
                    state.animals_on_right.union(set([action.animal])),
                )
        else:
            if action.animal == "":
                new_state = FarmerProblemState(
                    True, state.animals_on_left, state.animals_on_right
                )
            else:
                new_state = FarmerProblemState(
                    True,
                    state.animals_on_left.union(set([action.animal])),
                    state.animals_on_right.difference(set([action.animal])),
                )
        return new_state

    def path_cost(self, c, state1, action, state2):
        return c + 1


if __name__ == "__main__":
    problem = FarmerProblem()
    node = breadth_first_graph_search(problem)
    print(node.path())
