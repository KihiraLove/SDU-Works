from dataclasses import dataclass
from Point import Point
from Environment import Environment

@dataclass
class Problem:
    """
    Container for problem space
    :param env: Polygonal environment used for planning.
    :type env: Environment
    :param start: Start point.
    :type start: Point
    :param goal: Goal point.
    :type goal: Point
    """
    env: Environment
    start: Point
    goal: Point