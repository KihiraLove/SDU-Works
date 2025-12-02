from dataclasses import dataclass
from typing import List, Optional
from Point import Point
from Environment import Environment
from VisibilityGraphPlanner import VisibilityGraphPlanner
from QuadtreeGridPlanner import QuadtreeGridPlanner
from Problem import Problem


@dataclass
class PlanningResult:
    """
    Container for the outputs of both planners on a single environment
    :param env: Polygonal environment used for planning
    :type env: Environment
    :param start: Start point
    :type start: Point
    :param goal: Goal point
    :type goal: Point
    :param vg_planner: Visibility-graph planner instance
    :type vg_planner: VisibilityGraphPlanner
    :param vg_path: Shortest path using visibility graph
    :type vg_path: list[Point] | None
    :param grid_planner: Grid-based planner instance
    :type grid_planner: QuadtreeGridPlanner
    :param grid_path: BFS path on the grid
    :type grid_path: list[Point] | None
    """
    problem: Problem
    vg_planner: VisibilityGraphPlanner
    vg_path: Optional[List[Point]]
    grid_planner: QuadtreeGridPlanner
    grid_path: Optional[List[Point]]

    def vg_path_to_tikz(self) -> str:
        """
        Tikz representation of visibility graph path
        :return: tikz representation of visibility graph path
        :rtype: str
        """
        tikz = ("% Visibility-graph shortest path\n"
                "\\draw[very thick,blue] ")
        for point in self.vg_path:
            if point is not self.vg_path[-1]:
                tikz += point.to_tikz_line_from_point()
            else:
                tikz += self.vg_path[-1].to_tikz_line_end()
        return tikz

    def grid_path_to_tikz(self) -> str:
        tikz = ("% Grid-based approximate path\n"
                "\\draw[very thick,red,dashed] ")
        for point in self.grid_path:
            if point is not self.grid_path[-1]:
                tikz += point.to_tikz_line_from_point()
            else:
                tikz += self.grid_path[-1].to_tikz_line_end()
        return tikz