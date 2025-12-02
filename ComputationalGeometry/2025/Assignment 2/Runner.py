from Problem import Problem
from Logger import Logger
from Configuration import Configuration
from PlanningResult import PlanningResult
from VisibilityGraphPlanner import VisibilityGraphPlanner
from QuadtreeGridPlanner import QuadtreeGridPlanner


class Runner:
    def __init__(self, problem: Problem, logger:Logger, config:Configuration):
        """
        :param problem: object containing environment, start, goal
        :type problem: Problem
        :param logger: logger object
        :type logger: Logger
        :param config: running configuration
        :type config: Configuration
        """
        self.problem = problem
        self.logger = logger
        self.config = config

    def run_planners(self) -> PlanningResult:
        """
        Run both planners (visibility graph and grid-based) on a given environment.
        The function builds the visibility graph, computes the shortest path, then
        builds the grid and computes the approximate path. All intermediate
        objects and paths are returned in a PlanningResult object
        :return: Planning results for subsequent analysis and visualization
        :rtype: PlanningResult
        """

        # Visibility graph planner.
        self.logger.info("Building visibility graph and computing exact shortest path")
        vg_planner = VisibilityGraphPlanner(self.problem, self.config, self.logger)
        vg_planner.build_graph()
        vg_path = vg_planner.shortest_path()

        # Grid / quadtree planner.
        self.logger.info("Building grid and computing approximate shortest path (BFS)")
        grid_planner = QuadtreeGridPlanner(self.problem, self.config, self.logger)
        grid_planner.build_grid()
        grid_path = grid_planner.shortest_path()

        self.logger.info("Both planners finished, preparing report")
        return PlanningResult(
            problem=self.problem,
            vg_planner=vg_planner,
            vg_path=vg_path,
            grid_planner=grid_planner,
            grid_path=grid_path,
        )