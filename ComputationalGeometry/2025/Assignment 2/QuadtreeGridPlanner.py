from typing import List, Tuple, Dict, Optional, Iterable, Set
from collections import deque
from Point import Point
from GridCell import GridCell
from Logger import Logger
from Geometric import Geometric
from Configuration import Configuration
from Problem import Problem


class QuadtreeGridPlanner:
    """
    Approximate planner based on a uniform grid / full quadtree.

    The idea is to discretize the plane into a square grid of cells. Each cell
    is classified as either free or blocked based on whether its center lies
    inside any obstacle. We then build an implicit graph whose nodes are free
    cells and whose edges connect horizontally and vertically adjacent free
    cells. Shortest (fewest steps) path is computed using BFS.

    This implementation is intentionally simple and emphasizes clarity over
    geometric tightness. A more advanced version could:

    * Use adaptive subdivision where only cells intersecting obstacles are
      refined (a true quadtree).
    * Classify cells more robustly using polygon-square intersection tests.
    * Use weighted edges and A* search with Euclidean heuristics.
    """

    def __init__(self, problem: Problem, config: Configuration, logger: Logger) -> None:
        """
        Initialize the planner with an environment and discretization settings.

        :param problem: object containing polygonal environment with obstacles, start, and goal
        :type problem: Problem
        :param config: running configuration object
        :type config: Configuration
        :param logger: logger object
        :type logger: Logger
        """
        self.logger = logger

        # Number of cells along one side of the square grid, total number of cells is grid_size**2
        if config.grid_size <= 0:
            self.logger.value_error("grid_size must be positive.")

        self.problem = problem
        self.config = config

        # Bounding square parameters (to be computed when we know start/goal).
        self.left: float = 0.0
        self.bottom: float = 0.0
        self.cell_size: float = 1.0

        # Grid of cells. We use a 2D list [j][i] to store rows and columns.
        self.grid: List[List[GridCell]] = []



    def _compute_bounding_square(self, extra_points: Iterable[Point]) -> None:
        """
        Compute a square bounding box that contains all obstacles and
        the provided extra points (e.g., start and goal).

        The square is obtained by first computing the minimal axis-aligned
        bounding box of all points, then enlarging it by ``margin_ratio``
        on all sides, and finally making it a square by taking the larger
        of width and height.

        The resulting square is described by ``self.left`` and
        ``self.bottom`` (coordinates of the lower-left corner) and the
        uniform cell size ``self.cell_size``.

        :param extra_points: Additional points to include (usually start/goal).
        :type extra_points: collections.abc.Iterable[Point]
        """
        self.logger.debug("Computing bounding square for grid planner.")
        xs: List[float] = []
        ys: List[float] = []

        # Collect all obstacle vertices.
        for v in self.problem.env.all_vertices():
            xs.append(v.x)
            ys.append(v.y)

        # Add the extra points such as start and goal.
        for p in extra_points:
            xs.append(p.x)
            ys.append(p.y)

        if not xs or not ys:
            # Fallback in case the environment is empty (unlikely in practice).
            xs = [0.0, 1.0]
            ys = [0.0, 1.0]

        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        width = max_x - min_x
        height = max_y - min_y

        # Avoid degenerate zero-size bounding boxes.
        min_side = max(width, height, 1.0)

        # Relative margin added around the axis-aligned bounding box of all relevant points(obstacle vertices + start/goal).
        # For example, ``0.1`` means we enlarge the bounding box by 10% on all sides before overlaying the grid.
        margin = self.config.margin_ratio * min_side
        side = min_side + 2.0 * margin

        self.left = min_x - margin
        self.bottom = min_y - margin
        self.cell_size = side / self.config.grid_size

        self.logger.debug(
            f"Bounding square: left={self.left:.3f}, bottom={self.bottom:.3f}, "
            f"side={side:.3f}, cell_size={self.cell_size:.4f}."
        )

    def _world_to_cell(self, p: Point) -> Tuple[int, int]:
        """
        Convert a point in world coordinates to integer cell indices.

        The mapping is based on the previously computed bounding square.
        Indices are clamped to the valid range ``[0, grid_size - 1]``.

        :param p: Point in world coordinates.
        :type p: Point
        :return: Tuple ``(i, j)`` of grid indices.
        :rtype: tuple[int, int]
        """
        i = int((p.x - self.left) / self.cell_size)
        j = int((p.y - self.bottom) / self.cell_size)

        # Clamp indices to be safe.
        i = max(0, min(self.config.grid_size - 1, i))
        j = max(0, min(self.config.grid_size - 1, j))
        return i, j

    def _cell_center(self, i: int, j: int) -> Point:
        """
        Compute the world-coordinate center of a given grid cell.

        :param i: Column index (0-based).
        :type i: int
        :param j: Row index (0-based).
        :type j: int
        :return: Center point of cell ``(i, j)``.
        :rtype: Point
        """
        cx = self.left + (i + 0.5) * self.cell_size
        cy = self.bottom + (j + 0.5) * self.cell_size
        return Point(cx, cy)

    def _classify_cells(self) -> None:
        """
        Build the grid of cells and classify each as free or blocked.

        A cell is considered blocked if its center lies inside or on the
        boundary of any obstacle polygon. This is a conservative but simple
        approximation: it guarantees that the path will not go through the
        interior of an obstacle as long as obstacles are not too small
        compared to the cell size.

        The result is stored in ``self.grid`` as a 2D list of :class:`GridCell`
        objects.
        """
        self.logger.debug(
            f"Classifying {self.config.grid_size * self.config.grid_size} grid cells as "
            "free or blocked."
        )
        self.grid = []
        free_count = 0
        blocked_count = 0
        for j in range(self.config.grid_size):
            row: List[GridCell] = []
            for i in range(self.config.grid_size):
                center = self._cell_center(i, j)
                blocked = Geometric().point_in_any_obstacle(center, self.problem.env)
                if blocked:
                    blocked_count += 1
                else:
                    free_count += 1
                row.append(GridCell(i=i, j=j, center=center, blocked=blocked))
            self.grid.append(row)
        self.logger.debug(
            f"Grid classification complete: {free_count} free cells, "
            f"{blocked_count} blocked cells."
        )

    def build_grid(self) -> None:
        """
        Construct the bounding square and classify a uniform grid of cells.

        This method must be called **before** attempting to plan a path.
        It first computes a bounding square that contains all obstacles,
        the start point, and the goal point, then overlays a uniform grid
        over this square and classifies each cell as free or blocked.
        """
        self.logger.debug("Building uniform grid for grid-based planner.")
        self._compute_bounding_square(extra_points=[self.problem.start, self.problem.goal])
        self._classify_cells()

    def _neighbors(self, i: int, j: int) -> Iterable[Tuple[int, int]]:
        """
        Generate 4-connected neighbors of a given cell within the grid bounds.

        Only neighbors within the index range ``[0, grid_size - 1]`` are
        yielded; connectivity is restricted to horizontal and vertical moves.

        :param i: Column index of the current cell.
        :type i: int
        :param j: Row index of the current cell.
        :type j: int
        :return: Generator of neighbor index pairs ``(ni, nj)``.
        :rtype: collections.abc.Iterable[tuple[int, int]]
        """
        if i > 0:
            yield i - 1, j
        if i < self.config.grid_size - 1:
            yield i + 1, j
        if j > 0:
            yield i, j - 1
        if j < self.config.grid_size - 1:
            yield i, j + 1

    def shortest_path(self) -> Optional[List[Point]]:
        """
        Compute an approximate shortest path using BFS on the grid.

        The start and goal points are first mapped to their containing cells.
        A breadth-first search is then performed over the implicit graph of
        free cells (cells whose centers lie outside all obstacles). The path
        found minimizes the number of grid steps (i.e. Manhattan distance).
        The returned path is the sequence of cell centers from start to goal.

        If either the start or the goal falls into a blocked cell (i.e. its
        containing cell center lies inside an obstacle), this function returns
        ``None`` to indicate that no path can be constructed at the chosen
        resolution.
        :return: List of points representing the path (cell centers) from
                 start to goal, or None if no path was found.
        :rtype: list[Point] | None
        :raises RuntimeError: If build_grid has not been called.
        """
        if not self.grid:
            raise RuntimeError("Grid is empty. Call build_grid() first.")

        start_ij = self._world_to_cell(self.problem.start)
        goal_ij = self._world_to_cell(self.problem.goal)

        si, sj = start_ij
        gi, gj = goal_ij

        self.logger.debug(
            f"Starting BFS on grid from cell {start_ij} to cell {goal_ij} "
            "(4-connected)."
        )

        if self.grid[sj][si].blocked:
            self.logger.debug("Start cell is blocked; cannot plan on grid.")
            return None
        if self.grid[gj][gi].blocked:
            self.logger.debug("Goal cell is blocked; cannot plan on grid.")
            return None

        # Standard BFS on grid.
        queue: deque[Tuple[int, int]] = deque()
        queue.append(start_ij)

        visited: Set[Tuple[int, int]] = {start_ij}
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        expansions = 0
        found = False

        while queue:
            i, j = queue.popleft()
            expansions += 1
            if (i, j) == goal_ij:
                self.logger.debug("Reached goal cell during BFS.")
                found = True
                break

            for ni, nj in self._neighbors(i, j):
                if (ni, nj) in visited:
                    continue
                cell = self.grid[nj][ni]
                if cell.blocked:
                    continue
                visited.add((ni, nj))
                parent[(ni, nj)] = (i, j)
                queue.append((ni, nj))
            if expansions % 1000 == 0:
                self.logger.debug(f"  BFS expansions so far: {expansions} nodes.")

        if not found:
            self.logger.debug(
                f"BFS terminated after expanding {expansions} cells without "
                "reaching goal."
            )
            return None

        # Reconstruct path by following parents from goal back to start.
        path_indices: List[Tuple[int, int]] = []
        cur = goal_ij
        while True:
            path_indices.append(cur)
            if cur == start_ij:
                break
            cur = parent[cur]
        path_indices.reverse()
        self.logger.debug(
            f"BFS finished. Expanded {expansions} cells. "
            f"Path length in steps: {max(0, len(path_indices) - 1)}."
        )
        # Map cell indices to their center points.
        return [self.grid[j][i].center for (i, j) in path_indices]