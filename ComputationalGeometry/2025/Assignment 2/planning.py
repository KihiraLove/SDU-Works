import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Set, Iterable

from geometry import Problem, Point, GridCell, Geometric
from meta import Configuration, Logger


class VisibilityGraphPlanner:
    """
    Planner that builds a visibility graph and computes shortest paths on it.
    The visibility graph has one vertex for each obstacle vertex, plus the
    start and goal points. Two vertices are connected if the open segment
    between them lies entirely in free space (i.e. does not intersect the
    interior of any obstacle). Each edge is weighted by the Euclidean distance
    between its endpoints.
    Shortest paths are then computed using Dijkstra's algorithm.
    This implementation uses a simple, naive O(n^2 m) visibility test,
    where n is the number of vertices and m is the number of edges.
    """

    def __init__(self, problem: Problem, config: Configuration, logger: Logger):
        """
        Initialize the planner with a given polygonal environment.

        :param problem: object containing polygonal environment with obstacles, start, and goal
        :type problem: Problem
        :param config: running configuration object
        :type config: Configuration
        :param logger: logger object
        :type logger: Logger
        """
        self.problem = problem
        self.logger = logger
        self.config = config
        # These will be populated when building the graph.
        self.vertices: List[Point] = []
        self.index_of: Dict[Point, int] = {}
        self.adj: Dict[int, List[Tuple[int, float]]] = {}

    @staticmethod
    def _euclidean_distance(a: Point, b: Point) -> float:
        """
        Compute Euclidean distance between two points.
        :param a: First point.
        :type a: Point
        :param b: Second point.
        :type b: Point
        :return: Euclidean distance between ``a`` and ``b``.
        :rtype: float
        """
        dx = a.x - b.x
        dy = a.y - b.y
        return math.hypot(dx, dy)

    def _add_vertex(self, p: Point) -> int:
        """
        Add a vertex to the graph if it is not already present.
        Points are compared using exact floating-point equality. In practical
        applications it is often better to avoid duplicates explicitly when
        building the environment.
        :param p: Point to insert.
        :type p: Point
        :return: Index of the point in :attr:`vertices`.
        :rtype: int
        """
        if p in self.index_of:
            return self.index_of[p]
        idx = len(self.vertices)
        self.vertices.append(p)
        self.index_of[p] = idx
        return idx

    def build_graph(self) -> None:
        """
        Construct the visibility graph for the current environment.
        The method collects all obstacle vertices plus the start and goal
        points as graph vertices, then tests all :math:`O(n^2)` pairs for
        mutual visibility using the segment-obstacle intersection test.
        The resulting adjacency list is stored in ``self.adj`` as a dictionary
        mapping vertex indices to lists of ``(neighbor_index, edge_length)``
        pairs.
        """
        # Clear any existing graph.
        self.vertices = []
        self.index_of = {}
        self.adj = {}

        # Add all obstacle vertices.
        for v in self.problem.env.all_vertices():
            self._add_vertex(v)

        # Add start and goal.
        self._add_vertex(self.problem.start)
        self._add_vertex(self.problem.goal)

        n = len(self.vertices)
        self.logger.debug(f"Collected {n} vertices (including start/goal).")

        # Initialize adjacency lists.
        for i in range(n):
            self.adj[i] = []

        total_pairs = n * (n - 1) // 2
        self.logger.debug(f"Testing visibility for {total_pairs} vertex pairs (O(n^2) process).")

        geom = Geometric()
        checked_pairs = 0
        visible_edges = 0
        # Test each pair of distinct vertices for visibility.
        for i in range(n):
            p = self.vertices[i]
            for j in range(i + 1, n):
                q = self.vertices[j]
                checked_pairs += 1
                if not geom.segment_hits_obstacle(p, q, self.problem.env):
                    w = self._euclidean_distance(p, q)
                    self.adj[i].append((j, w))
                    self.adj[j].append((i, w))
                    visible_edges += 1
                if checked_pairs % 1000 == 0:
                    self.logger.debug(f"  Processed {checked_pairs}/{total_pairs} pairs so far...")

        self.logger.debug(
            f"Finished visibility computation: {visible_edges} undirected edges "
            f"added out of {total_pairs} possible pairs."
        )

    def edge_count(self) -> int:
        """
        Compute the number of undirected edges in the visibility graph.
        The internal representation stores edges in an adjacency list where
        each undirected edge appears exactly twice (once for each endpoint).
        This function sums all adjacency lengths and divides by two.

        :return: Number of undirected edges.
        :rtype: int
        """
        total = sum(len(neigh) for neigh in self.adj.values())
        return total // 2

    def shortest_path(self) -> Optional[List[Point]]:
        """
        Run Dijkstra's algorithm to compute a shortest path between two points.
        This method assumes that :meth:`build_graph` has already been called
        with the same ``start`` and ``goal`` points, so that they exist
        as vertices in the internal graph.
        :return: List of points representing the path in order from
                 start to goal (inclusive), or None if no path exists.
        :rtype: list[Point] | None
        """
        if not self.vertices:
            raise RuntimeError("Graph is empty. Call build_graph() first.")

        if self.problem.start not in self.index_of or self.problem.goal not in self.index_of:
            raise ValueError("Start or goal not present in the visibility graph.")

        start_idx = self.index_of[self.problem.start]
        goal_idx = self.index_of[self.problem.goal]
        self.logger.debug(f"Running Dijkstra's algorithm on visibility graph from vertex {start_idx} to {goal_idx}.")

        n = len(self.vertices)

        # Initialize distance and predecessor arrays.
        dist = [math.inf] * n
        prev = [-1] * n
        dist[start_idx] = 0.0

        # Priority queue of (distance, vertex_index) pairs.
        heap: List[Tuple[float, int]] = [(0.0, start_idx)]

        visited_count = 0

        # Standard Dijkstra loop.
        while heap:
            d_u, u = heapq.heappop(heap)
            if d_u > dist[u] + 1e-12:
                # Outdated entry in the queue.
                continue
            visited_count += 1
            if u == goal_idx:
                self.logger.info("Reached goal vertex during Dijkstra; stopping early.")
                break

            for v, w_uv in self.adj.get(u, []):
                alt = d_u + w_uv
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(heap, (alt, v))

        if dist[goal_idx] == math.inf:
            self.logger.info("No path found in visibility graph (goal unreachable).")
            return None

        # Reconstruct path by following predecessors from goal to start.
        path_indices: List[int] = []
        cur = goal_idx
        while cur != -1:
            path_indices.append(cur)
            cur = prev[cur]
        path_indices.reverse()
        self.logger.info(f"Dijkstra finished. Visited {visited_count} vertices. Path length in vertices: {len(path_indices)}.")
        # Convert vertex indices back to points.
        return [self.vertices[i] for i in path_indices]


class UniformGridPlanner:
    """
    Approximate planner based on a uniform grid.
    The idea is to discretize the plane into a square grid of cells. Each cell
    is classified as either free or blocked based on whether its center lies
    inside any obstacle. We then build an implicit graph whose nodes are free
    cells and whose edges connect horizontally and vertically adjacent free
    cells. Shortest (fewest steps) path is computed using BFS.
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
        The resulting square is described by self.left and
        self.bottom (coordinates of the lower-left corner) and the
        uniform cell size self.cell_size

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
        Indices are clamped to the valid range [0, grid_size - 1].
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
        :return: Center point of cell (i, j).
        :rtype: Point
        """
        cx = self.left + (i + 0.5) * self.cell_size
        cy = self.bottom + (j + 0.5) * self.cell_size
        return Point(cx, cy)

    def _classify_cells(self) -> None:
        """
        Build the grid of cells and classify each as free or blocked.

        A cell is considered blocked if its center lies inside or on the
        boundary of any obstacle polygon. This is a simple approximation:
        the BFS path will only visit cells whose centers lie in free space.
        However, the polyline obtained by connecting cell centers can still
        pass near or slightly through obstacles, especially if obstacles
        are small or thin relative to the cell size.

        The result is stored in self.grid as a 2D list of GridCell
        objects.
        """
        self.logger.debug(
            f"Classifying {self.config.grid_size * self.config.grid_size} grid cells as "
            "free or blocked."
        )
        self.grid = []
        geom = Geometric()
        free_count = 0
        blocked_count = 0
        for j in range(self.config.grid_size):
            row: List[GridCell] = []
            for i in range(self.config.grid_size):
                center = self._cell_center(i, j)
                blocked = geom.point_in_any_obstacle(center, self.problem.env)
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


@dataclass
class PlanningResult:
    """
    Container for the outputs of both planners on a single environment.

    :param problem: object containing polygonal environment, start, and goal used for planning
    :type problem: Problem
    :param vg_planner: Visibility-graph planner instance
    :type vg_planner: VisibilityGraphPlanner
    :param vg_path: Shortest path using visibility graph
    :type vg_path: list[Point] | None
    :param grid_planner: Grid-based planner instance
    :type grid_planner: UniformGridPlanner
    :param grid_path: BFS path on the grid, or ``None``
    :type grid_path: list[Point] | None
    """
    problem: Problem
    vg_planner: VisibilityGraphPlanner
    vg_path: Optional[List[Point]]
    grid_planner: UniformGridPlanner
    grid_path: Optional[List[Point]]

    def vg_path_to_tikz(self) -> str:
        """
        Tikz representation of visibility graph path
        :return: tikz representation of visibility graph path
        :rtype: str
        """
        tikz = ("% Visibility-graph shortest path\n"
                "\\draw[very thick,blue] ")
        # Draw line through points until point before last
        for point in self.vg_path[:-1]:
            tikz += point.to_tikz_line_from_point()
        # Connect line to last point
        tikz += self.vg_path[-1].to_tikz_line_end()
        return tikz

    def grid_path_to_tikz(self) -> str:
        tikz = ("% Grid-based approximate path\n"
                "\\draw[very thick,red,dashed] ")

        # Draw line through points until point before last
        for point in self.grid_path[:-1]:
            tikz += point.to_tikz_line_from_point()
        # Connect line to last point
        tikz += self.grid_path[-1].to_tikz_line_end()
        return tikz


class Runner:
    def __init__(self, problem: Problem, logger: Logger, config: Configuration):
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
        grid_planner = UniformGridPlanner(self.problem, self.config, self.logger)
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
