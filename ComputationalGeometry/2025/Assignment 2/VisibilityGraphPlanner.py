import math
import heapq
from typing import List, Tuple, Dict, Optional
from Point import Point
from Geometric import Geometric
from Problem import Problem
from Logger import Logger
from Configuration import Configuration

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
        self._add_vertex( self.problem.start)
        self._add_vertex( self.problem.goal)

        n = len(self.vertices)
        self.logger.debug(f"Collected {n} vertices (including start/goal).")

        # Initialize adjacency lists.
        for i in range(n):
            self.adj[i] = []

        total_pairs = n * (n - 1) // 2
        self.logger.debug(f"Testing visibility for {total_pairs} vertex pairs (O(n^2) process).")

        checked_pairs = 0
        visible_edges = 0
        # Test each pair of distinct vertices for visibility.
        for i in range(n):
            p = self.vertices[i]
            for j in range(i + 1, n):
                q = self.vertices[j]
                checked_pairs += 1
                if not Geometric().segment_hits_obstacle(p, q, self.problem.env):
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
                 start to goal (inclusive), or ``None`` if no path exists.
        :rtype: list[Point] | None
        """
        if not self.vertices:
            raise RuntimeError("Graph is empty. Call build_graph() first.")

        if self.problem.start not in self.index_of or self.problem.goal not in self.index_of:
            raise ValueError("Start or goal not present in the visibility graph.")

        start_idx = self.index_of[self.problem.start]
        goal_idx = self.index_of[self.problem.goal]
        self.logger.debug(
            f"Running Dijkstra's algorithm on visibility graph from vertex "
            f"{start_idx} to {goal_idx}."
        )

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