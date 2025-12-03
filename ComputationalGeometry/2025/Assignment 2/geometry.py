import math
from dataclasses import dataclass
from typing import List, Iterable, Tuple


#: Small epsilon used for floating-point robustness in geometric predicates.
EPS: float = 1e-9


@dataclass(frozen=True)
class Point:
    """
    Simple 2D point.
    The class is immutable and hashable so that points can be used as dictionary keys or elements in sets.
    :param x: X-coordinate.
    :type x: float
    :param y: Y-coordinate.
    :type y: float
    """
    x: float
    y: float

    def to_tikz_line_from_point(self) -> str:
        """
        TikZ representation of a line from this point.
        :return: TikZ representation of a line from this point
        :rtype: str
        """
        return f"({self.x},{self.y}) -- "

    def to_tikz_line_end(self) -> str:
        """
        TikZ representation of a line ending on this point.
        :return: TikZ representation of a line ending on this point
        :rtype: str
        """
        return f"({self.x},{self.y});\n"

    def to_tikz(self, colour: str, p_type: str) -> str:
        """
        TikZ representation of point.
        :param colour: colour of the point
        :type colour: str
        :param p_type: start or goal
        :type p_type: str
        :return: TikZ representation of point
        :rtype: str
        """
        content = f"\\filldraw[{colour}!70!black] ({self.x},{self.y}) circle[radius=0.08]; % "
        if p_type in ["start", "goal"]:
            return content + p_type + "\n"
        raise ValueError("Invalid point type")


@dataclass
class GridCell:
    """
    Simple data holder for a grid cell used in the grid-based planner.
    In this implementation we build a uniform grid of square cells.
    Each cell is identified by its integer grid coordinates ``(i, j)``.
    :param i: Column index of the cell (0-based)
    :type i: int
    :param j: Row index of the cell (0-based)
    :type j: int
    :param center: Center point of the cell in world coordinates
    :type center: Point
    :param blocked: Flag indicating whether this cell is considered occupied by an obstacle.
    :type blocked: bool
    """
    i: int
    j: int
    center: Point
    blocked: bool = False


@dataclass
class PolygonObstacle:
    """
    Simple polygonal obstacle described by its vertices in counter-clockwise order.
    The polygon is assumed to be simple (non-self-intersecting), and closed implicitly,
    so there is an edge from the last vertex back to the first one.
    :param vertices: Vertices of the polygon in counter-clockwise order
    :type vertices: list[Point]
    """
    vertices: List[Point]

    def edges(self) -> Iterable[Tuple[Point, Point]]:
        """
        Iterate over polygon edges as pairs of endpoints.
        The last vertex is connected back to the first one.
        :return: Generator of edges (u, v)
        :rtype: Iterable[tuple[Point, Point]]
        """
        n = len(self.vertices)
        for i in range(n):
            yield self.vertices[i], self.vertices[(i + 1) % n]

    def to_tikz(self) -> str:
        """
        Create TikZ representation of polygon.
        :return: TikZ representation of polygon
        :rtype: str
        """
        tikz = "\\filldraw[fill=gray!20,draw=black] "
        for point in self.vertices:
            tikz += point.to_tikz_line_from_point()
        # Closing the polygon off by connecting back to the first point
        tikz += self.vertices[0].to_tikz_line_end()
        return tikz


@dataclass
class Environment:
    """
    Polygonal environment consisting of a set of polygonal obstacles.
    The free space is the complement of the union of obstacles. In this simple
    model there is no explicit outer boundary. The plane is assumed to be
    infinite and obstacles are "holes" in it.
    :param obstacles: List of polygonal obstacles
    :type obstacles: list[PolygonObstacle]
    """
    obstacles: List[PolygonObstacle]

    def all_vertices(self) -> List[Point]:
        """
        Collect all vertices of all obstacles.
        :return: List of all obstacle vertices (possibly with duplicates)
        :rtype: list[Point]
        """
        out: List[Point] = []
        for poly in self.obstacles:
            out.extend(poly.vertices)
        return out

    def all_edges(self) -> Iterable[Tuple[Point, Point]]:
        """
        Iterate over all edges of all obstacles.
        :return: Generator of all obstacle edges
        :rtype: Iterable[tuple[Point, Point]]
        """
        for poly in self.obstacles:
            yield from poly.edges()


@dataclass
class Problem:
    """
    Container for problem space.
    :param env: Polygonal environment used for planning
    :type env: Environment
    :param start: Start point
    :type start: Point
    :param goal: Goal point
    :type goal: Point
    """
    env: Environment
    start: Point
    goal: Point


class Geometric:
    @staticmethod
    def orient(a: Point, b: Point, c: Point) -> float:
        """
        Oriented area / orientation test for three points.
        The sign of the result indicates the turn direction when going from
        ``a`` to ``b`` to ``c``:
        - ``> 0`` – counter-clockwise turn (left of ``ab``).
        - ``< 0`` – clockwise turn (right of ``ab``).
        - ``= 0``: Collinear.
        Geometrically this is twice the signed area of triangle ``abc``.

        :param a: First point.
        :type a: Point
        :param b: Second point.
        :type b: Point
        :param c: Third point.
        :type c: Point
        :return: Signed area value.
        :rtype: float
        """
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

    @staticmethod
    def on_segment(a: Point, b: Point, p: Point) -> bool:
        """
        Test whether a point lies on segment ``ab`` (including endpoints).
        The function first checks whether ``a``, ``b``, and ``p`` are collinear
        using :func:`orient`, and then checks whether ``p`` lies within the axis-aligned bounding box of the segment.

        :param a: First segment endpoint.
        :type a: Point
        :param b: Second segment endpoint.
        :type b: Point
        :param p: Query point.
        :type p: Point
        :return: ``True`` if ``p`` lies on ``ab`` up to an epsilon, ``False`` otherwise.
        :rtype: bool
        """
        if abs(Geometric.orient(a, b, p)) > EPS:
            return False
        return (
            min(a.x, b.x) - EPS <= p.x <= max(a.x, b.x) + EPS
            and min(a.y, b.y) - EPS <= p.y <= max(a.y, b.y) + EPS
        )

    @staticmethod
    def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
        """
        Test whether two closed segments ``ab`` and ``cd`` intersect.
        Intersections include:
        - Proper crossings.
        - Endpoint touches.
        - Collinear overlaps.

        :param a: First endpoint of segment 1.
        :type a: Point
        :param b: Second endpoint of segment 1.
        :type b: Point
        :param c: First endpoint of segment 2.
        :type c: Point
        :param d: Second endpoint of segment 2.
        :type d: Point
        :return: ``True`` if the segments intersect in any way, ``False`` otherwise.
        :rtype: bool
        """
        o1 = Geometric.orient(a, b, c)
        o2 = Geometric.orient(a, b, d)
        o3 = Geometric.orient(c, d, a)
        o4 = Geometric.orient(c, d, b)

        # Proper intersection: strict sign change for both segment pairs.
        if o1 * o2 < -EPS and o3 * o4 < -EPS:
            return True

        # Collinear and endpoint intersections.
        if abs(o1) <= EPS and Geometric.on_segment(a, b, c):
            return True
        if abs(o2) <= EPS and Geometric.on_segment(a, b, d):
            return True
        if abs(o3) <= EPS and Geometric.on_segment(c, d, a):
            return True
        if abs(o4) <= EPS and Geometric.on_segment(c, d, b):
            return True

        return False

    @staticmethod
    def point_in_polygon(p: Point, poly: List[Point]) -> bool:
        """
        Determine whether a point lies inside a simple polygon.
        The implementation uses the standard ray casting / even–odd rule with
        a horizontal ray to the right. Points lying exactly on an edge are treated as inside.
        :param p: Query point.
        :type p: Point
        :param poly: Polygon vertices in counter-clockwise order.
        :type poly: list[Point]
        :return: ``True`` if ``p`` is inside or on the boundary, ``False`` otherwise.
        :rtype: bool
        """
        inside = False
        n = len(poly)

        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]

            # If the point lies directly on an edge, treat it as inside.
            if Geometric.on_segment(a, b, p):
                return True

            # Ensure a.y <= b.y for consistency of the ray casting test.
            if a.y > b.y:
                a, b = b, a

            # Ignore edges that are completely above or below the ray.
            # We use a half-open interval [a.y, b.y) to avoid double counting.
            if p.y <= a.y or p.y > b.y:
                continue

            # Compute x-coordinate of the intersection of the edge
            # with the horizontal line through p.y.
            x_int = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y)

            # If the intersection is strictly to the right of p, flip parity.
            if x_int > p.x:
                inside = not inside

        return inside

    @staticmethod
    def point_in_any_obstacle(p: Point, env: Environment) -> bool:
        """
        Check whether a point lies inside any obstacle in the environment.
        :param p: Query point.
        :type p: Point
        :param env: Polygonal environment with obstacles.
        :type env: Environment
        :return: ``True`` if ``p`` lies inside (or on the boundary of) at least one obstacle polygon, ``False`` otherwise.
        :rtype: bool
        """
        for poly in env.obstacles:
            if Geometric.point_in_polygon(p, poly.vertices):
                return True
        return False

    @staticmethod
    def segment_hits_obstacle(p: Point, q: Point, env: Environment) -> bool:
        """
        Test whether segment ``pq`` intersects any obstacle edge in the environment.
        Intersections that occur only at shared endpoints (when ``p`` or ``q``
        coincide with an obstacle vertex) are ignored. This allows visibility  along polygon edges.
        :param p: First segment endpoint.
        :type p: Point
        :param q: Second segment endpoint.
        :type q: Point
        :param env: Polygonal environment.
        :type env: Environment
        :return: ``True`` if the segment interior intersects any obstacle edge in a non-trivial way, ``False`` otherwise.
        :rtype: bool
        """
        for poly in env.obstacles:
            for a, b in poly.edges():
                # Skip edges that share at least one endpoint with pq.
                # This allows visibility along polygon edges.
                if (a == p) or (a == q) or (b == p) or (b == q):
                    continue

                if Geometric.segments_intersect(p, q, a, b):
                    return True

        return False

    @staticmethod
    def polyline_length(path: List[Point]) -> float:
        """
        Compute the length of a polyline represented as a list of points.
        The length is the sum of Euclidean distances between consecutive points.
        :param path: Sequence of points representing the polyline.
        :type path: list[Point]
        :return: Total length of the polyline.
        :rtype: float
        """
        if len(path) < 2:
            return 0.0
        total = 0.0
        for a, b in zip(path, path[1:]):
            dx = a.x - b.x
            dy = a.y - b.y
            total += math.hypot(dx, dy)
        return total