from typing import List
import math
from Point import Point
from Environment import Environment

#: Small epsilon used for floating-point robustness in geometric predicates.
EPS: float = 1e-9

class Geometric:
    def orient(self, a: Point, b: Point, c: Point) -> float:
        """
        Oriented area / orientation test for three points.

        The sign of the result indicates the turn direction when going from
        ``a`` to ``b`` to ``c``:

        * ``> 0`` – counter-clockwise turn (left of ``ab``).
        * ``< 0`` – clockwise turn (right of ``ab``).
        * ``= 0``: Collinear.

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

    def on_segment(self, a: Point, b: Point, p: Point) -> bool:
        """
        Test whether a point lies on segment ``ab`` (including endpoints).

        The function first checks whether ``a``, ``b``, and ``p`` are collinear
        using :func:`orient`, and then checks whether ``p`` lies within the axis-
        aligned bounding box of the segment.

        :param a: First segment endpoint.
        :type a: Point
        :param b: Second segment endpoint.
        :type b: Point
        :param p: Query point.
        :type p: Point
        :return: ``True`` if ``p`` lies on ``ab`` up to an epsilon, ``False`` otherwise.
        :rtype: bool
        """
        if abs(self.orient(a, b, p)) > EPS:
            return False
        return (
            min(a.x, b.x) - EPS <= p.x <= max(a.x, b.x) + EPS
            and min(a.y, b.y) - EPS <= p.y <= max(a.y, b.y) + EPS
        )

    def segments_intersect(self, a: Point, b: Point, c: Point, d: Point) -> bool:
        """
        Test whether two closed segments ``ab`` and ``cd`` intersect.

        Intersections include:

        * Proper crossings.
        * Endpoint touches.
        * Collinear overlaps.

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
        o1 = self.orient(a, b, c)
        o2 = self.orient(a, b, d)
        o3 = self.orient(c, d, a)
        o4 = self.orient(c, d, b)

        # Proper intersection: strict sign change for both segment pairs.
        if o1 * o2 < -EPS and o3 * o4 < -EPS:
            return True

        # Collinear and endpoint intersections.
        if abs(o1) <= EPS and self.on_segment(a, b, c):
            return True
        if abs(o2) <= EPS and self.on_segment(a, b, d):
            return True
        if abs(o3) <= EPS and self.on_segment(c, d, a):
            return True
        if abs(o4) <= EPS and self.on_segment(c, d, b):
            return True

        return False

    def point_in_polygon(self, p: Point, poly: List[Point]) -> bool:
        """
        Determine whether a point lies inside a simple polygon.

        The implementation uses the standard ray casting / even–odd rule with
        a horizontal ray to the right. Points lying exactly on an edge are
        treated as inside.

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
            if self.on_segment(a, b, p):
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

    def point_in_any_obstacle(self, p: Point, env: Environment) -> bool:
        """
        Check whether a point lies inside any obstacle in the environment.

        :param p: Query point.
        :type p: Point
        :param env: Polygonal environment with obstacles.
        :type env: Environment
        :return: ``True`` if ``p`` lies inside (or on the boundary of) at least one
                 obstacle polygon, ``False`` otherwise.
        :rtype: bool
        """
        for poly in env.obstacles:
            if self.point_in_polygon(p, poly.vertices):
                return True
        return False

    def segment_hits_obstacle(self, p: Point, q: Point, env: Environment) -> bool:
        """
        Test whether segment ``pq`` intersects any obstacle edge in the environment.

        Intersections that occur only at shared endpoints (when ``p`` or ``q``
        coincide with an obstacle vertex) are ignored. This allows visibility
        along polygon edges.

        :param p: First segment endpoint.
        :type p: Point
        :param q: Second segment endpoint.
        :type q: Point
        :param env: Polygonal environment.
        :type env: Environment
        :return: ``True`` if the segment interior intersects any obstacle edge
                 in a non-trivial way, ``False`` otherwise.
        :rtype: bool
        """
        for poly in env.obstacles:
            for a, b in poly.edges():
                # Skip edges that share at least one endpoint with pq.
                # This allows visibility along polygon edges.
                if (a == p) or (a == q) or (b == p) or (b == q):
                    continue

                if self.segments_intersect(p, q, a, b):
                    return True

        return False

    def polyline_length(self, path: List[Point]) -> float:
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