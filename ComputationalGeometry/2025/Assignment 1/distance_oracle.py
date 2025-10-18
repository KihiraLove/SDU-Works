import math
from typing import Optional, Dict
from segment import Segment
from math_functions import cross, EPS

class DistanceOracle:
    """
    Caches distances t(θ) along the ray p + t * v(θ) to segment intersection,
    to support comparator calls inside the status tree. Cleared whenever θ changes.
    """
    __slots__ = ("px", "py", "theta", "vx", "vy", "cache")

    def __init__(self, px: float, py: float):
        self.px = px
        self.py = py
        self.theta = None  # type: Optional[float]
        self.vx = 0.0
        self.vy = 0.0
        self.cache: Dict[int, float] = {}

    def set_theta(self, theta: float) -> None:
        self.theta = theta
        self.vx = math.cos(theta)
        self.vy = math.sin(theta)
        self.cache.clear()

    def ray_segment_distance(self, s: Segment) -> float:
        """
        Return t >= 0 where ray p + t*v intersects segment s,
        or +inf if parallel or outside param ranges.
        For active segments, t will be finite and positive.
        Uses:
          p + t v = a + u w
          Solve with 2x2 with cross products:
            t = cross(a - p, w) / cross(v, w)
            u = cross(a - p, v) / cross(v, w)
        """
        if s.id in self.cache:
            return self.cache[s.id]

        ax, ay = s.a
        bx, by = s.b
        px, py = self.px, self.py
        vx, vy = self.vx, self.vy
        wx, wy = (bx - ax), (by - ay)

        denom = cross(vx, vy, wx, wy)
        if abs(denom) <= EPS:
            # Ray nearly parallel to segment; treat as no intersection
            t = float("inf")
            self.cache[s.id] = t
            return t

        apx, apy = (ax - px), (ay - py)
        t = cross(apx, apy, wx, wy) / denom
        u = cross(apx, apy, vx, vy) / denom

        if t < 0.0 - 1e-15 or u < -1e-15 or u > 1.0 + 1e-15:
            t = float("inf")
        self.cache[s.id] = t
        return t