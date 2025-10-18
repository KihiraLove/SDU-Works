from typing import Tuple
from math_functions import angle_from_p, feq, short_arc_direction, EPS


class Segment:
    __slots__ = ("id", "a", "b", "raw", "th_a", "th_b",
                 "start", "end", "wraps", "degenerate")

    def __init__(self, seg_id: int, a: Tuple[float, float], b: Tuple[float, float], raw_line: str, px: float, py: float):
        self.id = seg_id
        self.a = a  # (x1, y1)
        self.b = b  # (x2, y2)
        self.raw = raw_line  # original text form for output
        # angles from p to endpoints:
        self.th_a = angle_from_p(px, py, a[0], a[1])
        self.th_b = angle_from_p(px, py, b[0], b[1])

        # Determine the active angular interval for which the ray hits this segment:
        # A ray from p with angle θ intersects segment exactly for θ on smaller
        # circular arc between two endpoint directions. If both endpoint angles are
        # equal, treat as measure-zero
        d = short_arc_direction(self.th_a, self.th_b)
        if feq(d, 0.0):
            # Degenerate: segment faces p at a single direction only
            # We store start=end=th_a and mark degenerate
            self.start = self.th_a
            self.end = self.th_a
            self.wraps = False
            self.degenerate = True
        elif d > 0.0:
            # Short arc goes ccw from th_a to th_b
            self.start = self.th_a
            self.end = self.th_b
            self.wraps = self.start > self.end  # should be False in this branch, kept for safety
            self.degenerate = False
        else:
            # Short arc goes ccw from th_b to th_a
            self.start = self.th_b
            self.end = self.th_a
            self.wraps = self.start > self.end
            self.degenerate = False

    def active_at(self, theta: float) -> bool:
        """
        Is segment intersected by the ray at direction theta from p
        We treat intervals as open at endpoints and closed inside:
            (start, end) (mod 2π), via wraps flag.
        This avoids double counting at exact endpoint angles and
        matches the event processing where we remove then insert.
        """
        if self.degenerate:
            return False  # only active at a single angle; handled at its event
        if not self.wraps:
            return (self.start + EPS) < theta < (self.end - EPS)
        # wraps around 2π → active if theta > start or theta < end
        return theta > (self.start + EPS) or theta < (self.end - EPS)