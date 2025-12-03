from dataclasses import dataclass
from Point import Point


@dataclass
class GridCell:
    """
    Simple data holder for a grid cell used in the quadtree-based planner.

    In this simplified implementation, we build a *uniform* grid of square
    cells. Conceptually, this corresponds to a complete quadtree of fixed
    depth. Each leaf node in this quadtree can therefore be identified by
    its integer grid coordinates ``(i, j)``.

    :ivar i: Column index of the cell (0-based).
    :type i: int
    :ivar j: Row index of the cell (0-based).
    :type j: int
    :ivar center: Center point of the cell in world coordinates.
    :type center: Point
    :ivar blocked: Flag indicating whether this cell is considered occupied
                   by an obstacle.
    :type blocked: bool
    """
    i: int
    j: int
    center: Point
    blocked: bool = False