import math

EPS = 1e-12  # global epsilon for robust zero tests

def feq(a: float, b: float, eps: float = EPS) -> bool:
    """
    Return True if two floats, a and b are nearly equal
    :param a: value one
    :param b: value two
    :param eps: float precision tolerance
    :return: whether two floats are equal within tolerance
    """
    return abs(a - b) <= eps

def cross(ax: float, ay: float, bx: float, by: float) -> float:
    """
    2D cross product (a x b) = ax*by - ay*bx
    :param ax: x of vector one
    :param ay: y of vector one
    :param bx: x of vector two
    :param by: y of vector two
    :return: cross product
    """
    return ax * by - ay * bx

def dot(ax: float, ay: float, bx: float, by: float) -> float:
    """
    2D dot product
    :param ax: x of vector one
    :param ay: y of vector one
    :param bx: x of vector two
    :param by: y of vector two
    :return: dot product
    """
    return ax * bx + ay * by

def orient(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    """
    Oriented area / orientation test
    > 0  => C is to the left of AB
    < 0  => C is to the right of AB
    = 0  => A, B, C are collinear
    Implemented as cross(B-A, C-A)
    """
    return cross(bx - ax, by - ay, cx - ax, cy - ay)

def angle_from_p(px: float, py: float, qx: float, qy: float) -> float:
    """
    Angle in [0, 2π) from p to q, using atan2
    """
    theta = math.atan2(qy - py, qx - px)
    if theta < 0.0:
        theta += 2.0 * math.pi
    return theta

def wrap_angle(a: float) -> float:
    """
    Wrap angle to [0, 2π)
    """
    twopi = 2.0 * math.pi
    a = a % twopi
    if a < 0:
        a += twopi
    return a

def short_arc_direction(a: float, b: float) -> float:
    """
    Signed shortest angular difference b - a, mapped to (-π, π]
    Positive => ccw from a to b is shorter direction
    """
    twopi = 2.0 * math.pi
    d = (b - a) % twopi
    if d > math.pi:
        d -= twopi
    return d