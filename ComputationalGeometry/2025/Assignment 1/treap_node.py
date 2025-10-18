import random
from typing import Optional
from segment import Segment

class TreapNode:
    """
    Node for randomized balanced binary search tree
    """
    __slots__ = ("seg", "prio", "left", "right")
    def __init__(self, seg: Segment):
        self.seg = seg
        self.prio = random.random()
        self.left: Optional[TreapNode] = None
        self.right: Optional[TreapNode] = None

def _rotate_right(y: TreapNode) -> TreapNode:
    x = y.left
    y.left = x.right  # type: ignore
    x.right = y       # type: ignore
    return x

def _rotate_left(x: TreapNode) -> TreapNode:
    y = x.right
    x.right = y.left  # type: ignore
    y.left = x        # type: ignore
    return y