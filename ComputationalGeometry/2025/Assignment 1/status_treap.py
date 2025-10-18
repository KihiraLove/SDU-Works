import math
from typing import Optional
from distance_oracle import DistanceOracle
from treap_node import TreapNode, _rotate_right, _rotate_left
from segment import Segment
from math_functions import feq


class StatusTreap:
    """
    Status structure T (balanced BST) storing the ordered sequence of
    segments intersecting the current ray direction. Order is defined
    by closest first at the*current θ. Between events, relative order
    of active segments does not change, so the treap stays valid when θ changes.
    Insertions/removals only happen at event angles.

    Compare segments A,B using DistanceOracle at the current θ. Ties
    are broken by segment id to force a total order.
    """
    __slots__ = ("root", "oracle")

    def __init__(self, oracle: DistanceOracle):
        self.root: Optional[TreapNode] = None
        self.oracle = oracle

    def _less(self, a: Segment, b: Segment) -> bool:
        if a.id == b.id:
            return False
        da = self.oracle.ray_segment_distance(a)
        db = self.oracle.ray_segment_distance(b)
        if math.isinf(da) and math.isinf(db):
            return a.id < b.id
        if math.isinf(da):
            return False
        if math.isinf(db):
            return True
        if not feq(da, db):
            return da < db
        return a.id < b.id

    def insert(self, seg: Segment) -> None:
        """
        Insert into tree
        :param seg: segment
        :return: None
        """
        def _insert(node: Optional[TreapNode], s: Segment) -> TreapNode:
            if node is None:
                return TreapNode(s)
            if self._less(s, node.seg):
                node.left = _insert(node.left, s)
                if node.left.prio < node.prio:
                    node = _rotate_right(node)
            elif self._less(node.seg, s):
                node.right = _insert(node.right, s)
                if node.right.prio < node.prio:
                    node = _rotate_left(node)
            else:
                # already present; do nothing
                return node
            return node
        self.root = _insert(self.root, seg)

    def remove(self, seg: Segment) -> None:
        """
        Remove from tree
        :param seg: segment
        :return: None
        """
        def _remove(node: Optional[TreapNode], s: Segment) -> Optional[TreapNode]:
            if node is None:
                return None
            if s.id == node.seg.id:
                # delete this node
                if node.left is None and node.right is None:
                    return None
                if node.left is None:
                    node = _rotate_left(node)
                elif node.right is None:
                    node = _rotate_right(node)
                else:
                    # rotate by smaller priority
                    if node.left.prio < node.right.prio:
                        node = _rotate_right(node)
                    else:
                        node = _rotate_left(node)
                return _remove(node, s)
            if self._less(s, node.seg):
                node.left = _remove(node.left, s)
            else:
                node.right = _remove(node.right, s)
            return node
        self.root = _remove(self.root, seg)

    def min_segment(self) -> Optional[Segment]:
        """
        :return: currently nearest active segment, or None if empty
        """
        node = self.root
        if node is None:
            return None
        while node.left is not None:
            node = node.left
        return node.seg

    def clear(self) -> None:
        """
        Clear root of Treap
        :return: None
        """
        self.root = None