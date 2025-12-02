from dataclasses import dataclass
from typing import List, Iterable, Tuple
from PolygonObstacle import PolygonObstacle
from Point import Point

@dataclass
class Environment:
    """
    Polygonal environment consisting of a set of polygonal obstacles.
    The free space is the complement of the union of obstacles. In this simple
    model there is no explicit outer boundary; the plane is assumed to be
    infinite and obstacles are "holes" in it.
    :param obstacles: List of polygonal obstacles.
    :type obstacles: list[PolygonObstacle]
    """
    obstacles: List[PolygonObstacle]

    def all_vertices(self) -> List[Point]:
        """
        Collect all vertices of all obstacles.
        :return: List of all obstacle vertices (possibly with duplicates).
        :rtype: list[Point]
        """
        out: List[Point] = []
        for poly in self.obstacles:
            out.extend(poly.vertices)
        return out

    def all_edges(self) -> Iterable[Tuple[Point, Point]]:
        """
        Iterate over all edges of all obstacles.
        :return: Generator of all obstacle edges.
        :rtype: collections.abc.Iterable[tuple[Point, Point]]
        """
        for poly in self.obstacles:
            yield from poly.edges()