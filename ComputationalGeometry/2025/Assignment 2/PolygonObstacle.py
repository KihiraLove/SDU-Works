from dataclasses import dataclass
from typing import List, Iterable, Tuple
from Point import Point

@dataclass
class PolygonObstacle:
    """
    Simple polygonal obstacle described by its vertices in counter-clockwise order
    The polygon is assumed to be simple (non-self-intersecting) and closed implicitly
    that is, there is an edge from the last vertex back to the first one
    :param vertices: Vertices of the polygon in counter-clockwise order
    :type vertices: list[Point]
    """
    vertices: List[Point]

    def edges(self) -> Iterable[Tuple[Point, Point]]:
        """
        Iterate over polygon edges as pairs of endpoints
        The last vertex is connected back to the first one
        :return: Generator of edges (u, v)
        :rtype: Iterable[tuple[Point, Point]]
        """
        n = len(self.vertices)
        for i in range(n):
            yield self.vertices[i], self.vertices[(i + 1) % n]

    def to_tikz(self) -> str:
        """
        Create Tikz representation of polygon
        :return: tikz representation of polygon
        :rtype: str
        """
        tikz = "\\filldraw[fill=gray!20,draw=black] "
        for point in self.vertices:
            tikz += point.to_tikz_line_from_point()
        # Closing the polygon off by connecting back to the first point
        tikz += self.vertices[0].to_tikz_line_end()
        return tikz