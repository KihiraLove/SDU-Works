from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    """
    Simple 2D point

    The class is immutable and hashable so that points can be used as dictionary
    keys or elements in sets.

    :param x: X-coordinate.
    :type x: float
    :param y: Y-coordinate.
    :type y: float
    """
    x: float
    y: float

    def to_tikz_line_from_point(self) -> str:
        """
        Tikz representation of a line from this point
        :return: tikz representation of a line from this point
        :rtype: str
        """
        return f"({self.x},{self.y}) -- "

    def to_tikz_line_end(self) -> str:
        """
        Tikz representation of a line ending on this point
        :return: tikz representation of a line ending on this point
        :rtype: str
        """
        return f"({self.x},{self.y});\n"

    def to_tikz(self, colour: str) -> str:
        """
        Tikz representation of point
        :param colour: colour of the point
        :type colour: str
        :return: tikz representation of point
        :rtype: str
        """
        return f"\\filldraw[{colour}!70!black] ({self.x},{self.y}) circle[radius=0.08]; % start\n"