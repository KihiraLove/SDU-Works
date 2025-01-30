import math


class Point:
    def __init__(self, x: float, y: float) -> None:
        """
        A simple point in 2D space
        :param x: X
        :param y: Y
        """
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        """
        Helper function for printing values
        :return: string representation of a point
        """
        return f"Point({self.x}, {self.y})"


class LineSegment:
    def __init__(self, p1: Point, p2: Point) -> None:
        """
        A simple line segment between two points
        :param p1: Point 1
        :param p2: Point 2
        """
        self.p1 = p1
        self.p2 = p2

    def __repr__(self) -> str:
        """
        Helper function for printing values
        :return: string representation of a line segment
        """
        return f"LineSegment({self.p1} -> {self.p2})"

    def to_tikz(self, visible: bool) -> str:
        """
        Function to generate the latex representation of a line segment
        :param visible: whether the line segment is visible or not
        :return: latex representation of a line segment
        """
        color = "blue" if visible else "black"
        tikz_segment = f"\\draw[{color}] ({self.p1.x}, {self.p1.y}) -- ({self.p2.x}, {self.p2.y});\n"
        tikz_segment += f"\\fill[{color}] ({self.p2.x}, {self.p2.y}) circle (5pt);\n"
        tikz_segment += f"\\fill[{color}] ({self.p1.x}, {self.p1.y}) circle (5pt);\n"
        return tikz_segment

    def angle(self, p: Point) -> list[float]:
        """
        Calculate the angle from point p to the segment
        :param p: viewpoint
        :return: sorted angles for the two end points of a line segment
        """
        angle1 = math.atan2(self.p1.y - p.y, self.p1.x - p.x)
        angle2 = math.atan2(self.p2.y - p.y, self.p2.x - p.x)
        return sorted([angle1, angle2])

    def distance(self, p: Point)-> float:
        """
        Calculate the distance from point p to the segment
        :param p: viewpoint
        :return: shorter distance of the distances from the viewpoint to teh ending points of a line ending
        """
        return min(math.hypot(self.p1.x - p.x, self.p1.y - p.y),
                   math.hypot(self.p2.x - p.x, self.p2.y - p.y))


class Event:
    def __init__(self, angle: float, segment: LineSegment, is_start: bool) -> None:
        """
        Class to hold events
        :param angle: angle from viewpoint
        :param segment: segment the event belongs to
        :param is_start: whether the event is a start or end event
        """
        self.angle = angle
        self.segment = segment
        self.is_start = is_start

    def __repr__(self) -> str:
        """
        Helper function for printing values
        :return: string representation of an event
        """
        event_type = "Start" if self.is_start else "End"
        return f"{event_type} Event at angle {self.angle} for {self.segment}"