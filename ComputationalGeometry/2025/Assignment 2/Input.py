from typing import List
from Point import Point
from PolygonObstacle import PolygonObstacle
from Environment import Environment
from Logger import Logger
from Problem import Problem
from Configuration import Configuration


class Input:
    def __init__(self, logger: Logger, config: Configuration):
        """
        :param logger: logger object
        :type logger: Logger
        :param config: configuration object
        :type config: Configuration
        """
        self.logger = logger
        self.config = config

    def create_problem_from_file_or_demo(self) -> Problem:
        """
        :return: object containing the environment, start, and goal
        :rtype: Problem
        """
        if self.config.demo_mode or self.config.input_file_path is None:
            self.logger.info("Using built-in demo environment.")
            return self.build_demo_environment()
        self.logger.info(f"Reading custom environment from: {self.config.input_file_path}")
        return self.parse_input_file()

    def parse_input_file(self) -> Problem:
        """
        Parse a custom input file describing the environment, start, and goal.
        The format is a simple, line-oriented text format designed for ease of manual editing.
        Lines starting with ``#`` are treated as comments and ignored.
        Blank lines are ignored.

        The file have to contain:

        * A single ``START`` line.
        * A single ``GOAL`` line.
        * One or more obstacle definitions.

        The grammar is as follows (pseudo-BNF):
           file           ::= { comment | start | goal | obstacle | blank }*
           comment        ::= '#' <rest of line>
           blank          ::= <empty line> | whitespace-only line
           start          ::= 'START' x y
           goal           ::= 'GOAL'  x y
           obstacle       ::= 'OBSTACLE' newline vertex+ 'END'
           vertex         ::= x y

        where ``x`` and ``y`` are floating-point numbers.

        :param input_file_path: Path to the input file.
        :type input_file_path: str
        :return: object containing the environment, start, and goal
        :rtype: Problem
        :raises ValueError: If the file is malformed or misses required elements.
        """
        self.logger.info(f"Parsing input file {self.config.input_file_path}")
        obstacles: List[PolygonObstacle] = []
        start: Point = None
        goal: Point = None

        with open(self.config.input_file_path, "r", encoding="utf8") as f:
            lines = f.readlines()
            self.logger.debug(f"Input file:\n{f.read()}")

        i = 0
        n = len(lines)

        while i < n:
            line = lines[i].strip()

            # Skip comments and empty lines.
            if not line or line.startswith("#"):
                i += 1
                continue

            tokens = line.split()

            if tokens[0].upper() == "START":
                if len(tokens) != 3:
                    self.logger.value_error(f"Invalid START line at {i + 1}")
                x = float(tokens[1])
                y = float(tokens[2])
                start = Point(x, y)
                self.logger.info(f"Read START at line {i + 1}: ({x}, {y}).")
                i += 1
                continue

            if tokens[0].upper() == "GOAL":
                if len(tokens) != 3:
                    self.logger.value_error(f"Invalid GOAL line at {i + 1}")
                x = float(tokens[1])
                y = float(tokens[2])
                goal = Point(x, y)
                self.logger.info(f"Read GOAL at line {i + 1}: ({x}, {y}).")
                i += 1
                continue

            if tokens[0].upper() == "OBSTACLE":
                # Read vertices until "END".
                self.logger.info(f"Reading OBSTACLE starting at line {i + 1}.")
                vertices: List[Point] = []
                i += 1
                while i < n:
                    line2 = lines[i].strip()
                    if not line2 or line2.startswith("#"):
                        i += 1
                        continue
                    if line2.upper() == "END":
                        self.logger.info(f"Finished OBSTACLE at line {i + 1} with {len(vertices)} vertices.")
                        i += 1
                        break
                    parts = line2.split()
                    if len(parts) != 2:
                        self.logger.value_error(f"Invalid obstacle vertex at line {i + 1}")
                    x = float(parts[0])
                    y = float(parts[1])
                    vertices.append(Point(x, y))
                    i += 1

                if len(vertices) < 3:
                    self.logger.value_error("Obstacle with fewer than 3 vertices.")
                obstacles.append(PolygonObstacle(vertices=vertices))
                continue

            # If the line does not match any known keyword, treat it as an error.
            self.logger.value_error(f"Unrecognized keyword in line {i + 1}: {line!r}")

        if start is None:
            self.logger.value_error("Input file does not define a START.")
        if goal is None:
            self.logger.value_error("Input file does not define a GOAL.")

        self.logger.info(f"Finished parsing input file. Parsed {len(obstacles)} obstacles, one START and one GOAL.")

        return Problem(Environment(obstacles=obstacles), start, goal)

    @staticmethod
    def build_demo_environment() -> Problem:
        """
        Builds a small demo environment
        The environment consists of two axis-aligned rectangular obstacles.
        A start and a goal point are placed on opposite sides of these obstacles.
        :return: Problem object containing the environment, start, and goal
        :rtype: tuple[Environment, Point, Point]
        """
        rect1 = PolygonObstacle(
            vertices=[
                Point(2.0, 1.0),
                Point(4.0, 1.0),
                Point(4.0, 3.0),
                Point(2.0, 3.0),
            ]
        )
        rect2 = PolygonObstacle(
            vertices=[
                Point(6.0, 2.0),
                Point(8.0, 2.0),
                Point(8.0, 4.0),
                Point(6.0, 4.0),
            ]
        )
        env = Environment(obstacles=[rect1, rect2])
        start = Point(1.0, 2.0)
        goal = Point(9.0, 3.0)
        return Problem(
            env=env,
            start=start,
            goal=goal
        )