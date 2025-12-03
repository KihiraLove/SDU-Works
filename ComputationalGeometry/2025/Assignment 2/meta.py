import os
from datetime import datetime


class Configuration:
    def __init__(self):
        """
        Container class for running configuration
        :ivar input_file_path: path file containing custom environment, requirement for file format can be found in :class:`io_handler.Input.parse_input_file`
        :ivar output_file_name: name which will be used for output file generation
        :ivar grid_size: size for grid based planner (NxN cells)
        :ivar demo_mode: True to use hard coded problem, False to use custom input
        :ivar debug: True to enable debug mode
        :ivar log_dir: directory to keep logs
        :ivar log_format: time format to use in naming log files
        :ivar log_timestamp_format: format for timestamping log messages
        :ivar output_dir: directory to save output files
        :ivar enable_latex: enable output to LaTeX file
        :ivar enable_matplotlib: enable output Matplotlib plot to png file
        :ivar enable_interactive: show interactive matplotlib plot
        :ivar enable_detailed_report: enable outputting detailed report to log and terminal
        :ivar margin_ratio: margin ratio for the bounding square in the grid
        """
        self.input_file_path: str = "input.txt"
        self.output_file_name: str = "result"
        self.output_dir: str = "output"
        self.demo_mode: bool = True
        self.debug: bool = True
        self.grid_size: int = 64
        self.log_dir: str = "logs"
        self.log_format: str = "%Y%m%d_%H%M%S"
        self.log_timestamp_format: str = "%Y%m%d%H%M%S"
        self.enable_latex: bool = True
        self.enable_matplotlib: bool = True
        self.enable_interactive: bool = False
        self.enable_detailed_report: bool = True
        self.margin_ratio: float = 0.2

        # create dirs if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


class Logger:
    """
    Manages printing and logging
    """
    def __init__(self, config: Configuration) -> None:
        """
        :param config: Running configuration
        :type config: Configuration
        """
        self.config = config
        self.log = ""

    def __del__(self) -> None:
        """
        Write log string to log file at deconstruct time
        :return: None
        """
        with open(os.path.join(self.config.log_dir, f"{datetime.now().strftime(self.config.log_format)}.txt"), "w+") as file:
            file.write(self.log)

    def debug(self, msg: str) -> None:
        """
        Print a debug message and append it to log if debug is enabled.
        Messages are prefixed with [DEBUG] so that they can be distinguished from higher-level explanatory prints.
        :param msg: Message to print.
        :type msg: str
        :return: None
        """
        if self.config.debug:
            message = f"[DEBUG] {msg}"
            self.append_log(message)
            print(message)


    def append_log(self, msg: str) -> None:
        """
        Append a log message to the log string
        :param msg: message to append
        :type msg: str
        :return: None
        """
        self.log += f"{datetime.now().strftime(self.config.log_timestamp_format)} {msg}\n"

    def info(self, msg: str) -> None:
        """
        Print an INFO message and append it to log
        Messages are prefixed with [INFO]
        :param msg: Message to print.
        :type msg: str
        :return: None
        """
        message = f"[INFO] {msg}"
        print(message)
        self.append_log(message)

    def value_error(self, msg: str) -> None:
        """
        Print an ERROR message and append it to log
        Messages are prefixed with [ERROR]
        :param msg: Message to print.
        :type msg: str
        :return: None
        """
        message = f"[ERROR] {msg}"
        print(message)
        self.append_log(message)
        raise ValueError(message)

    def warn(self, msg: str) -> None:
        """
        Print an WARN message and append it to log
        Messages are prefixed with [WARN]
        :param msg: Message to print.
        :type msg: str
        :return: None
        """
        message = f"[WARN] {msg}"
        print(message)
        self.append_log(message)


class Singleton(type):
    """
    Metaclass for Singleton classes to inherit from
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]