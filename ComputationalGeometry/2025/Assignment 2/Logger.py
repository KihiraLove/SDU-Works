import os
from datetime import datetime
from Configuration import Configuration

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
            file.close()

    def debug(self, msg: str) -> None:
        """
        Print a debug message and append it to log if the global :data:`DEBUG` flag is enabled.
        Messages are prefixed with ``[DEBUG]`` so that they can be distinguished
        from higher-level explanatory prints.
        :param msg: Message to print.
        :type msg: str
        :return: None
        """
        if self.debug:
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
        Messages are prefixed with ``[INFO]``
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
        Messages are prefixed with ``[ERROR]``
        :param msg: Message to print.
        :type msg: str
        :return: None
        """
        message = f"[ERROR] {msg}"
        self.append_log(message)
        raise ValueError(message)

    def warn(self, msg: str) -> None:
        """
        Print an WARN message and append it to log
        Messages are prefixed with ``[WARN]``
        :param msg: Message to print.
        :type msg: str
        :return: None
        """
        message = f"[WARN] {msg}"
        print(message)
        self.append_log(message)