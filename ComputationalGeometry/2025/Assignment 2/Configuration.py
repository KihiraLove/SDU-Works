class Configuration:
    def __init__(self):
        """
        Container class for running configuration
        :config input_file_path: path file containing custom environment, requirement for file format can be found in Input.py
        :config output_file_name: name which will be used for output file generation
        :config grid_size: size for grid based planner (NxN cells)
        :config demo_mode: True to use hard coded problem, False to use custom input
        :config debug: True to enable debug mode
        :config log_dir: directory to keep logs
        :config log_format: time format to use in naming log files
        :config log_timestamp_format: format for timestamping log messages
        :config output_dir: directory to save output files
        :config enable_latex: enable output to LaTeX file
        :config enable matplotlib: enable output Matplotlib plot to png file
        :config enable_interactive: show interactive matplotlib plot
        :config enable_detailed_report: enable outputting detailed report to log and terminal
        :config margin_ratio: margin ratio for the bounding square in the grid
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
