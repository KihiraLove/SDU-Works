"""
author: doker24, Domonkos Kertész
Computational geometry, fall 2025
also available:
https://github.com/KihiraLove/SDU-Works/tree/main/ComputationalGeometry/2025/Assignment%202

### Info ###
This program calls an external 'pdflatex' subprocess on your terminal to generate a PDF.
If 'pdflatex' is not available, or the subprocess fails, .tex file will be available for manual use.

### Packages ###
Apart from standard packages, this program optionally uses matplotlib.pyplot to generate a .png,
but can run without it, only outputting a LaTeX and a PDF document.

standard packages used:
- os
- subprocess
- typing
- math
- heapq
- collections
- dataclasses
- datetime
"""

from meta import Configuration, Logger
from io_handler import Input, Output
from planning import Runner

def main():
    """
    Top level driver. Pipeline:
    - Load running configuration
    - Create Logger
    - Create input reader
    - Build or load environment
    - Create runner
    - Run planners
    - Create output writer
    - Generate and save output
    """
    config = Configuration()

    logger = Logger(
        config=config
    )

    input_reader = Input(
        logger=logger,
        config=config
    )
    problem = input_reader.create_problem_from_file_or_demo()

    runner = Runner(
        problem=problem,
        logger=logger,
        config=config
    )
    result = runner.run_planners()

    output = Output(
        result=result,
        logger=logger,
        config=config
    )
    output.generate_and_save()

if __name__ == '__main__':
    main()