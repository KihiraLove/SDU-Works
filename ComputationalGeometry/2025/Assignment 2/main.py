from Input import Input
from Output import Output
from Logger import Logger
from Runner import Runner
from Configuration import Configuration


"""
author: doker24, Domonkos Kertész
Computational geometry, fall 2025
also available: https://github.com/KihiraLove/SDU-Works/tree/main/ComputationalGeometry/2025/Assignment%202
"""

### Important ###
# This program calls a subprocess on your terminal to generate a pdf,
# it requires pdflatex to be available.
# I only tested it on one machine, if it fails on yours, .tex file will be available to be used manually

### Packages ###
# Apart from standard packages, this program uses matplotlib.pyplot but can run without it
# standard packages used:
# - os
# - subprocess
# - typing
# - math
# - heapq
# - collections
# - dataclasses
# - datetime

def main():
    # Load running configuration
    config = Configuration()

    # Create Logger
    logger = Logger(
        config=config
    )

    # Create input reader and build or load environment
    input = Input(
        logger=logger,
        config=config
    )
    problem = input.create_problem_from_file_or_demo()

    # Create runner and run planners
    runner = Runner(
        problem=problem,
        logger=logger,
        config=config
    )
    result = runner.run_planners()

    # Create output writer, generate and save output
    output = Output(
        result=result,
        logger=logger,
        config=config
    )
    output.generate_and_save()

if __name__ == '__main__':
    main()