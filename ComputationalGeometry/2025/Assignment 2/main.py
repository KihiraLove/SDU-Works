from Input import Input
from Output import Output
from Logger import Logger
from Runner import Runner
from Configuration import Configuration


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