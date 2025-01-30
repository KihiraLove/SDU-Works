from datetime import datetime
from Utils import Config


def generate_time() -> datetime:
    """
    Get current datetime as formatted datatime object without having to use this abomination of a code at a million places
    :return: Current datetime with format
    """
    return datetime.strptime(datetime.now().strftime(Config.TIME_FORMAT), Config.TIME_FORMAT)
