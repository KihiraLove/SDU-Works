from Entries.Shelf import Shelf
from Utils.Singleton import Singleton


class ShelfManager(metaclass=Singleton):
    """
    Singleton class that holds data for all Shelfs
    """
    def __init__(self, data: list[Shelf|object]=None):
        """
        Constructor for ShelfManager, to access existing singleton, create this without parameters
        :param data: None
        """
        self.data = data
        return


def create(data: list[Shelf|object]) -> ShelfManager:
    """
    Constructor for ShelfManager, do not use this to access existing singleton, use __init__ method instead
    :param data: list of Shelf objects
    :return: returns created singleton instance for ShelfManager
    """
    return ShelfManager(data)