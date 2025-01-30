from Entries.WarehouseTask import WarehouseTask
from Utils.Singleton import Singleton


class WarehouseTaskManager(metaclass=Singleton):
    """
    Singleton class that holds data for all WarehouseTasks
    """
    def __init__(self, data: list[WarehouseTask|object]=None):
        """
        Constructor for WarehouseTaskManager, to access existing singleton, create this without parameters
        :param data: None
        """
        self.data = data
        return


def create(data: list[WarehouseTask|object]) -> WarehouseTaskManager:
    """
    Constructor for WarehouseTaskManager, do not use this to access existing singleton, use __init__ method instead
    :param data: list of WarehouseTask objects
    :return: returns created singleton instance for WarehouseTaskManager
    """
    return WarehouseTaskManager(data)