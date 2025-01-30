from Entries.Material import Material
from Enums.StatusKey import StatusKey
from Managers.EntryManagers.SalesOrderManager import SalesOrderManager
from Managers.SessionManager import SessionManager
from Utils.Time import generate_time
from Utils.Singleton import Singleton


class MaterialManager(metaclass=Singleton):
    """
    Singleton class that holds data for all Materials
    """
    def __init__(self, data: list[Material|object]=None):
        """
        Constructor for MaterialManager, to access existing singleton, create this without parameters
        :param data: None
        """
        self.data = data
        return

    def get_by_id(self, id: int) -> Material|None:
        """
        get a Material by id
        :param id: id of Material
        :return: Material or None
        """
        for material in self.data:
            if material.get_id() == id:
                return material
        return None

    def subtract_material(self, id: int, amount: int) -> StatusKey:
        """
        Subtract from materials stock during sales order, automatically order more material if
        stock gets lower than minimum stock, or sales order in unfulfillable due to not having the materials
        :param id: id of Material
        :param amount: amount of Material
        :return: Status of ORDER action
        """
        material = self.get_by_id(id)
        new_stock = material.stock - amount
        sales_order_manager = SalesOrderManager()
        session_manager = SessionManager()
        current_user_id = session_manager.user.get_id()
        status = StatusKey.ORDERED
        if new_stock < material.min_stock:
            required_amount = material.min_stock - new_stock + 100
            sales_order_manager.create_sales_order(generate_time(), "New", True, required_amount, current_user_id)
            status = StatusKey.AUTO_ORDER
        if new_stock < 0:
            status = StatusKey.DELAYED_ORDER

        material.stock = new_stock
        return status

    def names_ids_amounts(self) -> list[str]:
        """
        Get names and ids for all materials
        :return: list of names and ids
        """
        names_ids = []
        for material in self.data:
            names_ids.append(f"{material.name} ({material.id})")
        return names_ids


def create(data: list[Material|object]) -> MaterialManager:
    """
    Constructor for MaterialManager, do not use this to access existing singleton, use __init__ method instead
    :param data: list of Material objects
    :return: returns created singleton instance for MaterialManager
    """
    return MaterialManager(data)