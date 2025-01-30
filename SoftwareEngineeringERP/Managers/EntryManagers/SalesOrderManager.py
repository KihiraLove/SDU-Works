from datetime import datetime

from Entries.SalesOrder import SalesOrder
from Utils.Singleton import Singleton


class SalesOrderManager(metaclass=Singleton):
    """
    Singleton class that holds data for all SalesOrders
    """
    def __init__(self, data: list[SalesOrder|object]=None):
        """
        Constructor for SalesOrderManager, to access existing singleton, create this without parameters
        :param data: None
        """
        self.data = data
        return

    def create_sales_order(self, date: datetime, status: str, is_inbound: bool, business_partner_id: int, id_of_current_user: int) -> int:
        """
        Create new SalesOrder and save it to SalesOrderManager
        :param date: date and time of sales order
        :param status: status of sales order
        :param is_inbound: is sales order inbound
        :param business_partner_id: id of linked business partner
        :return: id of new sales order
        """
        id = len(self.data)
        self.data.append(SalesOrder(id , date, status, is_inbound, business_partner_id, id_of_current_user))
        return id

    def get_by_bp_id(self, bp_id) -> list[SalesOrder]:
        """
        Get SalesOrder by Business Partner ID
        :param bp_id: Business Partner ID
        :return: all orders from Business Partner
        """
        orders = []
        for order in self.data:
            if order.business_partner_id == bp_id:
                orders.append(order)
        return orders

    def get_by_id(self, id: int) -> SalesOrder:
        """
        Get SalesOrder by ID
        :param id: id of sales order
        :return: sales order by id
        """
        for order in self.data:
            if order.id == id:
                return order

    def get_by_user_id(self, user_id: int) -> list[SalesOrder]:
        """
        Get SalesOrder by creator Users ID
        :param user_id: creator Users ID
        :return: all orders from creator User
        """
        orders = []
        for order in self.data:
            if order.user_id == user_id:
                orders.append(order)
        return orders


def create(data: list[SalesOrder|object]) -> SalesOrderManager:
    """
    Constructor for SalesOrderManager, do not use this to access existing singleton, use __init__ method instead
    :param data: list of SalesOrder objects
    :return: returns created singleton instance for SalesOrderManager
    """
    return SalesOrderManager(data)