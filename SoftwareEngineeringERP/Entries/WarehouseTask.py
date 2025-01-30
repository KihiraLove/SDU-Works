from datetime import datetime
from Entries.Bases.WarehouseTaskBase import WarehouseTaskBase
from Utils import Config
from Utils.Parsing import parse_int, parse_int_or_none, parse_datetime


class WarehouseTask(WarehouseTaskBase):
    """
    Data class for WarehouseTask type
    """
    def __init__(self, id: int, date: datetime, status: str, sales_order_id: int|None, user_id: int|None) -> None:
        """
        Constructor for WarehouseTask type
        :param id: id of warehouse task
        :param date: date when warehouse task was created
        :param status: status of warehouse task
        :param sales_order_id: id of linked sales order
        :param user_id: id of linked user
        """
        super().__init__(id, date, status)
        self.sales_order_id = sales_order_id
        self.user_id = user_id
        return

    def __repr__(self):
        """
        JSON representation of WarehouseTask
        :return: JSON string of WarehouseTask
        """
        return (f"{{"
                f"\"id\": \"{self.id}\","
                f"\"date\": \"{self.date.strftime(Config.TIME_FORMAT)}\","
                f"\"status\": \"{self.status}\","
                f"\"sales_order_id\": \"{self.sales_order_id}\","
                f"\"user_id\": \"{self.user_id}\""
                f"}}")

    def get_id(self) -> int:
        """
        Getter for id of warehouse task
        :return: id of warehouse task
        """
        return self.id

    def get_date(self) -> datetime:
        """
        Getter for creation date of warehouse task
        :return: creation date of warehouse task
        """
        return self.date

    def get_status(self) -> str:
        """
        Getter for status of warehouse task
        :return: status of warehouse task
        """
        return self.status

    def get_sales_order_id(self) -> int:
        """
        Getter for linked sales order id of warehouse task
        :return: linked sales order id of warehouse task
        """
        return self.sales_order_id

    def get_user_id(self) -> int:
        """
        Getter for linked user id of warehouse task
        :return: linked user id of warehouse task
        """
        return self.user_id


def from_string_list(string_list: list[str]) -> WarehouseTask:
    """
    Constructor for WarehouseTask type using string list
    :param string_list: string list of JSON object members
    :return:
    """
    id = parse_int(string_list[0])
    date = parse_datetime(string_list[1])
    status = string_list[2]
    sales_order_id = parse_int_or_none(string_list[3])
    user_id = parse_int_or_none(string_list[4])
    return WarehouseTask(id, date, status, sales_order_id, user_id)