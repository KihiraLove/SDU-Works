from datetime import datetime
from Entries.Bases.SalesOrderBase import SalesOrderBase
from Utils.Parsing import parse_bool, parse_int_or_none, parse_int, parse_datetime


class SalesOrder(SalesOrderBase):
    """
    Data class for SalesOrder type
    """
    def __init__(self, id: int, date: datetime, status: str, is_inbound: bool, business_partner_id: int|None, user_id: int) -> None:
        """
        Constructor for SalesOrder type
        :param id: id of sales order
        :param date: date of sales order
        :param status: status of sales order
        :param is_inbound: is sales order inbound
        :param business_partner_id: id of linked business partner, may be None
        :param user_id: id of creator user
        """
        super().__init__(id, date, status, is_inbound)
        self.business_partner_id = business_partner_id
        self.user_id = user_id
        return

    def __repr__(self):
        """
        JSON representation of sales order
        :return: JSON string of sales order
        """
        return (f"{{"
                f"\"id\": \"{self.id}\","
                f"\"date\": \"{self.date}\","
                f"\"status\": \"{self.status}\","
                f"\"is_inbound\": \"{self.is_inbound}\","
                f"\"business_partner_id\": \"{self.business_partner_id}\","
                f"\"user_id\": \"{self.user_id}\""
                f"}}")

    def get_id(self) -> int:
        """
        Getter for sales order id
        :return: id of sales order
        """
        return self.id

    def get_date(self) -> datetime:
        """
        Getter for sales order date
        :return: date of sales order
        """
        return self.date

    def get_status(self) -> str:
        """
        Getter for sales order status
        :return: status of sales order
        """
        return self.status

    def get_is_inbound(self) -> bool:
        """
        Getter for sales order is_inbound
        :return: is sales order inbound
        """
        return self.is_inbound

    def get_business_partner_id(self) -> int:
        """
        Getter for id of linked business partner
        :return: id of linked business partner
        """
        return self.business_partner_id


def from_string_list(string_list: list[str]) -> SalesOrder:
    """
    Constructor for SalesOrder type using string list
    :param string_list: string list of JSON object members
    :return: new SalesOrder object
    """
    id = parse_int(string_list[0])
    date = parse_datetime(string_list[1])
    status = string_list[2]
    is_inbound = parse_bool(string_list[3])
    business_partner_id = parse_int_or_none(string_list[4])
    user_id = parse_int_or_none(string_list[5])
    return SalesOrder(id, date, status, is_inbound, business_partner_id, user_id)