from Entries.Bases.LabelBase import LabelBase
from Utils.Parsing import parse_int_or_none, parse_int


class Label(LabelBase):
    """
    Data class for Label type
    """
    def __init__(self, id: int, sales_order_id: int|None) -> None:
        """
        Constructor for Label type
        :param id: id of label
        :param sales_order_id: id of linked sales order object, may be None
        """
        super().__init__(id)
        self.sales_order_id = sales_order_id
        return

    def __repr__(self):
        """
        JSON representation of label
        :return: JSON string of label
        """
        return (f"{{"
                f"\"id\": \"{self.id}\","
                f"\"sales_order_id\": \"{self.sales_order_id}\""
                f"}}")

    def get_id(self) -> int:
        """
        Getter for id
        :return: id of label
        """
        return self.id

    def get_sales_order_id(self) -> int:
        """
        Getter for linked sales order id
        :return: id of linked sales order id
        """
        return self.sales_order_id


def from_string_list(string_list: list[str]) -> Label:
    """
    Constructor for Label type using string list
    :param string_list: string list of JSON object members
    :return: new Label object
    """
    id = parse_int(string_list[0])
    sales_order_id = parse_int_or_none(string_list[1])
    return Label(id, sales_order_id)