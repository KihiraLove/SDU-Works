from Entries.Bases.SalesItemBase import SalesItemBase
from Utils.Parsing import parse_int, parse_int_or_none


class SalesItem(SalesItemBase):
    """
    Data class for Sales Item type
    """
    def __init__(self, id: int, amount: int, material_id: int|None, sales_order_id: int|None) -> None:
        """
        Constructor for SalesItem type
        :param id: id of sales item
        :param amount: amount being sold
        :param material_id: material being sold
        :param sales_order_id: id of linked sales order
        """
        super().__init__(id, amount)
        self.material_id = material_id
        self.sales_order_id = sales_order_id
        return

    def __repr__(self):
        """
        JSON representation of sales item
        :return: JSON string of sales item
        """
        return (f"{{"
                f"\"id\": \"{self.id}\","
                f"\"amount\": \"{self.amount}\","
                f"\"material_id\": \"{self.material_id}\","
                f"\"sales_order_id\": \"{self.sales_order_id}\""
                f"}}")

    def get_id(self) -> int:
        """
        Getter for sales item id
        :return: sales item id
        """
        return self.id

    def get_amount(self) -> int:
        """
        Getter for amount being sold
        :return: amount being sold
        """
        return self.amount

    def get_material_id(self) -> int:
        """
        Getter for material being sold
        :return: material being sold
        """
        return self.material_id

    def get_sales_order_id(self) -> int:
        """
        Getter for linked sales order id
        :return: linked sales order id
        """
        return self.sales_order_id


def from_string_list(string_list: list[str]) -> SalesItem:
    """
    Constructor for SalesItem type using string list
    :param string_list: string list of JSON object members
    :return: new SalesItem object
    """
    id = parse_int(string_list[0])
    amount = parse_int(string_list[1])
    material_id = parse_int_or_none(string_list[2])
    sales_order_id = parse_int_or_none(string_list[3])
    return SalesItem(id, amount, material_id, sales_order_id)