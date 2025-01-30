from Entries.Bases.ShelfItemBase import ShelfItemBase
from Utils.Parsing import parse_int_or_none, parse_int


class ShelfItem(ShelfItemBase):
    """
    Data class for ShelfItem type
    """
    def __init__(self, id: int, amount: int, shelf_id: int|None, material_id: int|None) -> None:
        """
        Constructor of ShelfItem type
        :param id: id of shelf item
        :param amount: amount of material in shelf item
        :param shelf_id: id of linked shelf, may be None
        :param material_id: is of linked material, may be None
        """
        super().__init__(id, amount)
        self.shelf_id = shelf_id
        self.material_id = material_id
        return

    def __repr__(self):
        """
        JSON representation of shelf item
        :return: JSON string of shelf item
        """
        return (f"{{"
                f"\"id\": \"{self.id}\","
                f"\"amount\": \"{self.amount}\","
                f"\"shelf_id\": \"{self.shelf_id}\","
                f"\"material_id\": \"{self.material_id}\""
                f"}}")

    def get_id(self) -> int:
        """
        Getter for id of shelf item
        :return: id of shelf item
        """
        return self.id

    def get_amount(self) -> int:
        """
        Getter for amount of material in shelf item
        :return: amount of material in shelf item
        """
        return self.amount

    def get_shelf_id(self) -> int:
        """
        Getter for shelf id of linked shelf
        :return: id for linked shelf
        """
        return self.shelf_id

    def get_material_id(self) -> int:
        """
        Getter for material id of linked material
        :return: id for linked material
        """
        return self.material_id


def from_string_list(string_list: list[str]) -> ShelfItem:
    """
    Constructor of ShelfItem type using string list
    :param string_list: string list of JSON object members
    :return: new ShelfItem object
    """
    id = parse_int(string_list[0])
    amount = parse_int(string_list[1])
    shelf_id = parse_int_or_none(string_list[2])
    material_id = parse_int_or_none(string_list[3])
    return ShelfItem(id, amount, shelf_id, material_id)