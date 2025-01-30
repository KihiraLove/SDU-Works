from Entries.Bases.MaterialBase import MaterialBase
from Utils.Parsing import parse_int


class Material(MaterialBase):
    """
    Data class for Material type
    """
    def __init__(self, id: int, ext_id: int, name: str, min_stock: int, stock: int) -> None:
        """
        Constructor of Material type
        :param id: id of material
        :param ext_id: external id of material
        :param name: name of material
        :param min_stock: minimum stock of material
        :param stock: current stock of material
        """
        super().__init__(id, ext_id, name, min_stock, stock)
        return

    def __repr__(self):
        """
        JSON representation of material
        :return: JSON string of material
        """
        return (f"{{"
                f"\"id\": \"{self.id}\","
                f"\"ext_id\": \"{self.ext_id}\","
                f"\"name\": \"{self.name}\","
                f"\"min_stock\": \"{self.min_stock}\","
                f"\"stock\": \"{self.stock}\""
                f"}}")

    def get_id(self) -> int:
        """
        Getter for id of material
        :return: id of material
        """
        return self.id

    def get_ext_id(self) -> int:
        """
        Getter for external id of material
        :return: external id of material
        """
        return self.ext_id

    def get_name(self) -> str:
        """
        Getter for name of material
        :return: name of material
        """
        return self.name

    def get_min_stock(self) -> int:
        """
        Getter for minimum stock of material
        :return: minimum stock of material
        """
        return self.min_stock

    def get_stock(self) -> int:
        """
        Getter for current stock of material
        :return: current stock of material
        """
        return self.stock


def from_string_list(string_array: list[str]) -> Material:
    """
    Constructor of Material type using string list
    :param string_array: string list of JSON object members
    :return: new Material object
    """
    id = parse_int(string_array[0])
    ext_id = parse_int(string_array[1])
    name = string_array[2]
    min_stock = parse_int(string_array[3])
    stock = parse_int(string_array[4])
    return Material(id, ext_id, name, min_stock, stock)