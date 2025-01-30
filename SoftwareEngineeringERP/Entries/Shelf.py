from Entries.Bases.ShelfBase import ShelfBase
from Utils.Parsing import parse_int


class Shelf(ShelfBase):
    """
    Data class for Shelf type
    """
    def __init__(self, id: int, location: str) -> None:
        """
        Constructor for Shelf type
        :param id: id of shelf
        :param location: location of shelf
        """
        super().__init__(id, location)
        return

    def __repr__(self):
        """
        JSON representation of shelf
        :return: JSON string of shelf
        """
        return (f"{{"
                f"\"id\": \"{self.id}\","
                f"\"location\": \"{self.location}\""
                f"}}")

    def get_id(self) -> int:
        """
        Getter for id of shelf
        :return: id of shelf
        """
        return self.id

    def get_location(self) -> str:
        """
        Getter for location of shelf
        :return: location of shelf
        """
        return self.location


def from_string_list(string_list: list[str]) -> Shelf:
    """
    Constructor for Shelf type using string list
    :param string_list: string list of JSON object members
    :return: new Shelf object
    """
    id = parse_int(string_list[0])
    location = string_list[1]
    return Shelf(id, location)