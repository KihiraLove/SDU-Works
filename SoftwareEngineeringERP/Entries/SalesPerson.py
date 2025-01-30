from Entries.Bases.SalesPersonBase import SalesPersonBase
from Utils.Parsing import parse_bool, parse_int, parse_int_or_none


class SalesPerson(SalesPersonBase):
    """
    Data class for SalesPerson type
    """
    def __init__(self, id: int, name: str, phone_number: str, email: str, is_internal: bool, user_id: int|None) -> None:
        """
        Constructor for SalesPerson type
        :param id: id of sales person
        :param name: name of sales person
        :param phone_number: phone number of sales person
        :param email: email of sales person
        :param is_internal: is sales person internal or not
        :param user_id: user id of internal sales person, None if external
        """
        super().__init__(id, name, phone_number, email, is_internal)
        self.user_id = user_id
        return

    def __repr__(self):
        """
        JSON representation of sales person
        :return: JSON string of sales person
        """
        return (f"{{"
                f"\"id\": \"{self.id}\","
                f"\"name\": \"{self.name}\","
                f"\"phone_number\": \"{self.phone_number}\","
                f"\"email\": \"{self.email}\","
                f"\"is_internal\": \"{self.is_internal}\","
                f"\"user_id\": \"{self.user_id}\""
                f"}}")

    def get_id(self) -> int:
        """
        Getter for sales person id
        :return: id of sales person
        """
        return self.id

    def get_name(self) -> str:
        """
        Getter for sales person name
        :return: name of sales person
        """
        return self.name

    def get_phone_number(self) -> str:
        """
        Getter for sales person phone number
        :return: phone number of sales person
        """
        return self.phone_number

    def get_email(self) -> str:
        """
        Getter for sales person email
        :return: email of sales person
        """
        return self.email

    def get_is_internal(self) -> bool:
        """
        Getter for sales person is_internal
        :return: whether sales person is internal or not
        """
        return self.is_internal


def from_string_list(string_list: list[str]) -> SalesPerson:
    """
    Constructor for SalesPerson type using string list
    :param string_list: string list of JSON object members
    :return: new SalesPerson object
    """
    id = parse_int(string_list[0])
    name = string_list[1]
    phone_number = string_list[2]
    email = string_list[3]
    is_internal = parse_bool(string_list[4])
    user_id = parse_int_or_none(string_list[5])
    return SalesPerson(id, name, phone_number, email, is_internal, user_id)