from Entries.Bases.UserBase import UserBase
from Enums.UserType import UserType
from Utils.Parsing import parse_int, parse_user_type


class User(UserBase):
    """
    Data class for User type
    """
    def __init__(self, id: int, name: str, email: str, password: str, user_type: UserType) -> None:
        """
        Constructor for User type
        :param id: id of user
        :param name: name of user
        :param email: email of user
        :param password: password of user
        :param user_type: type of user
        """
        super().__init__(id, name, email, password, user_type)
        return

    def __repr__(self):
        """
        JSON representation of User
        :return: JSON string of User
        """
        return (f"{{"
                f"\"id\": \"{self.id}\","
                f"\"name\": \"{self.name}\","
                f"\"email\": \"{self.email}\","
                f"\"password\": \"{self.password}\","
                f"\"user_type\": \"{self.user_type}\""
                f"}}")

    def get_id(self) -> int:
        """
        Getter for id of user
        :return: id of user
        """
        return self.id

    def get_name(self) -> str:
        """
        Getter for name of user
        :return: name of user
        """
        return self.name

    def get_password(self) -> str:
        """
        Getter for password of user
        :return: password of user
        """
        return self.password

    def get_user_type(self) -> UserType:
        """
        Getter for type of user
        :return: type of user
        """
        return self.user_type

    def password_matches(self, password: str) -> bool:
        if password != self.password:
            return False
        return True


def from_string_list(string_list: list[str]) -> User:
    """
    Constructor for User type using string list
    :param string_list: string list of JSON object members
    :return: new User object
    """
    id = parse_int(string_list[0])
    name = string_list[1]
    email = string_list[2]
    password = string_list[3]
    user_type = parse_user_type(string_list[4])
    return User(id, name, email, password, user_type)