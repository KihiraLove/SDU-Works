from Enums.UserType import UserType


class UserBase:
    """
    Base class, do not use directly
    """
    def __init__(self, id: int, name: str, email: str, password: str, user_type: UserType) -> None:
        self.id = id
        self.name = name
        self.email = email
        self.password = password
        self.user_type = user_type
        return