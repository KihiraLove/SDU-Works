class SalesPersonBase:
    """
    Base class, do not use directly
    """
    def __init__(self, id: int, name: str, phone_number: str, email: str, is_internal: bool) -> None:
        self.id = id
        self.name = name
        self.phone_number = phone_number
        self.email = email
        self.is_internal = is_internal
        return