from datetime import datetime


class SalesOrderBase:
    """
    Base class, do not use directly
    """
    def __init__(self, id: int, date: datetime, status: str, is_inbound: bool) -> None:
        self.id = id
        self.date = date
        self.status = status
        self.is_inbound = is_inbound
        return