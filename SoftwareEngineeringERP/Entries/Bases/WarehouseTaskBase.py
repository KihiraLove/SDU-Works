from datetime import datetime


class WarehouseTaskBase:
    """
    Base class, do not use directly
    """
    def __init__(self, id: int, date: datetime, status: str) -> None:
        self.id = id
        self.date = date
        self.status = status
        return