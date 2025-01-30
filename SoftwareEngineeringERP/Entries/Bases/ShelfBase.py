class ShelfBase:
    """
    Base class, do not use directly
    """
    def __init__(self, id: int, location: str) -> None:
        self.id = id
        self.location = location
        return