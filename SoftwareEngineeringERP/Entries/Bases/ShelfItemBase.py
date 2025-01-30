class ShelfItemBase:
    """
    Base class, do not use directly
    """
    def __init__(self, id: int, amount: int) -> None:
        self.id = id
        self.amount = amount
        return