class MaterialBase:
    """
    Base class, do not use directly
    """
    def __init__(self, id: int, ext_id: int, name: str, min_stock: int, stock: int) -> None:
        self.id = id
        self.ext_id = ext_id
        self.name = name
        self.min_stock = min_stock
        self.stock = stock
        return