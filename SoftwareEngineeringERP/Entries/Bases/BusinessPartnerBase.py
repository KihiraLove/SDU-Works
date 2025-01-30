class BusinessPartnerBase:
    """
    Base class, do not use directly
    """
    def __init__(self, id: int, company: str, address: str) -> None:
        self.id = id
        self.company = company
        self.address = address