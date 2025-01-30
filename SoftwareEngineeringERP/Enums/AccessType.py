from enum import Enum


class AccessType(Enum):
    """
    Enum for access types
    """
    CREATE_BUSINESS_PARTNER = 1
    CREATE_MATERIAL = 2
    CREATE_USER = 3
    CREATE_SALES_ORDER = 4
    CREATE_STORAGE_BIN = 5
    RECEIVE_GOODS = 6
    WAREHOUSE_TASK = 7

    def __str__(self):
        """
        Convert AccessType enum to string
        :return: string representation of AccessType
        """
        return self.name