from enum import Enum


class StatusKey(Enum):
    """
    Enum for action statuses
    """
    EXIT = 1
    EMAIL_CORRECT = 2
    INCORRECT = 3
    CORRECT = 4
    LOGOUT = 5
    SALES_ORDER = 6
    MAIN = 7
    ADD_ROW = 8
    FIELD_ERROR = 9
    DELAYED_ORDER = 10
    AUTO_ORDER = 11
    ORDERED = 12

    def __str__(self):
        """
        Convert StatusKey enum to string
        :return: string representation of StatusKey
        """
        return self.name