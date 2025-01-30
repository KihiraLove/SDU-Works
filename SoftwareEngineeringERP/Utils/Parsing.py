from datetime import datetime
from Enums.UserType import UserType
from Utils import Config


def parse_bool(bool_str: str) -> bool:
    """
    Parse a boolean string to boolean
    :param bool_str: string representation of boolean
    :return: boolean
    """
    if bool_str == 'True':
        return True
    elif bool_str == 'False':
        return False
    raise ValueError('Invalid boolean value')

def parse_int_or_none(int_str: str) -> int:
    """
    Parse an integer or none string to integer or None
    :param int_str: integer or none string to parse
    :return: integer or None
    """
    if int_str == 'None':
        return None
    else:
        return int(int_str)

def parse_int(int_str: str) -> int:
    """
    Parse an integer string to integer
    :param int_str: integer string to parse
    :return: integer
    """
    return int(int_str)

def parse_datetime(datetime_str: str) -> datetime:
    """
    Parse a datetime string to datetime
    :param datetime_str: datetime string to parse
    :return: datetime
    """
    return datetime.strptime(datetime_str, Config.TIME_FORMAT)


def parse_user_type(user_string: str) -> UserType:
    """
    Parse string representation of UserType to UserType
    :param user_string: string representation of UserType
    :return: UserType
    """
    if user_string == 'MANAGER':
        return UserType.MANAGER
    elif user_string == 'SALES_PERSON':
        return UserType.SALES_PERSON
    elif user_string == 'WAREHOUSE_MANAGER':
        return UserType.WAREHOUSE_MANAGER
    elif user_string == 'WAREHOUSE_WORKER':
        return UserType.WAREHOUSE_WORKER
    raise ValueError('Invalid user type')
