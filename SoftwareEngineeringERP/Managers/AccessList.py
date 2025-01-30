from Enums.AccessType import AccessType
from Enums.UserType import UserType


class AccessList:
    """
    Dataclass to hold access lists and compare user access against.
    """
    def __init__(self):
        self.manager_access: set[AccessType] = {
                                            AccessType.CREATE_BUSINESS_PARTNER,
                                            AccessType.CREATE_MATERIAL,
                                            AccessType.CREATE_USER,
                                            AccessType.CREATE_SALES_ORDER,
                                            AccessType.CREATE_STORAGE_BIN,
                                            AccessType.RECEIVE_GOODS}

        self.sales_person_access: set[AccessType] = {
                                            AccessType.CREATE_BUSINESS_PARTNER,
                                            AccessType.CREATE_SALES_ORDER}

        self.warehouse_manager_access: set[AccessType] = {
                                            AccessType.CREATE_STORAGE_BIN,
                                            AccessType.RECEIVE_GOODS,
                                            AccessType.WAREHOUSE_TASK}

        self.warehouse_worker_access: set[AccessType] = {
                                            AccessType.WAREHOUSE_TASK}
        return

    def _get_access(self, user_type: UserType) -> set[AccessType]:
        """
        Get a list of access for user type
        :param user_type: user type to get access for
        :return: a set off accesses for user type
        """
        if user_type is UserType.MANAGER:
            return self.manager_access
        elif user_type is UserType.SALES_PERSON:
            return self.sales_person_access
        elif user_type is UserType.WAREHOUSE_MANAGER:
            return self.warehouse_manager_access
        elif user_type is UserType.WAREHOUSE_WORKER:
            return self.warehouse_worker_access
        raise ValueError('Invalid user type')

    def has_access(self, user_type: UserType, access: AccessType) -> bool:
        """
        Check if user has access
        :param access: the access type to check
        :param user_type: the user type of the user
        :return: whether the user has access
        """
        return access in self._get_access(user_type)