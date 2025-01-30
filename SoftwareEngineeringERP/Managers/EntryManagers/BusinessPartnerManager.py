from Entries.BusinessPartner import BusinessPartner
from Utils.Singleton import Singleton


class BusinessPartnerManager(metaclass=Singleton):
    """
    Singleton class that holds data for all BusinessPartners
    """
    def __init__(self, data: list[BusinessPartner|object]=None):
        """
        Constructor for BusinessPartnerManager, to access existing singleton, create this without parameters
        :param data: None
        """
        self.data = data
        return

    def names_and_ids(self) -> list[str]:
        """
        Returns a list of names and ids of all BusinessPartners
        :return: list of names and ids
        """
        names_ids = []
        for business_partner in self.data:
            names_ids.append(f"{business_partner.company}({business_partner.id})")
        return names_ids



def create(data: list[BusinessPartner|object]) -> BusinessPartnerManager:
    """
    Constructor for BusinessPartnerManager, do not use this to access existing singleton, use __init__ method instead
    :param data: list of BusinessPartner objects
    :return: returns created singleton instance for BusinessPartnerManager
    """
    return BusinessPartnerManager(data)