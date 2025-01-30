from Entries.Bases.BusinessPartnerBase import BusinessPartnerBase
from Utils.Parsing import parse_int, parse_int_or_none


class BusinessPartner(BusinessPartnerBase):
    """
    Data class for BusinessPartner type
    """
    def __init__(self, id: int, company: str, address: str, sales_person_id: int|None) -> None:
        """
        Constructor of BusinessPartner type
        :param id: id of business partner
        :param company: company of business partner
        :param address: address of business partner
        :param sales_person_id: id of linked sales person, may be None
        """
        super().__init__(id, company, address)
        self.sales_person_id = sales_person_id
        return

    def __repr__(self):
        """
        JSON representation of business partner
        :return: JSON string of business partner
        """
        return (f"{{"
                f"\"id\": \"{self.id}\","
                f"\"company\": \"{self.company}\","
                f"\"address\": \"{self.address}\","
                f"\"sales_person_id\": \"{self.sales_person_id}\""
                f"}}")

    def get_id(self) -> int:
        """
        Getter for id of business partner
        :return: id of business partner
        """
        return self.id

    def get_company(self) -> str:
        """
        Getter for company of business partner
        :return: company of business partner
        """
        return self.company

    def get_address(self) -> str:
        """
        Getter for address of business partner
        :return: address of business partner
        """
        return self.address

    def get_sales_person_id(self) -> int:
        """
        Getter for linked sales person id
        :return: id of linked sales person
        """
        return self.sales_person_id


def from_string_list(string_list: list[str]) -> BusinessPartner:
    """
    Constructor of BusinessPartner type using string list
    :param string_list: string list of JSON object members
    :return: new BusinessPartner object
    """
    id = parse_int(string_list[0])
    company = string_list[1]
    address = string_list[2]
    sales_person_id = parse_int_or_none(string_list[3])
    return BusinessPartner(id, company, address, sales_person_id)