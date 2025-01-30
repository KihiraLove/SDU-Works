import unittest

from Entries.BusinessPartner import BusinessPartner


class TestBusinessPartnerEntry(unittest.TestCase):
    def setUp(self):
        self.business_partner = BusinessPartner(0, "The LEGO Group", "LEGO Campus, Billund", 0)

    def test_init(self):
        self.assertIsInstance(self.business_partner,BusinessPartner)
        self.assertEqual(self.business_partner.company, "The LEGO Group")
        self.assertEqual(self.business_partner.address, "LEGO Campus, Billund")
        self.assertEqual(self.business_partner.sales_person_id, 0)


