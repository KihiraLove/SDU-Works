from collections import namedtuple
from Entries import BusinessPartner, Label, Material, SalesItem, SalesOrder, SalesPerson, Shelf, ShelfItem, User, WarehouseTask
from Managers.EntryManagers import BusinessPartnerManager, LabelManager, MaterialManager, SalesItemManager, SalesOrderManager, SalesPersonManager, ShelfManager, ShelfItemManager, UserManager, WarehouseTaskManager

# This modul is used to create generic functions

Type_ManagerType = namedtuple('Type_ManagerType', ['type', 'manager_type'])

# use TYPES.keys() and TYPES.values()
TYPES = {"BusinessPartner": Type_ManagerType(BusinessPartner, BusinessPartnerManager),
         "Label": Type_ManagerType(Label, LabelManager),
         "Material": Type_ManagerType(Material, MaterialManager),
         "SalesItem": Type_ManagerType(SalesItem, SalesItemManager),
         "SalesOrder": Type_ManagerType(SalesOrder, SalesOrderManager),
         "SalesPerson": Type_ManagerType(SalesPerson, SalesPersonManager),
         "Shelf": Type_ManagerType(Shelf, ShelfManager),
         "ShelfItem": Type_ManagerType(ShelfItem, ShelfItemManager),
         "User": Type_ManagerType(User, UserManager),
         "WarehouseTask": Type_ManagerType(WarehouseTask, WarehouseTaskManager)}
