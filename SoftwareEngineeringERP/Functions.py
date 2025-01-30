from Enums.UserType import UserType
from Managers.EntryManagers.MaterialManager import MaterialManager
from Managers.EntryManagers.SalesOrderManager import SalesOrderManager
from Managers.EntryManagers.UserManager import UserManager
from main import flow_manager

# Since our program flow is tightly coupled with our UI, we cannot implement the asked functions without UI
# All of them exists in code, here is an outline on how to use them

def create_order():
    # Order creation is tightly coupled with UI, see this function
    flow_manager.submit_sales_order(None)

def list_orders_from_shop(store_id: int):
    sales_order_manager = SalesOrderManager()
    orders = sales_order_manager.get_by_bp_id(store_id)
    print(orders)

def view_order(id: int):
    sales_order_manager = SalesOrderManager()
    order = sales_order_manager.get_by_id(id)

def pack_order():
    # We assume this function is the same as: subtract_material
    material_id = -1
    material_amount = 0
    material_manager = MaterialManager()
    material_manager.subtract_material(material_id, material_amount)

def create_user():
    user_manager = UserManager()
    new_user = user_manager.create_new_user("New users name", "New users email", "New users password", UserType.MANAGER)
    print(new_user)

def list_users():
    user_manager = UserManager()
    users = user_manager.data
    print(users)

def list_users_orders(id: int):
    sales_order_manager = SalesOrderManager()
    orders = sales_order_manager.get_by_user_id(id)
    print(orders)