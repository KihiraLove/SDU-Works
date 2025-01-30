
import os
from Entries.BusinessPartner import BusinessPartner
from Entries.Label import Label
from Entries.Material import Material
from Entries.SalesItem import SalesItem
from Entries.SalesOrder import SalesOrder
from Entries.SalesPerson import SalesPerson
from Entries.Shelf import Shelf
from Entries.ShelfItem import ShelfItem
from Entries.User import User
from Entries.WarehouseTask import WarehouseTask
from Enums.UserType import UserType
from Managers import FileManager
from Managers.FileManager import get_json_path, write_to_file
from Utils.Config import DATA_DIR
from Utils.Time import generate_time
from Utils.Types import TYPES

# This file is not part of the software, it only generates the JSON files to be used by the software

def resource_exists(resource_path: str) -> bool:
    """
    Check if resource exists
    :param resource_path: path to resource
    :return: whether resource exists
    """
    return os.path.exists(resource_path)


def generate_data_if_not_exist() -> None:
    """
    Generate Data dir and json files
    :return: None
    """
    if not resource_exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    for name in TYPES:
        filepath = get_json_path(name)
        if resource_exists(filepath):
            continue
        write_to_file(filepath, generate_content(name))
    return


def generate_content(name: str) -> str:
    """
    Generate JSON content for given name
    :param name: name to generate JSON content for
    :return: JSON content
    """
    content = []
    if name == "BusinessPartner":
        content = generate_business_partner_data()
    elif name == "Label":
        content = generate_label_data()
    elif name == "Material":
        content = generate_material_data()
    elif name == "SalesItem":
        content = generate_sales_item_data()
    elif name == "SalesOrder":
        content = generate_sales_order_data()
    elif name == "SalesPerson":
        content = generate_sales_person_data()
    elif name == "Shelf":
        content = generate_shelf_data()
    elif name == "ShelfItem":
        content = generate_shelf_item_data()
    elif name == "User":
        content = generate_user_data()
    elif name == "WarehouseTask":
        content = generate_warehouse_task_data()

    return FileManager.serialize(content)



def generate_business_partner_data() -> list[object]:
    """
    Generate dummy business partner data
    :return: list of dummy business partner objects
    """
    return [BusinessPartner(0, "The LEGO Group", "123 Maple Street Rivertown", 2),
            BusinessPartner(1, "Novo Nordisk", "45 Crescent Avenue Brightville", 3),
            BusinessPartner(2, "Lidl", "62 Pinecrest Road Sunnydale", 4),]


def generate_label_data() -> list[object]:
    """
    Generate dummy label data
    :return: list of dummy label objects
    """
    return [Label(0, 4),
            Label(1, 0),
            Label(2, 3),
            Label(3, 2),
            Label(4, 5),
            Label(5, 1)]


def generate_material_data() -> list[object]:
    """
    Generate dummy material data
    :return: list of dummy material objects
    """
    return [Material(0, 1235, "Titanium screw 10mm", 2000, 69001),
            Material(1, 2376, "Titanium screw 5mm", 2000, 56001),
            Material(2, 2356, "Irish butter", 1, 2),
            Material(3, 23778, "Irish butter salted", 6, 23),
            Material(4, 62354, "Steel pickaxe", 4124, 6112),
            Material(5, 2326, "Fresh crab shell", 100, 233),
            Material(6, 671, "Fine silk", 5, 6)]


def generate_sales_item_data() -> list[object]:
    """
    Generate dummy sales item data
    :return: list of dummy sales item objects
    """
    return [SalesItem(0, 1000, 0, 0),
            SalesItem(1, 1000, 0, 1),
            SalesItem(2, 1000, 1, 0),
            SalesItem(3, 1000, 1, 1),
            SalesItem(4, 5, 4, 2),
            SalesItem(5, 1000, 1, 3),
            SalesItem(6, 10, 4, 3),
            SalesItem(7, 1000, 0, 3),
            SalesItem(8, 1000, 1, 4),
            SalesItem(9, 1000, 0, 4),
            SalesItem(10, 1, 2, 5),
            SalesItem(11, 2, 6, 5),]


def generate_warehouse_task_data() -> list[object]:
    """
    Generate dummy warehouse task data
    :return: list of dummy warehouse task objects
    """
    return [WarehouseTask(0, generate_time(), "Waiting to arrive", 0, 3),
            WarehouseTask(1, generate_time(), "Waiting to arrive", 1, 3),
            WarehouseTask(2, generate_time(), "Completed", 2, 4),
            WarehouseTask(3, generate_time(), "In progress", 3, 4),
            WarehouseTask(4, generate_time(), "Completed", 4, 3),
            WarehouseTask(5, generate_time(), "Completed", 5, 3)]


def generate_user_data() -> list[object]:
    """
    Generate dummy user data
    :return: list of dummy user objects
    """
    return [User(0, "Johnathan Carter", "admin", "1234", UserType.MANAGER),
             User(1, "Amelia Hawkins", "ah@email.dk", "1234", UserType.SALES_PERSON),
             User(2, "Marcus Bennett", "mb@email.dk", "1234", UserType.SALES_PERSON),
             User(3, "Sophia Delgado", "sd@email.dk", "1234", UserType.WAREHOUSE_MANAGER),
             User(4, "Ethan Cross", "ec@email.dk", "1234", UserType.WAREHOUSE_WORKER)]


def generate_shelf_item_data() -> list[object]:
    """
    Generate dummy shelf item data
    :return: list of dummy shelf item objects
    """
    return [ShelfItem(0, 6, 0, 6),
            ShelfItem(1, 233, 1, 5),
            ShelfItem(2, 6000, 2, 4),
            ShelfItem(3, 112, 3, 4),
            ShelfItem(4, 23, 4, 3),
            ShelfItem(5, 2, 5, 2),
            ShelfItem(6, 20000, 0, 0),
            ShelfItem(7, 20000, 1, 0),
            ShelfItem(8, 29001, 5, 0),
            ShelfItem(9, 25000, 4, 1),
            ShelfItem(10, 25000, 5, 1),
            ShelfItem(11, 6001, 3, 1),]


def generate_shelf_data() -> list[object]:
    """
    Generate dummy shelf data
    :return: list of dummy shelf objects
    """
    return [Shelf(0, "A1"),
            Shelf(1, "A2"),
            Shelf(2, "A3"),
            Shelf(3, "B1"),
            Shelf(4, "B2"),
            Shelf(5, "B3"), ]


def generate_sales_person_data() -> list[object]:
    """
    Generate dummy sales person data
    :return: list of dummy sales person objects
    """
    return [SalesPerson(0, "Amelia Hawkins", "000000000", "ah@email.dk", True, 1),
                     SalesPerson(1, "Marcus Bennett", "111111111", "mb@email.dk", True, 2),
                     SalesPerson(2, "Olivia Grant", "31213456134", "og@ext-email.dk", False, None),
                     SalesPerson(3, "Daniel Pierce", "31213456134", "dp@ext-email.dk", False, None),
                     SalesPerson(4, "Emily Sanders", "31213456134", "es@ext-email.dk", False, None)]


def generate_sales_order_data() -> list[object]:
    """
    Generate dummy sales order data
    :return: list of dummy sales order objects
    """
    return [SalesOrder(0, generate_time(), "In transit", True, 0, 0),
            SalesOrder(1, generate_time(), "In warehouse", True, 1, 0),
            SalesOrder(2, generate_time(), "Completed", True, 2, 0),
            SalesOrder(3, generate_time(), "In warehouse", False, 0, 0),
            SalesOrder(4, generate_time(), "In transit", False, 1, 0),
            SalesOrder(5, generate_time(), "Competed", False, 2, 0)]