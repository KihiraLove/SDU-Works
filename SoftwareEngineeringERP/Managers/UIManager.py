from Enums.AccessType import AccessType
from Enums.StatusKey import StatusKey
from Managers.AccessList import AccessList
from Managers.EntryManagers.BusinessPartnerManager import BusinessPartnerManager
from Managers.EntryManagers.MaterialManager import MaterialManager
from Managers.EntryManagers.SalesOrderManager import SalesOrderManager
from Managers.SessionManager import SessionManager
from Utils.Time import generate_time
from Utils.Singleton import Singleton
import PySimpleGUI as sg


class UIManager(metaclass=Singleton):
    """
    manage UI windows and layouts
    """
    def __init__(self):
        """
        Constructor for UIManager
        """
        self.window = None
        self.dynamic_row_itr = 0
        return

    def close_window(self) -> None:
        """
        Closes the UI window
        :return: None
        """
        self.window.close()

    def read_window(self) -> tuple:
        """
        Read actions and inputs from UI window
        :return: events and values
        """
        return self.window.read()

    @staticmethod
    def show_message(title: str, message: str) -> None:
        """
        Shows a message as new popup window
        :param title: title of popup window
        :param message: message to show
        :return: None
        """
        sg.popup(message, title=title)

    def create_login_window(self) -> None:
        """
        Creates UI for logging in using PySimpleGUI
        :return: created window
        """
        if self.window is not None:
            self.window.close()

        layout = [
            [sg.Text("Email:", size=(10, 1)), sg.Input(key="-email-", size=(25, 1))],
            [sg.Text("Password:", size=(10, 1)), sg.Input(key="-password-", password_char="*", size=(25, 1))],
            [sg.Button("Login", bind_return_key=True), sg.Button("Cancel")],
            [sg.Button("Fast Login (Admin)", bind_return_key=True), sg.Button("Fast Login (User)", bind_return_key=True)]
        ]

        self.window = sg.Window("Login", layout, finalize=True)

    def _get_next_dynamic_row(self) -> list:
        """
        Create the next dynamic row for adding items to sales order
        :return: list of UI elements
        """
        material_manager = MaterialManager()
        materials = material_manager.names_ids_amounts()
        rows = [sg.Combo(materials, size=(20, 1), key=f"-MATERIAL-{self.dynamic_row_itr}-"),
                sg.Input(size=(20, 1), key=f"-MATERIAL_AMOUNT-{self.dynamic_row_itr}-")]
        self.dynamic_row_itr +=1
        return rows

    def update_window(self, status: StatusKey|None):
        """
        Update the UI window based of last actions status
        :param status: last actions status
        :return:
        """
        if status is StatusKey.FIELD_ERROR:
            # Skip update, popup is shown to user instead
            return
        if status is StatusKey.ADD_ROW:
            # add a new dynamic row
            new_row = self._get_next_dynamic_row()
            self.window.extend_layout(self.window["-DYNAMIC_ROWS-"], [new_row])
            return
        self.close_window()
        layout = []
        if status is None:
            # something went wrong, reopen window in next cycle
            return
        if status is StatusKey.MAIN:
            # user selected main menu, create layout for main menu
            layout = self._create_main_layout()
        if status is StatusKey.SALES_ORDER:
            # user selected sales order menu, create layout for sales order menu
            layout = self._create_sales_order_layout()

        # create and show UI window
        self.window = sg.Window(
            "Main Window",
            layout,
            element_justification="left",
            margins=(0, 0),
            finalize=True
        )
        return

    @staticmethod
    def _create_main_layout() -> list[object]:
        """
        Create the main menu Layout
        :return: list of UI elements
        """
        session_manager = SessionManager()
        access_list = AccessList()
        logged_in_users_type = session_manager.current_users_type()
        layout = []
        if access_list.has_access(logged_in_users_type, AccessType.CREATE_SALES_ORDER):
            layout.append([sg.Button("Sales Orders", size=(20, 2), key="-SALES-", pad=(20, 20))])
        layout.append([
            [sg.VPush()],
            [sg.VPush(),
             sg.Push(),
             sg.Button("Logout", size=(20, 2), key="-LOGOUT-", pad=(20, 20)),
             sg.Button("Exit", size=(20, 2), key="-EXIT-", pad=(20, 20))]
        ])
        return layout

    def _create_sales_order_layout(self) -> list[object]:
        """
        Create the sales order layout
        :return: list of UI elements
        """
        self.dynamic_row_itr = 0

        so_manager = SalesOrderManager()
        bp_manager = BusinessPartnerManager()

        business_partners = bp_manager.names_and_ids()
        id = len(so_manager.data)
        date = generate_time()
        dynamic_rows = [self._get_next_dynamic_row()]

        # Static layout
        layout = [
            [sg.Text("Create Sales Order/Purchase Order", font=("Helvetica", 16), justification="center", expand_x=True)],
            [sg.Text("ID:", size=(15, 1)), sg.Input(id, readonly=True, size=(20, 1), key="-ID-")],
            [sg.Text("Type:", size=(15, 1)), sg.Combo(["Purchase Order", "Sales Order"], size=(20, 1), key="-IS_INBOUND-")],
            [sg.Text("Business Partners:", size=(15, 1)), sg.Combo(business_partners, size=(20, 1), key="-BUSINESS_PARTNER-")],
            [sg.Text("Date:", size=(15, 1)), sg.Input(date, readonly=True, size=(20, 1), key="-DATE-")],
            [sg.Text("Items:", size=(15, 1))],
            [sg.Column(dynamic_rows, key="-DYNAMIC_ROWS-", vertical_scroll_only=True, size=(500, 200), scrollable=True, expand_x=True)],
            [sg.Button("Add Item", key="-ADD_ITEM-")],
            [sg.HorizontalSeparator()],
            [sg.Button("Save", key="-SAVE-", size=(10, 1)), sg.Button("Discard", key="-MAIN-", size=(10, 1)),
             sg.Button("Exit", key="-EXIT-", size=(10, 1))]
        ]

        return layout

    def get_dynamic_list_size(self) -> int:
        """
        Get the current size of dynamic sales item list
        :return: Current size of dynamic sales item list
        """
        return self.dynamic_row_itr


