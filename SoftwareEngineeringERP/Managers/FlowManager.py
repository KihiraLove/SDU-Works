from Enums.StatusKey import StatusKey
from Managers.DataManager import DataManager
from Managers.EntryManagers.LabelManager import LabelManager
from Managers.EntryManagers.MaterialManager import MaterialManager
from Managers.EntryManagers.SalesItemManager import SalesItemManager
from Managers.EntryManagers.SalesOrderManager import SalesOrderManager
from Managers.SessionManager import SessionManager
from Managers.UIManager import UIManager
import PySimpleGUI as sg


class FlowManager:
    """
    Manage the flow of the program
    """
    def __init__(self, data_manager: DataManager, ui_manager: UIManager, session_manager: SessionManager):
        """
        Constructor for FlowManager
        :param data_manager: data manager
        :param ui_manager: ui manager
        :param session_manager: session manager
        """
        self.data_manager = data_manager
        self.ui_manager = ui_manager
        self.session_manager = session_manager
        return

    def start_flow(self):
        """
        Enter function into workflow
        :return: None
        """
        while True:
            if not self.session_manager.logged_in():
                status = self.login_flow()
            else:
                status = self.main_flow()
            if status == StatusKey.EXIT:
                self.ui_manager.close_window()
                break
        return

    def login_flow(self) -> StatusKey:
        """
        UI flow to log in user
        :return: StatusKey whether the log in was successful or the user closed the window
        """
        self.ui_manager.create_login_window()
        while True:
            event, values = self.ui_manager.read_window()
            if event in (sg.WINDOW_CLOSED, "Cancel"):
                return StatusKey.EXIT
            if event == "Login":
                status = self.session_manager.login(email=values["-email-"], password=values["-password-"])
                if status == StatusKey.CORRECT:
                    return StatusKey.CORRECT
                elif status == StatusKey.EMAIL_CORRECT:
                    self.ui_manager.show_message("Error", "Invalid password.")
                else:
                    self.ui_manager.show_message("Error", "Invalid email.")
            if event == "Fast Login (Admin)":
                # added for quick login
                self.session_manager.login("admin", "1234")
                return StatusKey.CORRECT
            if event == "Fast Login (User)":
                # added for quick login
                self.session_manager.login("user", "1234")
                return StatusKey.CORRECT

    def main_flow(self) -> StatusKey:
        """
        Used to control the main menu and sales order flows
        :return: Status of last action
        """
        status = StatusKey.MAIN
        while True:
            self.ui_manager.update_window(status)
            status = None
            event, values = self.ui_manager.read_window()

            if event in (sg.WINDOW_CLOSED, "-EXIT-"):
                # User exited
                return StatusKey.EXIT
            elif event == "-LOGOUT-":
                # user logged out
                self.session_manager.logout()
                return StatusKey.LOGOUT
            elif event == "-SALES-":
                # User entered sales order menu
                status = StatusKey.SALES_ORDER
                continue
            elif event == "-MAIN-":
                # user entered main menu
                status = StatusKey.MAIN
                continue
            elif event == "-ADD_ITEM-":
                # user added new item to sales order
                status = StatusKey.ADD_ROW
                continue
            elif event == "-SAVE-":
                # user saved sales order
                status = self.submit_sales_order(values)
                continue


    def submit_sales_order(self, values: dict) -> StatusKey:
        """
        Submit a new sales order
        :param values: value read from UI
        :return: status of action
        """
        sales_order_manager = SalesOrderManager()
        sales_item_manager = SalesItemManager()
        material_manager = MaterialManager()
        label_manager = LabelManager()
        id_of_current_user = self.session_manager.user.get_id()
        try:
            is_inbound = False
            if values["-IS_INBOUND-"] == "Purchase Order":
                is_inbound = True

            business_partner_id = int(values["-BUSINESS_PARTNER-"].split("(")[1].removesuffix(")"))
            date = values["-DATE-"]
            sales_items = []
            for itr in range(self.ui_manager.get_dynamic_list_size()):
                # (Material ID, EXT ID, Material Amount)
                material_id = int(values[f"-MATERIAL-{itr}-"].split("(")[1].removesuffix(")"))
                material_amount = int(values[f"-MATERIAL_AMOUNT-{itr}-"])
                item = (material_id, material_amount)
                sales_items.append(item)
            s_message = ""
            s_title = "Purchase Order Successful" if is_inbound else "Sales Order Successful"
            sales_order_id = sales_order_manager.create_sales_order(date, "New", is_inbound, business_partner_id, id_of_current_user )
            # generate label for order
            label_manager.create_label(sales_order_id)
            delay_flag = False
            for item in sales_items:
                sales_item_manager.create_sales_item(item[0], item[1], sales_order_id)
                if not is_inbound:
                    status = material_manager.subtract_material(item[0], item[1])
                else:
                    # We don't know the correct status, since this is handled by external partner
                    status = StatusKey.ORDERED
                material_name = material_manager.get_by_id(item[0]).name
                s_message += f"material: {material_name} was ordered with status: {status}\n"
                if status == StatusKey.DELAYED_ORDER:
                    delay_flag = True

            if delay_flag:
                s_message += "Order is delayed due to low stock, automatic order to fill up stock was opened"

            self.ui_manager.show_message(s_title, s_message)
        except Exception as e:
            self.ui_manager.show_message("Field Error", f"Please check values: {e}")
            return StatusKey.FIELD_ERROR
        return StatusKey.MAIN
