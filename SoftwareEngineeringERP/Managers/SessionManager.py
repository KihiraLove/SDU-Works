from Entries.User import User
from Enums.StatusKey import StatusKey
from Enums.UserType import UserType
from Managers.EntryManagers.UserManager import UserManager
from Utils.Singleton import Singleton


class SessionManager(metaclass=Singleton):
    def __init__(self, user: User=None):
        """
        Class to handle log in session
        :param user: current user, None by default, use login function to set user
        """
        self.user = user
        return

    def login(self, email: str, password: str) -> StatusKey:
        """
        Logs the user in, sets self.user
        :param email: email to log in with
        :param password: password of user
        :return: StatusKey whether login was successful or not
        """
        try:
            user = UserManager().search(email=email)
            if user.password_matches(password):
                self.user = user
                return StatusKey.CORRECT
            return StatusKey.EMAIL_CORRECT
        except ValueError as e:
            print(e)
            return StatusKey.INCORRECT

    def logout(self) -> None:
        """
        Logout the user by setting the user to None
        :return: None
        """
        self.user = None
        return

    def logged_in(self) -> bool:
        """
        Function to check if user is logged in
        :return: boolean whether user is logged in
        """
        if self.user is None:
            return False
        return True

    def current_users_type(self) -> UserType:
        """
        Function to get current users type
        :return: type of current user
        """
        return self.user.user_type