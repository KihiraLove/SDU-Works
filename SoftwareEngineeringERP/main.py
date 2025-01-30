from Managers.DataManager import DataManager
from Managers.FlowManager import FlowManager
from Managers.SessionManager import SessionManager
from Managers.UIManager import UIManager
from Utils.GenerateData import generate_data_if_not_exist

# Made for Software Engineering
# SDU
# Autumn semester 2024
# Software Engineering
# Made by Team Gravia Moribunda
# Authors:
#           Domonkos Kertész (KihiraLove@github)
#           Sara Lattarulo
#           Tamás Koroknay
#           Thiago Kawahara

# Python reStructuredText documentation was used
# UI is dependent on package PySimpleGUI
# Functions that are required but don't have UI can be found in Functions.py
# Tests were started in test.py using the unittest package, but we ran out of time

# Login UI has two fast login buttons that automatically log in as admin or user
# Otherwise login info is: "admin" "1234"


# This isn't part of the software, it generates JSON files with dummy data to be used
generate_data_if_not_exist()


# initialize data
data_manager = DataManager()

# start managers
session_manager = SessionManager()
ui_manager = UIManager()
flow_manager = FlowManager(data_manager, ui_manager, session_manager)

# start workflow
flow_manager.start_flow()

# write data at shutdown
data_manager.write_data()