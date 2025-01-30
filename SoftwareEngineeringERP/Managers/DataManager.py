from Managers.FileManager import read_from_file, get_json_path, deserialize, serialize, write_to_file
from Utils.Types import TYPES


class DataManager(object):
    """
    Manage data between JSON files and in memory objects, create Entry manager singletons
    """
    def __init__(self):
        """
        Create Entry manager singletons and save a reference into a dict, with keys from Config.TYPES.Keys()
        """
        self.entry_managers = dict()
        self._create_entry_managers()
        return

    def _create_entry_managers(self):
        """
        Create Entry manager singletons
        :return: None
        """
        for name in TYPES.keys():
            # create path for JSON location for given name
            json_file_path = get_json_path(name)
            # read file contents
            file_content = read_from_file(json_file_path)
            # deserialize file contents into list of objects of type of given name
            deserialized_content = deserialize(file_content, TYPES[name].type)

            # using the given name as key, get the Type_ManagerType named tuple from TYPES
            # .manager_type is a field of named tuple, which returns the Manager type
            # create(deserialized_content) calls the create function of Manager type returned from the named tuple
            # we save this new EntryManager into entry_managers with the given name as key
            self.entry_managers[name] = TYPES[name].manager_type.create(deserialized_content)
        return

    def write_data(self):
        """
        Write all object data from memory into JSON files
        :return: None
        """
        # iterate over all entry managers by name
        for name in self.entry_managers.keys():
            # get json path for given name
            json_file_path = get_json_path(name)
            # serialize data from EntryManager into JSON
            # data is a list of Entry objects
            json_content = serialize(self.entry_managers[name].data)
            # write JSON content to file
            write_to_file(json_file_path, json_content)
        return
