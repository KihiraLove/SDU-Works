def read_from_file(filename: str):
    """
    Reads data from file
    :param filename: name of file to read from
    :return: contents of file
    """
    with open(filename, 'r') as file:
        content = file.read()
    return content


def write_to_file(filename: str, content: str):
    """
    Writes content to file
    :param filename: name of file to write to
    :param content: content to write
    :return: None
    """
    with open(filename, 'w') as file:
        file.write(content)
    return


def serialize(obj_list: list[object]) -> str:
    """
    Serialize a list of objects to a JSON string
    :param obj_list: list of objects
    :return: JSON string
    """
    return f"{{{obj_list.__repr__()}}}"


def deserialize(json_string: str, object_type: type) -> list[object]:
    """
    Deserialize a JSON string to a list of objects
    :param json_string: JSON string of array of JSON objects
    :param object_type: type of object to deserialize into
    :return: list of objects
    """
    # remove JSON object opening and closing brackets
    # remove JSON array object opening and closing brackets
    # split at JSON array member objects closing bracket + JSON array delimiter + next JSON array member object opening brackets
    string_list = json_string.removeprefix("{").removesuffix("}").removeprefix("[").removesuffix("]").split("}, {")
    object_list = []
    for string in string_list:
        # each object type we want to deserialize has its own constructor to create a new object from JSON object members
        # object_type.from_string is prone to malicious code execution, but an easy solution here
        object_list.append(object_type.from_string_list(clean_up_json_to_string_list(string)))
    return object_list


def clean_up_json_to_string_list(json_string: str) -> list[str]:
    """
    Cleans up JSON string of a single JSON object
    :param json_string: JSON string of a single JSON object
    :return: string list of JSON object members
    """
    # remove JSON object opening and closing brackets, remove JSON object members string identifier, split at delimiter
    string_list = json_string.strip().removeprefix("{").removesuffix("}").replace("\"", "").split(",")
    for index, string in enumerate(string_list):
        # Remove JSON object member key at key-value delimiter
        # Using max split because datetime has multiple : characters
        string_list[index] = string.split(":", 1)[1].strip()
    return string_list

def get_json_path(type_name: str) -> str:
    """
    Get JSON file path for object type
    :param type_name: name of object type
    :return: path to JSON file for type
    """
    return f"./Data/{type_name}.json"
