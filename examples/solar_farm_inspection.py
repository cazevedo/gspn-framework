#! /usr/bin/env python3
# Standard libs
from concurrent.futures.thread import ThreadPoolExecutor
import os
import sys
import numpy as np
import json
import ast
# Files from my package
import policy
import gspn as pn
import gspn_tools

def main():
    project_path = "C:/Users/calde/Desktop/ROBOT"
    sys.path.append(os.path.join(project_path))

    with open(
            'C:/Users/calde/Desktop/gspn_framework_package/common/src/gspn_framework_package/gspn_execution_input_2.json') as f:
        data = json.load(f)

    tool = gspn_tools.GSPNtools()
    to_open = 'C:/Users/calde/Desktop/gspn_framework_package/common/src/gspn_framework_package/' + data["gspn"]
    my_pn = tool.import_xml(to_open)[0]

    p_to_f_mapping = ast.literal_eval(data["place_to_function_mapping"])
    policy_dictionary = ast.literal_eval(data["policy_dictionary"])
    places_tuple = ast.literal_eval(data["places_tuple"])
    created_policy = policy.Policy(policy_dictionary, places_tuple)

    my_execution = GSPNexecution(my_pn, p_to_f_mapping, created_policy, project_path)
    my_execution.execute_gspn()


if __name__ == "__main__":
    main()