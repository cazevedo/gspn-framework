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

'''
__token_states is a list with the states of each token ['Free', 'Occupied', 'Done'] means that token 1 is Free, token 2
is Occupied and token 3 is Done.
__token_positions is a list with the places where each token is ['p1', 'p2', 'p2'] means that token 1 is on p1, token 2
is on p2 and token 3 is on p2.
'''


class GSPNexecution(object):

    def __init__(self, gspn, place_to_function_mapping, policy, project_path):
        '''
        :param gspn: a previously created gspn
        :param place_to_function_mapping: dictionary where key is the place and the value is the function
        :param policy: Policy object
        :param project_path: string with project path

        token_states is a list with len==number of tokens with Strings that represent the state of each token
        modules is a list with references to the imported functions that are used in the place to function mapping
        '''
        self.__gspn = gspn
        self.__token_states = []
        self.__token_positions = []

        self.__place_to_function_mapping = place_to_function_mapping

        self.__policy = policy
        self.__project_path = project_path

        self.__number_of_tokens = 0
        self.__futures = []

    def get_token_states(self):
        return self.__token_states

    def get_path(self):
        return self.__project_path

    def get_policy(self):
        return self.__policy

    def convert_to_tuple(self, marking, order):
        '''
        :param marking: dictionary with key= places; value= number of tokens in place
        :param order: tuple of strings with the order of the marking on the policy
        :return: tuple with the marking
        '''
        marking_list = []
        for element in order:
            for key in marking:
                if element == key:
                    marking_list.append(marking[key])
        return tuple(marking_list)

    def get_transitions(self, marking, policy_dictionary):
        '''
        :param marking: tuple with the current marking (should already be ordered)
        :param policy_dictionary: dictionary where key=Transition name; value=probability of transition
        :return: transition dictionary if marking is in policy_dictionary; False otherwise
        '''
        for mark in policy_dictionary:
            if marking == mark:
                return policy_dictionary[mark]
        return False

    def translate_arcs_to_marking(self, arcs, marking):
        translation = {}
        for i in arcs[0]:
            place = self.__gspn.index_to_places[i]
            translation[place] = 1
        return translation

    def fire_execution(self, transition, token_id):
        '''
        Fires the selected transition.
        :param transition: string with transition that should be fired
        :param token_id: int with the number of the token that is being fired
        '''
        arcs = self.__gspn.get_connected_arcs(transition, 'transition')
        index = self.__gspn.transitions_to_index[transition]
        marking = self.__gspn.get_current_marking()

        # 1 to 1
        if len(arcs[0]) == 1 and len(arcs[1][index]) == 1:
            new_place = self.__gspn.index_to_places[arcs[1][index][0]]
            self.__token_positions[token_id] = new_place
            self.__gspn.fire_transition(transition)
            marking_transition_file = open("marking_transition.txt", 'a')
            marking_transition_file.write(str(self.__gspn.get_current_marking()) + "=" + str(transition) + "\n")
            marking_transition_file.close()

        # 1 to many
        elif len(arcs[0]) == 1 and len(arcs[1][index]) > 1:
            i = 0
            for i in range(len(arcs[1][index])):
                if i == 0:
                    new_place = self.__gspn.index_to_places[arcs[1][index][i]]
                    self.__token_positions[token_id] = new_place
                else:
                    new_place = self.__gspn.index_to_places[arcs[1][index][i]]
                    self.__token_positions.append(new_place)
                    self.__token_states.append('Free')
                    self.__number_of_tokens = self.__number_of_tokens + 1
                    self.__futures.append(self.__number_of_tokens)
            self.__gspn.fire_transition(transition)
            marking_transition_file = open("marking_transition.txt", 'a')
            marking_transition_file.write(str(self.__gspn.get_current_marking()) + "=" + str(transition) + "\n")
            marking_transition_file.close()

        # many to 1
        elif len(arcs[0]) > 1 and len(arcs[1][index]) == 1:
            translation_marking = self.translate_arcs_to_marking(arcs, marking)
            check_flag = True

            # We go through the marking and check it
            for el in translation_marking:
                if marking[el] < translation_marking[el]:
                    check_flag = False

            # We go through the states and see if all of them are 'Waiting'
            number_of_waiting = 0
            for place in translation_marking:
                for pos_index in range(len(self.__token_positions)):
                    if self.__token_positions[pos_index] == place:
                        if self.__token_states[pos_index] == 'Waiting':
                            number_of_waiting = number_of_waiting + 1
                            break
            if number_of_waiting == len(translation_marking) - 1:
                check_flag = True
            else:
                check_flag = False

            if check_flag:
                new_place = self.__gspn.index_to_places[arcs[1][index][0]]
                old_place = self.__token_positions[token_id]
                self.__token_positions[token_id] = new_place
                self.__gspn.fire_transition(transition)
                marking_transition_file = open("marking_transition.txt", 'a')
                marking_transition_file.write(str(self.__gspn.get_current_marking()) + "=" + str(transition) + "\n")
                marking_transition_file.close()
                for place_index in arcs[0]:
                    place_with_token_to_delete = self.__gspn.index_to_places[place_index]
                    if place_with_token_to_delete != old_place:
                        for j in range(len(self.__token_positions)):
                            if place_with_token_to_delete == self.__token_positions[j]:
                                index_to_del = j
                                self.__token_positions[index_to_del] = "null"
                                self.__token_states[index_to_del] = "VOID"
                                break
            else:
                self.__token_states[token_id] = 'Waiting'

        # many to many
        elif len(arcs[0]) > 1 and len(arcs[1][index]) > 1:
            translation_marking = self.translate_arcs_to_marking(arcs, marking)
            check_flag = True
            # We go through the marking and check it
            for el in translation_marking:
                if marking[el] < translation_marking[el]:
                    check_flag = False

            # We go through the states and see if all of them are 'Waiting'
            number_of_waiting = 0
            for place in translation_marking:
                for pos_index in range(len(self.__token_positions)):
                    if self.__token_positions[pos_index] == place:
                        if self.__token_states[pos_index] == 'Waiting':
                            number_of_waiting = number_of_waiting + 1
                            break
            if number_of_waiting == len(translation_marking) - 1:
                check_flag = True
            else:
                check_flag = False

            if check_flag:
                # Create tokens on next places
                i = 0
                for i in range(len(arcs[1][index])):
                    if i == 0:
                        new_place = self.__gspn.index_to_places[arcs[1][index][i]]
                        self.__token_positions[token_id] = new_place
                    else:
                        new_place = self.__gspn.index_to_places[arcs[1][index][i]]
                        self.__token_positions.append(new_place)
                        self.__token_states.append('Free')
                        self.__number_of_tokens = self.__number_of_tokens + 1
                        self.__futures.append(self.__number_of_tokens)
                        self.__gspn.fire_transition(transition)
                        marking_transition_file = open("marking_transition.txt", 'a')
                        marking_transition_file.write(str(self.__gspn.get_current_marking()) + "=" + str(transition) + "\n")
                        marking_transition_file.close()

                # Delete tokens from previous places
                for place_index in arcs[0]:
                    place_with_token_to_delete = self.__gspn.index_to_places[place_index]
                    for j in range(len(self.__token_positions)):
                        if place_with_token_to_delete == self.__token_positions[j]:
                            index_to_del = j
                            self.__token_positions[index_to_del] = "null"
                            self.__token_states[index_to_del] = "VOID"
                            break
            else:
                self.__token_states[token_id] = 'Waiting'


    def execute_gspn(self):
        '''
       Setup of the execution:
        1- token_states list and number of (initial) tokens;
        2- token_positions list;
        3- project path.
        '''

        # Setup token_states list and number of (initial) tokens
        self.__number_of_tokens = self.__gspn.get_number_of_tokens()
        i = 0
        while i < self.__number_of_tokens:
            self.__token_states.append('Free')
            self.__futures.append(i)
            i = i + 1

        # Setup token_positions list
        marking = self.__gspn.get_current_marking()
        for place in marking:
            j = 0
            while j != marking[place]:
                self.__token_positions.append(place)
                j = j + 1

        # Setup project path
        path_name = self.get_path()
        self.__project_path = os.path.join(path_name)
        sys.path.append(self.__project_path)

        # Create file to write the fired transition and the resulting marking
        # This file will be used as an output of the execution and on the case
        # when the user is running the Visualization Module.
        marking_transition_file = open("marking_transition.txt", 'w')
        marking_transition_file.close()

        # Create file to write status of execution.
        # This file will be used with the Visualization Module in order to know
        # when the user intends to stop the execution or not.
        execution_status_file = open("execution_status_file.txt", 'w')
        execution_status_file.write("EXECUTING")
        execution_status_file.close()
        print("vim até aqui")

        '''
        Main execution cycle. At every instant, the threads check whether the tokens are done with their functions
        or not.
        max_workers = self.__number_of_tokens * 3 because of the case where we have new tokens being created.
        '''
        with ThreadPoolExecutor(max_workers=self.__number_of_tokens * 3) as executor:
            while True:
                exe_stat_file = open("execution_status_file.txt", 'r')
                content = exe_stat_file.read()
                if content == "EXECUTING":
                    number_tokens = len(self.__token_positions)
                    for thread_number in range(number_tokens):

                        if self.__token_states[thread_number] == 'Free':
                            place = self.__token_positions[thread_number]
                            splitted_path = self.__place_to_function_mapping[place].split(".")

                            # On the first case we have path = FILE.FUNCTION
                            if len(splitted_path) <= 2:
                                function_location = splitted_path[0]
                                function_name = splitted_path[1]
                                module_to_exec = __import__(function_location)
                                function_to_exec = getattr(module_to_exec, function_name)

                            # On the second case we have path = FOLDER. ... . FILE.FUNCTION
                            else:
                                new_path = splitted_path[0]
                                for element in splitted_path[1:]:
                                    if element != splitted_path[-1]:
                                        new_path = new_path + "." + element

                                # dirpath = os.getcwd()
                                function_location = new_path
                                function_name = splitted_path[-1]
                                module_to_exec = __import__(function_location, fromlist=[function_name])
                                function_to_exec = getattr(module_to_exec, function_name)

                            self.__token_states[thread_number] = 'Occupied'
                            self.__futures[thread_number] = executor.submit(function_to_exec, thread_number)

                        if self.__token_states[thread_number] == 'Occupied' and self.__futures[thread_number].done():
                            self.__token_states[thread_number] = 'Done'

                        if self.__token_states[thread_number] == 'Done':
                            result = self.__futures[thread_number].result()
                            print("BEFORE", self.__gspn.get_current_marking())
                            if result is None:
                                execution_policy = self.get_policy()
                                current_marking = self.__gspn.get_current_marking()
                                order = execution_policy.get_places_tuple()
                                marking_tuple = self.convert_to_tuple(current_marking, order)
                                pol_dict = execution_policy.get_policy_dictionary()
                                transition_dictionary = self.get_transitions(marking_tuple, pol_dict)

                                if transition_dictionary:
                                    print("Immediate Transition")
                                    transition_list = []
                                    probability_list = []
                                    for transition in transition_dictionary:
                                        transition_list.append(transition)
                                        probability_list.append(transition_dictionary[transition])
                                    transition_to_fire = np.random.choice(transition_list, 1, False, probability_list)[0]
                                    print("TRANSITION TO FIRE", transition_to_fire)
                                    self.fire_execution(transition_to_fire, thread_number)
                                else:
                                    print("The place has no outbound connections.")
                                    self.__token_states[thread_number] = 'Inactive'

                            else:
                                print("Exponential Transition")
                                self.fire_execution(result, thread_number)
                            print("AFTER", self.__gspn.get_current_marking())

                            if self.__token_states[thread_number] == 'Waiting':
                                print("i am waiting")

                            elif self.__token_states[thread_number] == 'Done':
                                self.__token_states[thread_number] = 'Free'
                            print("--------")

                else:
                    executor.shutdown()
                    exe_stat_file = open("execution_status_file.txt", 'w')
                    exe_stat_file.write("DONE")
                    exe_stat_file.close()



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
