#! /usr/bin/env python3
# Standard libs
from concurrent.futures.thread import ThreadPoolExecutor
import os
import sys
import numpy as np
from datetime import datetime
import time
import random
import json
import ast
# ROS libs
import rospy
import actionlib
# Files from my package
from gspn_framework_package import policy
from gspn_framework_package import gspn as pn
from gspn_framework_package import gspn_tools
from gspn_framework_package import gspn_analysis

import gspn_framework_package.msg
from gspn_framework_package.msg import ExecGSPNAction
from gspn_framework_package.msg import GSPNFiringData
from gspn_framework_package.srv import CurrentPlace, CurrentPlaceResponse, FireSyncTransition, FireSyncTransitionResponse

GEN_CURRENT_PLACE = 0
GEN_CURRENT_STATUS = "DONE"
GEN_ROBOT_ID = 0

def service_fire_sync(argument):
    global GEN_CURRENT_PLACE
    global GEN_CURRENT_STATUS
    GEN_CURRENT_PLACE = argument.place_to_go
    GEN_CURRENT_STATUS = "READY"
    return FireSyncTransitionResponse("SUCCESS")

def service_current_place_status_function(argument):
    return CurrentPlaceResponse(GEN_CURRENT_PLACE, GEN_CURRENT_STATUS, GEN_ROBOT_ID)


def analyze_gspn_structure(gspn_to_analyze, resources):
    '''
    Function to determine whether the input gspn is valid or not.
    1- We remove the places of the gspn that are considered "resources"
    2- We remove every transition that has 0 input or output arcs
    3- We generate the reachibility graph for the new gspn
    4- We check if the number of tokens stays the same
    '''
    print("Initiating gspn analysis...")

    '''Step 1'''
    new_gspn_to_analyze = gspn_to_analyze
    for resource_place in resources:
        if resource_place in new_gspn_to_analyze.get_places():
            new_gspn_to_analyze.remove_place(resource_place)

    '''Step 2'''
    for transition in new_gspn_to_analyze.get_transitions():
        arcs = new_gspn_to_analyze.get_connected_arcs(transition, 'transition')
        index = new_gspn_to_analyze.transitions_to_index[transition]
        if len(arcs[0]) == 0 or len(arcs[1]) == 0:
            new_gspn_to_analyze.remove_transition(transition)

    new_gspn_to_analyze.set_new_initial_marking()

    '''Step 3'''
    gspn_analysis_object = gspn_analysis.CoverabilityTree(new_gspn_to_analyze)
    gspn_analysis_object.generate()
    reachibility_graph_nodes = gspn_analysis_object.nodes
    max_tokens = []
    for node in reachibility_graph_nodes:
        total_tokens = 0
        for element in reachibility_graph_nodes[node][0]:
            total_tokens = total_tokens + element[1]
        max_tokens.append(total_tokens)

    '''Step 4'''
    return min(max_tokens) == max(max_tokens)

    '''
    Old Version which is working at 100%

    transitions = gspn_to_analyze.get_transitions()
    for transition in transitions:
        arcs = gspn_to_analyze.get_connected_arcs(transition, 'transition')
        index = gspn_to_analyze.transitions_to_index[transition]

        if len(arcs[0]) > 1 and len(arcs[1][index]) == 1:
            # We reject  because a robot cannot disappear
            return False

        elif len(arcs[0]) == 1 and len(arcs[1][index]) > 1:
            # We are prunning away the places where no physical robot will exist
            non_resource_places_counter = 0
            for place_index in arcs[1][index]:
                place = gspn_to_analyze.index_to_places[place_index]
                if place not in resources:
                    non_resource_places_counter = non_resource_places_counter + 1
            if non_resource_places_counter > 1:
                return False

        elif len(arcs[0]) > 1 and len(arcs[1][index]) > 1:
            # We are prunning away the places where no physical robot will exist
            non_resource_places_counter = 0
            for place_index in arcs[1][index]:
                place = gspn_to_analyze.index_to_places[place_index]
                if place not in resources:
                    non_resource_places_counter = non_resource_places_counter + 1
            if non_resource_places_counter != len(arcs[0]):
                return False

        print("Analysis complete.")
        return True
        '''



class GSPNExecutionROS(object):

    def __init__(self, gspn, place_to_client_mapping, resources, policy, initial_place, robot_id, full_synchronization):
        '''
        :param gspn: a previously created gspn
        :param place_to_client_mapping: dictionary where key is the place and the value is the function
        :param resources: list with the name of the places that are resources
        :param policy: Policy object
        :param initial_place: string with the name of the robot's initial place
        :param robot_id: number of the robot
        :param full_synchronization: boolean flag that says whether the robots have full sync or not

        self.__current_place indicates where the robot is
        self.__action_client is the action client of the robot
        self.__publisher is the publisher to the GSPNFiringData topic
        self.__subscriber is the subscriber to the GSPNFiringData topic
        '''
        self.__gspn = gspn
        self.__place_to_client_mapping = place_to_client_mapping
        self.__resources = resources
        self.__policy = policy
        self.__current_place = initial_place
        self.__robot_id = robot_id
        self.__full_synchronization = full_synchronization

        self.__action_client = 0
        self.__publisher = 0
        self.__subscriber = 0

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

    def translate_arcs_to_marking(self, arcs):
        '''
        : param arcs: set of connected add_arcs
        : return: dictionary where key is the place and the value is always 1
        This function essentially is used to determine which places are connected to the next transition.
        The purpose of this is to later on compare it with the current marking.
        '''
        translation = {}
        for i in arcs[0]:
            place = self.__gspn.index_to_places[i]
            translation[place] = 1
        return translation

    '''
    Callback functions section:
    In this section, we included every callback function that is being used both by our Action Clients and
    our publishers/subscribers:
    - topic_listener_callback;
    - topic_talker_callback;
    - action_get_result_callback;
    - action_goal_response_callback;
    - action_feedback_callback;
    - action_send_goal
    '''

    def topic_listener_callback(self, msg):
        arcs = self.__gspn.get_connected_arcs(msg.transition, 'transition')
        index = self.__gspn.transitions_to_index[msg.transition]
        if len(arcs[0]) <= 1 or len(arcs[1][index]) <= 1:
            if msg.robot_id != self.__robot_id:
                rospy.loginfo('I heard Robot %s firing %s' % (msg.robot_id, msg.transition))
                self.__gspn.fire_transition(msg.transition)
                print("AFTER", self.__gspn.get_current_marking())
            else:
                rospy.loginfo('I heard myself firing %s' % msg.transition)


    def topic_talker_callback(self, fired_transition):
        # the topic message is composed by four elements:
        # - fired transition;
        # - current marking;
        # - robot_id;
        # - timestamp.
        msg = GSPNFiringData()
        current_time = rospy.get_rostime()
        msg.transition = str(fired_transition)
        msg.marking = str(self.__gspn.get_current_marking())
        msg.robot_id = self.__robot_id
        msg.timestamp = str(current_time)

        self.__publisher.publish(msg)
        rospy.loginfo('Robot %s firing %s'% (msg.robot_id, msg.transition))


    def service_send_request(self, flag):
        # The flag only specifies how much do we want to return
        number_connections = self.__publisher.get_num_connections()
        current_robot_id = 1
        answers = []
        while current_robot_id <= number_connections:
            if current_robot_id != self.__robot_id:
                service_name = '/robot_' + str(current_robot_id) + '/current_place_robot_' + str(current_robot_id)
                rospy.wait_for_service(service_name, timeout=10)

                try:
                    service_current_place_status_function = rospy.ServiceProxy(service_name, CurrentPlace)
                    answer = service_current_place_status_function()
                    if flag == "simple":
                        answers.append(answer.current_place)
                    else:
                        answers.append(answer.robot_id)
                        answers.append(answer.current_place)
                        answers.append(answer.current_status)

                except rospy.ServiceException as e:
                    print("Service call failed: %s"%e)

            current_robot_id = current_robot_id + 1
        return answers


    def action_get_result_callback(self, status, result):

        # status == 3 means SUCCESS
        if status == 3:
            print(': Goal succeeded! Result: {0}'.format(result.transition))
            global GEN_CURRENT_STATUS
            GEN_CURRENT_STATUS = "DONE"
            print("current place ", self.__current_place, "current robot ", self.__robot_id)

            bool_output_arcs = self.check_output_arcs(self.__current_place)

            if bool_output_arcs:
                print("The place has output arcs.")
                print("BEFORE", self.__gspn.get_current_marking())
                if result.transition == 'None':
                    print("Immediate transition")

                    if self.__full_synchronization == True:
                        imm_transition_to_fire = self.get_policy_transition()
                        if imm_transition_to_fire == False:
                            print("The policy does not include this case.")
                            return
                        else:
                            self.fire_execution(imm_transition_to_fire)
                            self.topic_talker_callback(imm_transition_to_fire)
                    else:
                        answers = self.service_send_request("simple")
                        new_marking = {}
                        new_marking[self.__current_place] = 1
                        for place_list in answers:
                            if place_list in new_marking:
                                new_marking[place_list] = new_marking[place_list] + 1
                            else:
                                new_marking[place_list] = 1

                        old_marking = self.__gspn.get_places()

                        for nplace in old_marking:
                            if nplace not in new_marking:
                                new_marking[nplace] = 0

                        self.__gspn.set_places(new_marking)
                        imm_transition_to_fire = self.get_policy_transition()

                        if imm_transition_to_fire == False:
                            print("The policy does not include this case.")
                            return
                        else:
                            self.fire_execution(imm_transition_to_fire)

                else:
                    print("exponential transition")
                    print(result.transition)
                    self.fire_execution(result.transition)
                    if self.__full_synchronization == True:
                        self.topic_talker_callback(result.transition)

                print("AFTER", self.__gspn.get_current_marking())

                action_type = self.__place_to_client_mapping[self.__current_place][0]
                server_name = self.__place_to_client_mapping[self.__current_place][1]
                self.__action_client = actionlib.SimpleActionClient(server_name, action_type)
                self.action_send_goal(self.__current_place, action_type, server_name)

            else:
                print("The place has no output arcs.")

        else:
            print(self.__action_client._action_name + ': Goal failed with status: {0}'.format(status))


    def action_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            print('Goal rejected :( '+ self.__action_client._action_name)
            return

        print('Goal accepted :) '+ self.__action_client._action_name)

        self.__action_client._get_result_future = goal_handle.get_result_async()
        self.__action_client._get_result_future.add_done_callback(self.action_get_result_callback)


    def action_feedback_callback(self, feedback):
        print('Received feedback: {0}'.format(feedback.time_passed))


    def action_send_goal(self, current_place, action_type, server_name):
        self.__action_client.wait_for_server()
        print('Waiting for action server '+ server_name)
        goal = gspn_framework_package.msg.ExecGSPNGoal(current_place)
        print('Sending goal request to '+ server_name)
        global GEN_CURRENT_STATUS
        GEN_CURRENT_STATUS = "DOING"
        self.__action_client.send_goal(goal, done_cb=self.action_get_result_callback, feedback_cb=self.action_feedback_callback)


    '''
    GSPN Execution core functions section:
    In this section, we included the four main functions that are responsible for the execution of our
    GSPNs:
    - fire execution;
    - get_immediate_transition_result;
    - check_output_arcs;
    - ros_gspn_execution.
    '''

    def fire_execution(self, transition):
        '''
        Fires the selected transition.
        :param transition: string with transition that should be fired
        '''
        arcs = self.__gspn.get_connected_arcs(transition, 'transition')
        index = self.__gspn.transitions_to_index[transition]
        marking = self.__gspn.get_current_marking()
        global GEN_CURRENT_PLACE

        # One to one
        if len(arcs[0]) == 1 and len(arcs[1][index]) == 1:
            new_place = self.__gspn.index_to_places[arcs[1][index][0]]
            self.__gspn.fire_transition(transition)
            self.__current_place = new_place
            GEN_CURRENT_PLACE = new_place

        # One to many
        elif len(arcs[0]) == 1 and len(arcs[1][index]) > 1:
            for place_index in arcs[1][index]:
                place = self.__gspn.index_to_places[place_index]
                if place not in self.__resources:
                    new_place = self.__gspn.index_to_places[arcs[1][index][0]]
                    self.__current_place = new_place
                    GEN_CURRENT_PLACE = new_place

            self.__gspn.fire_transition(transition)

        # Many to many
        elif len(arcs[0]) > 1 and len(arcs[1][index]) > 1:

            # Let's see if the marking corresponds to the necessary one.
            translation_marking = self.translate_arcs_to_marking(arcs)
            check_flag = True
            for el in translation_marking:
                if marking[el] < translation_marking[el]:
                    check_flag = False

            if check_flag:
                # We only consider the places that are not resources
                # This will be useful when we are firing the transition
                possible_places = []
                for place_index in arcs[1][index]:
                    place = self.__gspn.index_to_places[place_index]
                    if place not in self.__resources:
                        possible_places.append(place)

                # We check where each robot is and whether it finished its task
                answers = self.service_send_request("complete")
                done_counter = 0
                done_ids = []
                for element in arcs[0]:
                    place_to_check = self.__gspn.index_to_places[element]
                    iterator = 0
                    if place_to_check != self.__current_place:
                        while iterator != len(answers):
                            if answers[iterator+1] == place_to_check and answers[iterator+2] == "DONE":
                                done_counter = done_counter + 1
                                done_ids.append(answers[iterator])
                            iterator = iterator + 3

                # If we have every robot waiting to be fired we enter this case
                # The -1 is because the present robot is not considered for the count
                if done_counter == len(arcs[0]) - 1:
                    print("I'm the last one to be ready to fire.")
                    for id in done_ids:
                        service_name = '/robot_' + str(id) + '/fire_sync_transition_' + str(id)
                        rospy.wait_for_service(service_name, timeout=10)

                        try:
                            service_handle_fire_sync_function = rospy.ServiceProxy(service_name, FireSyncTransition)
                            place_to_go = possible_places[-1]
                            possible_places.pop()

                            suc_flag = service_handle_fire_sync_function(place_to_go)

                        except rospy.ServiceException as e:
                          print("Service call failed: %s"%e)

                    place_to_go = possible_places[-1]
                    possible_places.pop()
                    GEN_CURRENT_PLACE = place_to_go

                else:
                    while GEN_CURRENT_STATUS == "DONE":
                        time.sleep(1)
                        print("I am waiting for a change in the status.")
                        continue

            else:
                while GEN_CURRENT_STATUS == "DONE":
                    time.sleep(1)
                    print("I am waiting for a change in the status.")
                    continue

            self.__gspn.fire_transition(transition)
            self.__current_place = GEN_CURRENT_PLACE

    def get_policy_transition(self):
        execution_policy = self.get_policy()
        current_marking = self.__gspn.get_current_marking()
        order = execution_policy.get_places_tuple()
        marking_tuple = self.convert_to_tuple(current_marking, order)
        pol_dict = execution_policy.get_policy_dictionary()
        transition_dictionary = self.get_transitions(marking_tuple, pol_dict)
        if transition_dictionary == False:
            return False
        transition_list = []
        probability_list = []
        for transition in transition_dictionary:
            transition_list.append(transition)
            probability_list.append(transition_dictionary[transition])
        transition_to_fire = np.random.choice(transition_list, 1, False, probability_list)[0]
        print("TRANSITION TO FIRE", transition_to_fire)
        return transition_to_fire

    def check_output_arcs(self, place):
        arcs = self.__gspn.get_connected_arcs(place, 'place')
        arcs_out = arcs[1]
        if len(arcs_out) >= 1:
            return True
        else:
            return False

    def ros_gspn_execution(self):
        '''
        Setup of the execution:
        1- project path;
        2- number of initial tokens;
        3- token_positions list;
        4- action servers;
        5- initial action clients.
        '''
        # Setup publisher and subscriber
        self.__publisher = rospy.Publisher('/TRANSITIONS_FIRED', GSPNFiringData, queue_size=10)
        self.__subscriber = rospy.Subscriber('/TRANSITIONS_FIRED', GSPNFiringData, self.topic_listener_callback, queue_size=10)

        # Setup action client
        action_type = self.__place_to_client_mapping[self.__current_place][0]
        server_name = self.__place_to_client_mapping[self.__current_place][1]
        self.__action_client = actionlib.SimpleActionClient(server_name, action_type)

        # Setup current place service (used when full_synchronization == False)
        current_place_service_name = 'current_place_robot_' + str(self.__robot_id)
        current_place_service = rospy.Service(current_place_service_name, CurrentPlace, service_current_place_status_function)

        # Setup current status and change place service (used when we have many to many case)
        fire_sync_service_name = 'fire_sync_transition_' + str(self.__robot_id)
        fire_sync_service = rospy.Service(fire_sync_service_name, FireSyncTransition, service_fire_sync)

        self.action_send_goal(self.__current_place, action_type, server_name)
        rospy.spin()

def main():

    namespace = str(rospy.get_namespace())
    splitted_1 = namespace.split("_")
    splitted_2 = splitted_1[1].split("/")
    node_name = "executor_" + str(splitted_2[0])
    print("Node Name", node_name)

    rospy.init_node(node_name)

    user_input_file = rospy.get_param("/user_input_file")
    user_input_path = rospy.get_param("/user_input_path")

    with open(user_input_file) as f:
        data = json.load(f)

    tool = gspn_tools.GSPNtools()
    to_open = user_input_path + data["gspn"]
    my_pn = tool.import_xml(to_open)[0]

    # After receiving the gspn, we need to analyze it
    resources = data["resources_list"]
    processed_resources = ast.literal_eval(resources)
    bool_accepted = analyze_gspn_structure(my_pn, processed_resources)
    print(bool_accepted)
    if bool_accepted == False:
        print("The input GSPN is not valid.")
        return

    p_to_c_mapping = ast.literal_eval(data["place_to_client_mapping"])
    # On the JSON I have to include Simple inside a string in order to work well.
    # This is due to the fact that JSON only accepts strings.
    # And so, now I need to parse it and change its value.
    for place in p_to_c_mapping:
        if p_to_c_mapping[place][0] == 'ExecGSPN':
            p_to_c_mapping[place][0] = ExecGSPNAction

    policy_dictionary = ast.literal_eval(data["policy_dictionary"])
    places_tuple = ast.literal_eval(data["places_tuple"])
    created_policy = policy.Policy(policy_dictionary, places_tuple)

    full_synchronization = ast.literal_eval(data["full_synchronization"])

    user_robot_id = rospy.get_param("~user_robot_id")
    user_current_place = rospy.get_param("~user_current_place")

    global GEN_CURRENT_PLACE
    global GEN_ROBOT_ID
    GEN_CURRENT_PLACE = user_current_place
    GEN_ROBOT_ID = int(user_robot_id)

    my_execution = GSPNExecutionROS(my_pn, p_to_c_mapping, resources, created_policy, str(user_current_place), int(user_robot_id), full_synchronization)
    #time.sleep(60)
    my_execution.ros_gspn_execution()


if __name__ == "__main__":
    main()
