#! /usr/bin/env python3
# Standard libs
import os
import sys
import pickle
import torch
import re
# Custom libs
# from gspn_lib import gspn_tools
import gspn_lib
import infinite_horizon_mr.solvers.actor_critic_multi_lra as solver
import infinite_horizon_mr.utils.build_solar_farm_model as build_model_util

# # ROS libs
# import rospy
# import actionlib
# # ROS msgs
# from std_msgs.msg import Bool, String
# import multi_jackal_tutorials.msg

use_cuda = torch.cuda.is_available()
print('CUDA available: ', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

class GSPNExecutor:
    '''
    Coordinates multi-robot systems following a given GSPN model and policy.
    '''
    def __init__(self, gpsn=None, policy=None, robot_map=None, action_map=None):
        self.load_models()
        # self.exec_env = gym.make('gspn_gym_env:multi-gspn-env-v1', gspn_model=self.gspn,
        #                  n_locations=self.n_locations, n_robots=self.n_robots, use_expected_time=True,
        #                  actions_maps=[self.actions_map, self.actions_map_inv],
        #                  reward_function=self.reward_function_type, verbose=False, idd='execute')
        #
        # rospy.loginfo('Connecting to navigation server...')
        # self.nav_client = actionlib.SimpleActionClient('/jackal2/NavigateWaypointActionServer',
        #                                           multi_jackal_tutorials.msg.NavigateWaypointAction)
        # rospy.loginfo('Waiting for navigation server...')
        # self.nav_client.wait_for_server()
        # rospy.loginfo('Connected to navigation server...')

        # self.event_in_msg = None
        # self.event_in_msg_received = False
        # self.scan = None
        # self.pub = rospy.Publisher('~event_out', String, queue_size=1)
        # rospy.Subscriber('~event_in', String, self.event_in_msg_callback)
        # self.scan_sub = None
        # rospy.loginfo('Ready to detect door status.')
        # self.loop_rate = rospy.Rate(rospy.get_param('~loop_rate', 10.0))
        # self.laser_scanner_topic = rospy.get_param('~scan_topic', 'scan_front')

    def run(self):
        current_marking = self.gspn.get_current_marking()

        en_timed_tr, en_imm_tr = self.gspn.get_enabled_transitions()
        # print(en_imm_tr)

        while en_imm_tr:
            action_transiton, action_gazebo = self.get_next_action(current_marking, en_imm_tr)
            print(action_transiton, action_gazebo)
            arcs_in, _ = self.gspn.get_connected_arcs(name=action_transiton, type='transition')
            print(arcs_in)

            robots_pl = []
            for pl in arcs_in:
                pl_name =  pl[0]
                if 'Available' not in pl_name:
                    robots_pl.append(pl_name)

            print(robots_pl)
            # get corresponding gazebo action server namespace + action_gazebo
            robots_ns = []
            for pl in robots_pl:
                robots_ns.append(self.robot_map[pl])

            self.gspn.fire_transition(action_transiton)
            for robot_namespace in robots_ns:
                rospy.loginfo('Connecting to navigation server...')
                # self.nav_client = actionlib.SimpleActionClient('/jackal0/NavigateWaypointActionServer',
                #                                                multi_jackal_tutorials.msg.NavigateWaypointAction)

                server_name = '/{}/{}'.format(robot_namespace, action_gazebo)
                self.action_client = actionlib.SimpleActionClient(server_name, msg_type)
                rospy.loginfo('Waiting for ' + server_name)
                self.action_client.wait_for_server()
                rospy.loginfo('Connected!')

            # TODO: add msg type to transition to action dict and read msg type in get_next_action

            sys.exit()


            # call gazebo actionlib



        nav_waypoint = multi_jackal_tutorials.msg.NavigateWaypointGoal()
        nav_waypoint.destination = 'panel1'
        nav_waypoint.origin = 'center'
        rospy.loginfo('Goal sent!')
        # check in the action server which one of these callbacks is necessary
        self.nav_client.send_goal(nav_waypoint,
                                active_cb=self.execute_cb,
                                feedback_cb=self.execute_cb,
                                done_cb=self.execute_cb)
        rospy.loginfo('Waiting for result...')
        self.nav_client.wait_for_result()
        print(self.nav_client.get_result())

    def execute_cb(self, result):
        pass
        # fire transition finished action

        # if random switch comes next, read battery level and fire it

        # update robot map

        # set flag to issue new action

    def get_next_action(self, marking, enabled_imm_transitions):
        enabled_actions_indexes = []
        for tr_name, tr_rate in enabled_imm_transitions.items():
            if tr_rate == 0:
                enabled_actions_indexes.append(self.actions_map_inv[tr_name])

        state = self.marking_to_state(marking)
        state = torch.tensor(state, device=device, dtype=torch.float32)
        action_idx = self.actor_critic.act(state, enabled_actions_indexes)
        action_transiton = self.actions_map[action_idx.item()]
        action_gazebo = self.get_gazebo_action(action_transiton)

        # action_transiton = 'L0_Charge'
        return action_transiton, action_gazebo

    def get_gazebo_action(self, transition):
        transition = re.sub('\d', '', transition)
        action_gazebo = self.transition_to_action[transition]

        return action_gazebo

    def marking_to_state(self, marking):
        # map dict marking to list marking
        state = [0]*len(marking.keys())
        for place_name, number_robots in marking.items():
            token_index = self.gspn.places_to_index[place_name]
            state[token_index] = number_robots

        return state

    def load_models(self):
        self.transition_to_action = {'L_L': [('/NavigateWaypointActionServer', multi_jackal_tutorials.msg.NavigateWaypointAction)],
                                    'L_Med_L': ['/NavigateWaypointActionServer'],
                                    'L_InspR': ['/InspectWaypointActionServer'],
                                    'L_InspL': ['/InspectWaypointActionServer'],
                                    'L_Med_InspR': ['/InspectWaypointActionServer'],
                                    'L_Med_InspL': ['/InspectWaypointActionServer'],
                                    'L_Charge': ['/ChargeJackalActionServer','/ChargeWarthogActionServer'],
                                    'L_MobileChargerL': ['/NavigateWaypointActionServer'],
                                    'L_MobileChargerWait': ['/WaitActionServer']}
        #
        # self.robot_map = {'jackal0': 'L0',
        #              'jackal1': 'L0_NavL1',
        #              'jackal2': 'L1_NavL2',
        #              'warthog': 'L0_MobileChargerNavL1'}

        self.robot_map = {'L0_Low': 'jackal0',
                     'L0_NavL1': 'jackal1',
                     'L1_NavL2': 'jackal2',
                     'L0_MobileCharger': 'warthog'}

        project_dir = os.path.dirname(solver.__file__)
        project_dir = project_dir.split('/infinite_horizon_mr')[0]

        # Set simulation parameters
        self.n_locations = 4
        self.n_robots = 2

        # activation_func = 'Tanh+ReLU'
        activation_func = 'LeakyReLU'
        # activation_func = 'Tanh'

        policy_version = 'best'
        # policy_version = 'last'

        # Set paths
        example_path = project_dir + '/bin/solar_farm_example'
        trained_models_path = example_path + '/trained_models'
        id_str = 'loc{}_r{}_{}_{}'.format(str(self.n_locations), str(self.n_robots),
                                          activation_func, policy_version)
        id_str_no_policy = '_'.join(id_str.split('_')[0:-1])

        policy_dir = '{}/model_sf_{}.pt'.format(trained_models_path, id_str)

        # # Load gspn model
        # gspn_file = '{}/gspn_sf_{}.pkl'.format(trained_models_path, id_str_no_policy)
        # with open(gspn_file, 'rb') as handle:
        #     self.gspn, _ = pickle.load(handle)
        #
        # # Load actions map
        # actions_map_file = '{}/action_map_sf_{}.pkl'.format(trained_models_path, id_str_no_policy)
        # with open(actions_map_file, 'rb') as handle:
        #     self.actions_map, self.actions_map_inv = pickle.load(handle)

        # --------------------
        gspn_model_path = example_path + '/gspn_model/SolarFarm_Location_template.PNPRO'
        self.gspn = build_model_util.build_model(self.n_locations, self.n_robots, gspn_model_path, loop_connect=False)

        # Create actions map
        imm_transitions = self.gspn.get_imm_transitions()
        self.actions_map = {}
        self.actions_map_inv = {}
        action_id = 0
        for tr_name, tr_rate in imm_transitions.items():
            if tr_rate == 0:
                self.actions_map[action_id] = tr_name
                self.actions_map_inv[tr_name] = action_id
                action_id += 1
        # -------------------

        # Actor-Critic
        self.actor_critic = solver.MultiAC_Exec(policy_path=policy_dir)

def main():
    # rospy.init_node('gspn_executor_node', anonymous=False)
    gspn_executor = GSPNExecutor()
    gspn_executor.run()

if __name__ == "__main__":
    main()