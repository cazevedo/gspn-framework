#! /usr/bin/env python3
# Standard libs
import os
import sys
import pickle
import gym
# Custom libs
# from gspn_lib import gspn_tools
import gspn_lib
import infinite_horizon_mr.solvers.actor_critic_multi_lra as solver

# # ROS libs
# import rospy
# import actionlib
# # ROS msgs
# from std_msgs.msg import Bool, String
# import multi_jackal_tutorials.msg

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

        while en_imm_tr:
            action_transiton, action_gazebo = self.get_next_action(current_marking, en_imm_tr)
            self.gspn.fire_transition(action_transiton)

            # get corresponding gazebo action server namespace + action_gazebo

            # call gazebo actionlib

        # nav_waypoint = multi_jackal_tutorials.msg.NavigateWaypointGoal()
        # nav_waypoint.destination = 'panel1'
        # nav_waypoint.origin = 'center'
        # rospy.loginfo('Goal sent!')
        # check in the action server which one of these callbacks is necessary
        # self.nav_client.send_goal(nav_waypoint,
        #                         active_cb=self.execute_cb,
        #                         feedback_cb=self.execute_cb,
        #                         done_cb=self.execute_cb)
        # rospy.loginfo('Waiting for result...')
        # self.nav_client.wait_for_result()
        # print(self.nav_client.get_result())

    def execute_cb(self, result):
        pass

    def update_gspn(self, transition):
        self.gspn.fire_transition(transition)

    def get_next_action(self, marking, enabled_imm_transitions):
        enabled_actions_indexes = []
        for tr_name, tr_rate in enabled_imm_transitions.items():
            if tr_rate == 0:
                enabled_actions_indexes.append(self.actions_map_inv[tr_name])

        action_idx = self.actor_critic.act(marking, enabled_actions_indexes)
        action_transiton = self.actions_map[action_idx]
        action_gazebo = self.transition_to_action[action_transiton]

        return action_transiton, action_gazebo

    def load_models(self):
        self.transition_to_action = {'L<>_L<>': ['/NavigateWaypointActionServer'],
                                    'L<>_Med_L<>': ['/NavigateWaypointActionServer'],
                                    'L<>_InspR': ['/InspectWaypointActionServer'],
                                    'L<>_InspL': ['/InspectWaypointActionServer'],
                                    'L<>_Med_InspR': ['/InspectWaypointActionServer'],
                                    'L<>_Med_InspL': ['/InspectWaypointActionServer'],
                                    'L<>_Charge': ['/ChargeJackalActionServer','/ChargeWarthogActionServer'],
                                    'L<>_MobileChargerL<>': ['/NavigateWaypointActionServer'],
                                    'L<>_MobileChargerWait': ['/WaitActionServer']}

        self.robot_map = {'jackal0': 'L0',
                     'jackal1': 'L0_NavL1',
                     'jackal2': 'L1_NavL2',
                     'warthog': 'L0_MobileChargerNavL1'}

        project_dir = os.path.dirname(solver.__file__)
        project_dir = project_dir.split('/infinite_horizon_mr')[0]

        # Set simulation parameters
        self.n_locations = 4
        self.n_robots = 2

        # activation_func = 'Tanh+ReLU'
        # activation_func = 'ReLU'
        activation_func = 'Tanh'

        policy_version = 'best'
        # policy_version = 'last'

        # Set paths
        example_path = project_dir + '/bin/solar_farm_example'
        trained_models_path = example_path + '/trained_models'
        id_str = 'loc{}_r{}_{}_{}'.format(str(self.n_locations), str(self.n_robots),
                                          activation_func, policy_version)
        id_str_no_policy = '_'.join(id_str.split('_')[0:-1])

        policy_dir = '{}/model_sf_{}.pt'.format(trained_models_path, id_str)

        # Load gspn model
        gspn_file = '{}/gspn_sf_{}.pkl'.format(trained_models_path, id_str_no_policy)
        with open(gspn_file, 'rb') as handle:
            self.gspn, _ = pickle.load(handle)

        # Load actions map
        actions_map_file = '{}/action_map_sf_{}.pkl'.format(trained_models_path, id_str_no_policy)
        with open(actions_map_file, 'rb') as handle:
            self.actions_map, self.actions_map_inv = pickle.load(handle)

        # Actor-Critic
        self.actor_critic = solver.MultiAC_Exec(policy_path=policy_dir)

def main():
    # rospy.init_node('gspn_executor_node', anonymous=False)
    gspn_executor = GSPNExecutor()
    gspn_executor.load_models()
    # gspn_executor.run()

if __name__ == "__main__":
    main()