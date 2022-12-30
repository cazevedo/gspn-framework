#! /usr/bin/env python3
# Standard libs
import os
import sys
import numpy as np
# ROS libs
import rospy
import actionlib
# Custom libs
# from gspn_lib import gspn_tools
import gspn_lib

# ROS msgs
from std_msgs.msg import Bool, String
import multi_jackal_tutorials.msg

class GSPNExecutor:
    '''
    Coordinates multi-robot systems following a given GSPN model and policy.
    '''
    def __init__(self):
        rospy.loginfo('Connecting to navigation server...')
        self.nav_client = actionlib.SimpleActionClient('/jackal2/NavigateWaypointActionServer',
                                                  multi_jackal_tutorials.msg.NavigateWaypointAction)
        rospy.loginfo('Waiting for navigation server...')
        self.nav_client.wait_for_server()
        rospy.loginfo('Connected to navigation server...')

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
        nav_waypoint = multi_jackal_tutorials.msg.NavigateWaypointGoal()
        nav_waypoint.destination = 'panel1'
        nav_waypoint.origin = 'center'
        rospy.loginfo('Goal sent!')
        self.nav_client.send_goal(nav_waypoint)
        rospy.loginfo('Waiting for result...')
        self.nav_client.wait_for_result()
        print(self.nav_client.get_result())

def main():
    rospy.init_node('gspn_executor_node', anonymous=False)
    gspn_executor = GSPNExecutor()
    gspn_executor.run()