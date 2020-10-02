#! /usr/bin/env python3
import roslib
import rospy
import actionlib
import time

import gspn_framework_package.msg
'''
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
'''

class MinimalActionServer(object):

    _feedback = gspn_framework_package.msg.ExecGSPNFeedback()
    _result = gspn_framework_package.msg.ExecGSPNResult()

    def __init__(self, name):
        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, gspn_framework_package.msg.ExecGSPNAction,
                                                execute_cb=self.execute_callback, auto_start = False)
        self._as.start()
        rospy.loginfo('DECIDE L2 SERVER : ONLINE')
        self._as.start()

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        """Accepts or rejects a client request to begin an action."""
        # This server allows multiple goals in parallel
        self.get_logger().info('DECIDE L2 SERVER : Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accepts or rejects a client request to cancel an action."""
        self.get_logger().info('DECIDE L2 SERVER : Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal):

        success = True

        if success:
            self._result.transition = 'None'
            self._as.set_succeeded(self._result)

        return self._result.transition


if __name__ == '__main__':
    rospy.init_node('action_server_6')
    minimal_action_server = MinimalActionServer(rospy.get_name())
    rospy.spin()
