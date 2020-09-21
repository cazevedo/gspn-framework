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
        '''
        self._action_server = ActionServer(
            self,
            Simple,
            'simple_3',
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)
        '''
        rospy.loginfo('SERVER 3 : ONLINE')
        self._as.start()

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        """Accepts or rejects a client request to begin an action."""
        # This server allows multiple goals in parallel
        self.get_logger().info('SERVER 3 : Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accepts or rejects a client request to cancel an action."""
        self.get_logger().info('SERVER 3 : Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal):
        """Executes a goal."""
        rospy.loginfo('SERVER 3 : Executing goal...')

        success = True

        self._feedback.time_passed = []

        # Start executing the action
        for i in range(1, 5):
            if self._as.is_preempt_requested():
                self._as.set_preempted()
                success = False
                rospy.loginfo('SERVER 3 : Goal canceled')
                break
            # Update Fibonacci sequence
            self._feedback.time_passed.append(i)

            rospy.loginfo('SERVER 3 : Publishing feedback: {0}'.format(self._feedback.time_passed))

            # Publish the feedback
            self._as.publish_feedback(self._feedback)

            # Sleep for demonstration purposes
            time.sleep(1)

        if success:
            self._result.transition = 'None'
            rospy.loginfo('SERVER 3 : Returning result: {0}'.format(self._result.transition))
            self._as.set_succeeded(self._result)

        return self._result.transition


if __name__ == '__main__':
    rospy.init_node('action_server_3')
    minimal_action_server = MinimalActionServer(rospy.get_name())
    rospy.spin()
