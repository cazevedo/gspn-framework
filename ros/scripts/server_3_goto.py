#! /usr/bin/env python3
import roslib
import rospy
import actionlib
import time

import gspn_framework_package.msg
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

class MinimalActionServer(object):

    _feedback = gspn_framework_package.msg.ExecGSPNFeedback()
    _result = gspn_framework_package.msg.ExecGSPNResult()

    def __init__(self, name, _namespace):
        self._action_name = name
        self._namespace = _namespace
        self._as = actionlib.SimpleActionServer(self._action_name, gspn_framework_package.msg.ExecGSPNAction,
                                                execute_cb=self.execute_callback, auto_start = False)
        self._as.start()
        '''
        self._action_server = ActionServer(
            self,
            Simple,
            'simple_1',
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

        # This server creates an action client to connect with move_base
        client = actionlib.SimpleActionClient('/'+self._namespace+'/move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        client.wait_for_server()
        rospy.loginfo("Server available, executing action...")

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = -0.5
        goal.target_pose.pose.position.y = 1.5
        goal.target_pose.pose.orientation.w = 1.0

        self._feedback.time_passed.append(1)
        time.sleep(1)

        client.send_goal(goal)
        wait = client.wait_for_result()

        if not wait:
            rospy.logerr("Action server not available!")
            rospy.signal_shutdown("Action server not available!")
        else:
            print("The move_base result was ", client.get_result())
            self._result.transition = 'None'
            rospy.loginfo('SERVER 3 : Returning result: {0}'.format(self._result.transition))
            self._as.set_succeeded(self._result)
        return self._result.transition


if __name__ == '__main__':
    rospy.init_node('action_server_3')
    _namespace = str(rospy.get_name()).split("/")[1]
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", _namespace)
    minimal_action_server = MinimalActionServer(rospy.get_name(), _namespace)
    rospy.spin()
