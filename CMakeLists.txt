cmake_minimum_required(VERSION 2.8.3)
project(gspn_framework_package)

find_package(catkin REQUIRED
 genmsg
 actionlib_msgs
 std_msgs
 rospy
 message_generation
)

catkin_python_setup()

add_action_files(
  DIRECTORY
  ros/action
  FILES
  ExecGSPN.action
)

add_service_files(
  DIRECTORY
  ros/msgs
  FILES
  CurrentPlace.srv
  FireSyncTransition.srv
)

add_message_files(
  DIRECTORY
  ros/msgs
  FILES
  GSPNFiringData.msg
)

generate_messages(
  DEPENDENCIES
  actionlib_msgs
  std_msgs
)

catkin_package(
 CATKIN_DEPENDS
 message_runtime
 actionlib_msgs
 std_msgs
 rospy
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
