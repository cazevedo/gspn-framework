Software framework that encapsulates in one unified tool the methods to model and analyze multi-robot problems as well as to execute synthesized policies in real robots.

The main characteristic features and functionalities provided by the GSPN Toolbox are:
• Model multi-robot problems as a GSPNR, capturing uncertainty in the choices outcomes and actions duration.
• Logical and performance analysis of the designed models providing formal guar- antees over them.
• Systematic modeling framework that enables a design-analysis-design methodol- ogy, that leads to improved and more accurate models.
• Compliance with third-party tools through import and export functions, as well as, through a Python API that facilitates easy and rapid prototyping of other tools that take advantage of the objects and algorithms provided by this toolbox.
• A lightweight web-based GUI that allows the visualization of designed models, analysis metrics and simulate the token game for a specific policy.
• Provides centralized planning and decentralized execution of policies.
• The execution is integrated with ROS through actionlib and it allows the execution of policies directly on real robots or in a simulator.


USAGE:
1) Launch your multi-robot system, including all the required action servers (according to ExecGSPN.action template);
2) Set the input parameters (GSPN object, map between token position and place name, policy, etc) in the /config/execution_input_parameters.json file.
3) Set executor launch file and launch it

Brief description of the package files:

* Many to many scenario sync services msgs.
gspn_framework_package/ros/msgs/FireSyncTransition.srv
gspn_framework_package/ros/msgs/CurrentPlace.srv

* Action server action msg template.
gspn_framework_package/ros/action/ExecGSPN.action

* Message type of the Firing_Transition syncronization topic.
gspn_framework_package/ros/msgs/GSPNFiringData.msg

* Visualization frontend.
gspn_framework_package/ros/src/gspn_framework_package_ros/templates/gspn_visualization_home.html

* Visualization backend.
gspn_framework_package/ros/src/gspn_framework_package_ros/gspn_online_visualization_node.py 

* Example of a GSPN to be executed
gspn_framework_package/ros/config/gspn_temperature_patrol.xml

* Execution cooridnator module input parameters.
gspn_framework_package/ros/config/execution_input_parameters.json

* Execution coordinator module.
gspn_framework_package/ros/src/gspn_framework_package_ros/gspn_execution_node.py

* Execution coordinator launch file template.
gspn_framework_package/ros/src/launch/gspn_executor.launch

* Test case that launches a temperature inspection scenario
gspn_framework_package/ros/test/temperature_patrol.launch
