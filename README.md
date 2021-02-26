<!-- TABLE OF CONTENTS -->
## Software Toolbox To Model, Plan and Execute Multi-Robot Tasks
 

## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
* [Usage Examples](#usage-examples)
  * [Standalone version usage](#standalone-version-usage)
  * [ROS version usage](#ros-version-usage)
* [Contributing](#contributing)
* [License](#license)



<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com)

This package extended the package created on https://github.com/cazevedo/gspn-framework by adding two new modules: the execution module and the visualization module. The architecture is generally presented in ![here](https://github.com/PedroACaldeira/gspn_framework_package/blob/master/imgs/framework_end.pdf?raw=true). And more specifically in ![here](https://github.com/PedroACaldeira/gspn_framework_package/blob/master/imgs/full-architecture.pdf?raw=true). -->

### Built With
This framework was generally built with: 
* [Concurrent futures](https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures)
* [Vis.js](https://visjs.org/)
* [Sparse](https://pypi.org/project/sparse/)
* [Numpy](https://numpy.org/)
* [Flask](https://flask.palletsprojects.com/en/1.1.x/)
* [Json](https://www.json.org/json-en.html)

The ROS Noetic implementation was built with:
* [Actionlib](http://wiki.ros.org/actionlib)

Framework tested in Gazebo using:
* [Movebase](http://wiki.ros.org/move_base)
* [Amcl](http://wiki.ros.org/amcl)
* [turtlebot3 gazebo](http://wiki.ros.org/turtlebot3_gazebo)


<!-- GETTING STARTED -->
## Getting Started

The necessary packages of software depend on the version of the framework that you wish to use. 
If you do not want to use the package with robots, you must run the following commands to install the necessary software:
* sparse
```sh
pip install sparse
```
* numpy
```sh
pip install numpy
```
* flask
```sh
pip install Flask
```
On the other hand, if you do wish to use this framework with robots, then you must run the previously mentioned commands and also install the remaining packages with:
```sh
sudo apt-get install ros-noetic-PACKAGE_NAME
```
Regarding the turtlebot3 packages, you must clone the repositories from the provided links because they are currently not available for ROS Noetic. 



<!-- USAGE EXAMPLES -->
## Usage Examples
Since this framework is composed by two different implementations, there are two different ways of running our proposed examples. 

### Standalone version usage 
Before running the example for the standalone version, you should change the "project_path" in the example JSON input file (gspn_execution_input.json) because that path is set to my computer and it will not work in yours. 
After doing so, you can either run the full standalone version (visualization + execution) or simply the execution module. 
An important note to take is that since the GSPN used on the example JSON input file has no immediate transitions, "places_tuple" and "policy_dictionary" are not used. However, those values are set for illustrative reasons. 
THh input file uses a very simple GSPN with three places and two exponential transitions. The functions used are also very simple as well (they only print a message into the terminal).  

In order to run the full standalone version, all you need to do is run the following command inside gspn_framework_package/common/src/gspn_framework_package/:
```sh
python3 gspn_visualization.py
```
Afterwards, click on the link on the terminal next to "Running on", choose the example JSON input and play around with the user interface. 

However, if you only want to run the execution module, you must run the following command inside gspn_framework_package/common/src/gspn_framework_package/:
```sh
python3 gspn_execution.py
```
On this case, you will be queried about the whereabouts of the input file and you can provide the path to the example JSON input file. 

<!-- USAGE EXAMPLES -->
### ROS version usage 
In order to use the ROS version, you can use the example provided inside gspn_framework_package/ros/Example . This example does not use the Gazzebo simulator and instead, it only uses our framework. Before running it, you will have to change the path to the gspn on example_input.json, the "user_input_file" on example_robots.launch and the "user_input_file" and "user_input_path" on example_servers.launch . Afterwards, run the following commands on separate terminals:
```sh
roscore
```
```sh
roslaunch gspn_framework_package example_servers.launch
```
The previous command will also open a terminal with the online visualization module, which you can open by clicking on the link provided by "Running on".
```sh
roslaunch gspn_framework_package example_robots.launch
```

<!-- CONTRIBUTING -->
## Contributing

Since we can never achieve our greatest potential alone, you can always contribute to this project if you are interested in it. 

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


