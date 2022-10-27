## Software Toolbox to Model, Plan and Execute Multi-Robot Tasks

## About the Package
ROS package that provides, through a user-friendly API, methods to model multi-robot coordination problems as generalized stochastic Petri nets with rewards (GSPNR). This formalism provides a compact way of capturing: action selection, uncertainty on the duration of action execution, and team goals. This package also implements algorithms that obtain optimal policies for the GSPNR model, while reasoning over uncertainty and optimizing team-level objectives. Furthermore, it is also integrated with ROS middleware and thus managing the execution in real multi-robot systems.

## Getting Started
Clone the repo:
```bash
git clone --recurse-submodules https://github.com/cazevedo/gspn-framework.git
```
Install the submodules:
```bash
cd gspn-framework/common/gspn-lib
pip install -e .
```
```bash
cd gspn-framework/common/gspn-gym-env
pip install -e .
```
Build the package:
```bash
catkin build --this
```
