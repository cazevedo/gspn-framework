# GSPN Framework
Framework that allows the design, execution and analysis of systems using generalized stochastic Petri nets (GSPN).
This is a Python framework that enables logical and performance analysis of the designed GSPN providing:
* Reachability;
* Boundedness;
* Safety and deadlocks;
* Transition throughput rate;
* Probability of having k tokens in a place;
* Expected number of tokens in a place;
* Evolution of the transition probabilities for all states;
* Mean wait time of a place;

The generalized stochastic Petri net can either be manually designed using this framework or imported from other tools, such as [PIPE](https://github.com/sarahtattersall/PIPE), that offers a GUI interface.
The designed Petri net can be simulated (token game), where the evolution of the net is updated at each transition.
The coverability tree and the equivalent continuous time Markov chain (CTMC) can also be obtained and visualized.


## Setup
```bash
pip install -r requirements.txt --upgrade
```

## Running the example
To run an example that shows some of the analysis capabilities of the framework just run:
```bash
python example_analysis.py
```

To run an example that shows how to expand a Petri net and how to execute it just run:
```bash
python example_execution.py
```

Both gspn_tools.py and gspn.py have small examples that you can check by running each one of these files individually.
