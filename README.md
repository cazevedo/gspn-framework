# GSPN Framework
Framework that allows the design, execution and analysis of generalized stochastic Petri nets (GSPN).
This is a Python framework compatible with the Petri Net Markup Language (PNML) standard that enables
quantitative (logical) and qualitative (performance) analysis of the designed GSPN providing:
* Reachability;
* Boundedness;
* Safety and deadlocks;
* Transition throughput rate;
* Probability of having k tokens in a place;
* Expected number of tokens in a place;
* Evolution of the transition probabilities for all states;
* Mean wait time of a place;

The designed Petri net can be simulated (token game), where the evolution of the net is updated at each transition.
The coverability tree and the equivalent continuous time Markov chain (CTMC) can also be obtained and visualized.


## Setup
```bash
pip install -r requirements.txt --upgrade
```

## Running the example
```bash
python example.py
```
Both gspn_tools.py and gspn.py have small examples that you can check by running each one of these files individually.
