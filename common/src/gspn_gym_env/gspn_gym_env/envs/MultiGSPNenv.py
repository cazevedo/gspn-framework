import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from gspn_framework_package import gspn_tools
# import gspn_tools

class MultiGSPNenv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, gspn_path, set_actions=None, verbose=False):
        self.verbose = verbose
        print('Multi GSPN Gym Env')
        pn_tool = gspn_tools.GSPNtools()
        self.mr_gspn = pn_tool.import_greatspn(gspn_path)[0]
        # pn_tool.draw_gspn(mr_gspn)
        self.timestamp = 0

        n_robots = self.mr_gspn.get_number_of_tokens()

        # [0, 1, 1, 0, 1, 2, 0]
        self.observation_space = spaces.MultiDiscrete(nvec=[n_robots+1]*len(self.mr_gspn.get_current_marking()))

        # # [0.0...1.0]
        # self.action_space = spaces.Box(low=0.0, high=1.0,
        #                                shape=(1,), dtype=np.float32)

        if set_actions:
            # when the number of robots (tokens) is smaller than the number of locations (places/transitions)
            # the most efficient approach is define a set of actions that are common to every robot
            # and replicate it with different names for each robot
            n_actions = len(set_actions) * n_robots
        else:
            # get number of transitions in order to get number of actions
            # when the number of robots (tokens) is considerably bigger than the number of locations (places/transitions)
            # the most efficient approach is to use every single transition as an individual action
            imm_transitions = self.mr_gspn.get_imm_transitions()
            actions = imm_transitions.copy()
            for tr_name, tr_rate in imm_transitions.items():
                if tr_rate != 0:
                    del actions[tr_name]

            n_actions = len(actions.keys())

        # # {0,1,...,n_actions}
        self.action_space = spaces.Discrete(n_actions)

    def step(self, action):
        # get disabled actions in current state
        disabled_actions_names, disabled_actions_indexes = self.get_disabled_actions()

        # get current state
        current_state = self.get_current_state()
        if self.verbose:
            print('S: ', current_state)

        # map input action to associated transition
        transition = self.action_to_transition(action)
        if self.verbose:
            print('Action: ', action)

        if transition != None:
            # get reward
            reward = self.reward_function(current_state, transition)

            # apply action
            self.mr_gspn.fire_transition(transition)
            # get execution time (until next decision state)
            # get also the sequence of the fired transitions (name
            # get the transition rate
            elapsed_time, fired_transitions, transition_rate = self.get_execution_time(action)
            self.timestamp += elapsed_time

            # in a MRS the fired timed transition may not correspond to the selected action
            # this is the expected time that corresponds to the selected action
            # action_expected_time = 1.0 / transition_rate
            action_expected_time = self.get_action_time(action)

        else:
            if self.verbose:
                print('Transition not enabled')
            # stay in the same state, return reward -1, timestamp 0
            # reward -1 to discourage actions that do not change the system state

            reward = -1
            # actions_info = ('action-not-available_'+str(action), -1)
            action_expected_time = 0

        if self.verbose:
            print('Reward: ', reward)
            print('Timestamp: ', self.timestamp)
            print('Action expected time: ', action_expected_time)
            print('Actions disabled: ', disabled_actions_names)

        # get enabled actions in the next state
        next_state_enabled_actions_names, next_state_enabled_actions_indexes = self.get_enabled_actions()

        # get next state
        next_state = self.marking_to_state()
        # next_state_string = self.get_current_state()
        if self.verbose:
            print("S': ", self.get_current_state())
            # print("S': ", next_state)
            print()

        episode_done = False

        return next_state, reward, episode_done, \
               {'timestamp': self.timestamp,
                'disabled_actions': (disabled_actions_names, disabled_actions_indexes),
                'next_state_enabled_actions': (next_state_enabled_actions_names, next_state_enabled_actions_indexes),
                'action_time': action_expected_time}
                # 'next_state_string': next_state_string}

    def reset(self):
        self.timestamp = 0
        self.mr_gspn.reset_simulation()
        next_state = self.marking_to_state()

        # get enabled actions in the next state
        next_state_enabled_actions_names, next_state_enabled_actions_indexes = self.get_enabled_actions()

        return next_state, {'timestamp': self.timestamp, 'actions_info': [],
                               'disabled_actions': (None, None),
                               'next_state_enabled_actions': (
                               next_state_enabled_actions_names, next_state_enabled_actions_indexes),
                               'action_time': None}

    def render(self, mode='human'):
        print('rendering not implemented')
        return True

    def close(self):
        self.reset()
        print('Au Revoir Shoshanna!')

    def get_current_state(self):
        sparse_state = self.mr_gspn.get_current_marking(sparse_marking=True)
        # current_state = list(sparse_state.keys())[0]

        return sparse_state

    def action_to_transition(self, action):
        action = self.from_index_to_action(int(action))

        # check if action exists in the enabled transitions; if don't fire any transition
        _, enabled_actions = self.mr_gspn.get_enabled_transitions()

        if action in enabled_actions.keys():
            return action
        else:
            return None

    def marking_to_state(self):
        # map dict marking to list marking
        marking_dict = self.mr_gspn.get_current_marking(sparse_marking=True)
        state = [0]*len(self.mr_gspn.get_current_marking().keys())
        for place_name, number_robots in marking_dict.items():
            token_index = self.mr_gspn.places_to_index[place_name]
            state[token_index] = number_robots

        return state

    def reward_function(self, sparse_state, transition):
        reward = 0.0

        # inspection test
        # if 'L4' in sparse_state.keys() and transition == '_6':
        #     reward = 10

        # inspection 1
        # if 'L4' in sparse_state.keys() and transition == '_6':
        #     reward = 1
        # elif 'L3' in sparse_state.keys() and transition == '_7':
        #     reward = 1

        # if 'FullL4' in sparse_state.keys() and 'FullL3' in sparse_state.keys() and transition == '_9':
        #     reward = 1

        # robot scalability
        # if 'L4' in sparse_state.keys() and transition == '_5':
        #     reward = 1
        if 'L3' in sparse_state.keys() and transition == '_4':
            reward = 1

        return reward

    def fire_timed_transitions(self):
        enabled_exp_transitions, enabled_imm_transitions = self.mr_gspn.get_enabled_transitions()

        wait_times = enabled_exp_transitions.copy()
        # sample from each exponential distribution prob_dist(x) = lambda * exp(-lambda * x)
        # in this case the beta rate parameter is used instead, where beta = 1/lambda
        for key, value in enabled_exp_transitions.items():
            wait_times[key] = np.random.exponential(scale=(1.0 / value), size=None)
            # wait_times[key] = np.random.normal(loc=value, scale=10)

        timed_transition = min(wait_times, key=wait_times.get)
        wait_until_fire = wait_times[timed_transition]

        self.mr_gspn.fire_transition(timed_transition)

        timed_transition_rate = enabled_exp_transitions[timed_transition]

        return wait_until_fire, timed_transition, timed_transition_rate

    def get_execution_time(self, action):
        total_elapsed_time = 0
        actions_info = [('action-available_'+str(action), 0)]
        transitions_rate = []

        enabled_timed_transitions, enabled_imm_transitions = self.mr_gspn.get_enabled_transitions()
        while(enabled_timed_transitions and not enabled_imm_transitions):
            action_time, timed_transition, tr_rate = self.fire_timed_transitions()
            total_elapsed_time += action_time
            actions_info.append((timed_transition, action_time))
            transitions_rate.append(tr_rate)
            # actions_info = (timed_transition, action_time)

            enabled_timed_transitions, enabled_imm_transitions = self.mr_gspn.get_enabled_transitions()

        return total_elapsed_time, actions_info, transitions_rate

    def get_action_info_attributes(self, action):
        action_name = action[0]
        action_number = int(action_name.split('_')[-1])
        action_time = action[1]

        return action_name, action_number, action_time

    def get_rates_ground_truth(self):
        timed_transitions = self.mr_gspn.get_timed_transitions()
        true_rates = {}
        for name, rate in timed_transitions.items():
            action = int(name.split('_')[-1])
            true_rates[action] = rate

        return true_rates

    def from_index_to_action(self, action_index):
        return '_'+str(action_index)

    def from_action_to_index(self, action_name):
        return int(action_name.split('_')[-1])

    def get_disabled_actions(self):
        enabled_exp_transitions, enabled_imm_transitions = self.mr_gspn.get_enabled_transitions()

        disabled_actions_indexes = list(range(self.action_space.n))
        disabled_actions_names = list(map(self.from_index_to_action, disabled_actions_indexes))

        for enabled_action in enabled_imm_transitions.keys():
            disabled_actions_names.remove(enabled_action)
            action_index = int(enabled_action.split('_')[-1])
            disabled_actions_indexes.remove(action_index)

        return disabled_actions_names, disabled_actions_indexes

    def get_enabled_actions(self):
        enabled_exp_transitions, enabled_imm_transitions = self.mr_gspn.get_enabled_transitions()
        enabled_actions_names = list(enabled_imm_transitions.keys())
        enabled_actions_indexes = list(map(self.from_action_to_index, enabled_actions_names))

        return enabled_actions_names, enabled_actions_indexes

    def get_action_time(self, action):
        transition = 'Finished_'+str(action)
        transition_rate = self.mr_gspn.get_transition_rate(transition)
        action_expected_time = 1.0/transition_rate
        return action_expected_time

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]