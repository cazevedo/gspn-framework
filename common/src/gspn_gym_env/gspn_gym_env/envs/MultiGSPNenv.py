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
            # get execution time until next decision state; get reward
            elapsed_time, actions_info = self.get_execution_time(action)
            self.timestamp += elapsed_time

        else:
            if self.verbose:
                print('Transition not enabled')
            # stay in the same state, return reward -1, timestamp 0
            # reward -1 to discourage actions that do not change the system state

            reward = -1
            actions_info = ('action-not-available_'+str(action), -1)

        if self.verbose:
            print('Reward: ', reward)
            print('Timestamp: ', self.timestamp)
            print('Action info: ', actions_info)

        # get next state
        next_state = self.marking_to_state()
        if self.verbose:
            print("S': ", self.get_current_state())
            # print("S': ", next_state)
            print()

        episode_done = False

        return next_state, reward, episode_done, \
               {'timestamp': self.timestamp, 'actions_info': actions_info}

    def reset(self):
        self.timestamp = 0
        self.mr_gspn.reset_simulation()
        current_state = self.marking_to_state()

        return current_state, {'timestamp': self.timestamp, 'actions_info': []}

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
        action = '_' + str(action)
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
        reward = 0
        # if 'L4' in sparse_state.keys() and transition == '_6':
        #     reward = 1
        # elif 'L3' in sparse_state.keys() and transition == '_7':
        #     reward = 1

        if 'L1' in sparse_state.keys() and transition == '_0':
            reward = 1
        elif 'L1' in sparse_state.keys() and transition == '_1':
            reward = 1
        elif 'L4' in sparse_state.keys() and transition == '_5':
            reward = 1

        return reward

    def fire_timed_transitions(self):
        enabled_exp_transitions, enabled_imm_transitions = self.mr_gspn.get_enabled_transitions()

        wait_times = enabled_exp_transitions.copy()
        # sample from each exponential distribution prob_dist(x) = lambda * exp(-lambda * x)
        # in this case the beta rate parameter is used instead, where beta = 1/lambda
        for key, value in enabled_exp_transitions.items():
            wait_times[key] = np.random.exponential(scale=(1.0 / value), size=None)

        timed_transition = min(wait_times, key=wait_times.get)
        wait_until_fire = wait_times[timed_transition]

        self.mr_gspn.fire_transition(timed_transition)

        return wait_until_fire, timed_transition

    def get_execution_time(self, action):
        total_elapsed_time = 0
        actions_info = ('action-available_'+str(action), 0)

        enabled_timed_transitions, enabled_imm_transitions = self.mr_gspn.get_enabled_transitions()
        while(enabled_timed_transitions and not enabled_imm_transitions):
            action_time, timed_transition = self.fire_timed_transitions()
            total_elapsed_time += action_time
            # actions_info.append((timed_transition, action_time))
            actions_info = (timed_transition, action_time)

            enabled_timed_transitions, enabled_imm_transitions = self.mr_gspn.get_enabled_transitions()

        return total_elapsed_time, actions_info

    def get_action_info_attributes(self, action):
        # print('att')
        # print(action)
        action_name = action[0]
        action_number = int(action_name.split('_')[-1])
        action_time = action[1]

        # print(action_name)
        # print(action_number)
        # print(action_time)

        return action_name, action_number, action_time

    def get_rates_ground_truth(self):
        timed_transitions = self.mr_gspn.get_timed_transitions()
        true_rates = {}
        for name, rate in timed_transitions.items():
            action = int(name.split('_')[-1])
            true_rates[action] = rate

        return true_rates

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]