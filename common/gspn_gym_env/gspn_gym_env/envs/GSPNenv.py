import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from gspn_lib import gspn_tools
# import gspn_tools

class GSPNenv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, gspn_path, n_robots, verbose=False):
        self.verbose = verbose
        print('GSPN Gym Env')
        pn_tool = gspn_tools.GSPNtools()
        self.mr_gspn = pn_tool.import_greatspn(gspn_path)[0]
        # pn_tool.draw_gspn(mr_gspn)
        self.timestamp = 0

        # [0, 1, 1, 0, 1, 2, 0]
        self.observation_space = spaces.MultiDiscrete(nvec=[n_robots+1]*len(self.mr_gspn.get_current_marking()))

        # # [0.0...1.0]
        # self.action_space = spaces.Box(low=0.0, high=1.0,
        #                                shape=(1,), dtype=np.float32)

        # # {0,1}
        self.action_space = spaces.Discrete(2)

    def step(self, action):
        # get current state
        current_state = self.get_current_state()
        if self.verbose:
            print('S: ', current_state)

        # map input action to associated transition
        transition = self.action_to_transition(current_state, action)
        if self.verbose:
            print('Action: ', action, transition)

        # get reward
        reward = self.reward_function(current_state, transition)
        if self.verbose:
            print('Reward: ', reward)

        # apply action
        self.mr_gspn.fire_transition(transition)
        # get execution time until next decision state
        self.timestamp += self.get_execution_time()
        if self.verbose:
            print('Timestamp: ', self.timestamp)

        # get next state
        next_state = self.marking_to_state()
        if self.verbose:
            print("S': ", self.get_current_state())
            print("S': ", next_state)
            print()

        episode_done = False

        return next_state, reward, episode_done,\
               {'timestamp': self.timestamp}

    def reset(self):
        self.timestamp = 0
        self.mr_gspn.reset_simulation()
        current_state = self.marking_to_state()

        return current_state, {'timestamp': self.timestamp}

    def render(self, mode='human'):
        print('rendering not implemented')
        return True

    def close(self):
        self.reset()
        print('Au Revoir Shoshanna!')

    def get_current_state(self):
        sparse_state = self.mr_gspn.get_current_marking(sparse_marking=True)
        current_state = list(sparse_state.keys())[0]

        return current_state

    def action_to_transition(self, state, action):
        # if action > 0.5 then go through the left door else go throught the right door
        if action < 0.5:
            if self.verbose:
                print('took left')
            return 'left_'+state
        else:
            if self.verbose:
                print('took right')
            return 'right_'+state

    def marking_to_state(self):
        # map dict marking to list marking
        marking_dict = self.mr_gspn.get_current_marking(sparse_marking=True)
        next_location = list(marking_dict.keys())[0]
        state = [0]*len(self.mr_gspn.get_current_marking().keys())
        token_index = self.mr_gspn.places_to_index[next_location]
        state[token_index] = 1

        return state

    def reward_function(self, state, transition):
        reward = 0
        # Start + Left Door
        # if state == 'Start' and transition == 'left_Start':
        if state == 'Start' and transition == 'right_Start':
            reward = 10
        # Intermediate + Right Door
        # elif state == 'Intermediate' and transition == 'left_Intermediate':
        # elif state == 'Intermediate' and transition == 'right_Intermediate':
        elif state == 'Intermediate':
            reward = 10
        # End + Left Door
        # elif state == 'End' and transition == 'left_End':
        elif state == 'End' and transition == 'right_End':
            reward = 10

        reward = 10

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

        return wait_until_fire

    def get_execution_time(self):
        elapsed_time = 0

        enabled_timed_transitions, enabled_imm_transitions = self.mr_gspn.get_enabled_transitions()
        while(enabled_timed_transitions and not enabled_imm_transitions):
            elapsed_time += self.fire_timed_transitions()
            enabled_timed_transitions, enabled_imm_transitions = self.mr_gspn.get_enabled_transitions()

        return elapsed_time

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]