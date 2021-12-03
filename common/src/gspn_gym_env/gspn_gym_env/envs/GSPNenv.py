import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import gspn
import gspn_tools
from pyglet.resource import location


class GSPNenv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, gspn_path):
        print('GSPN Gym Env')
        pn_tool = gspn_tools.GSPNtools()
        self.mr_gspn = pn_tool.import_greatspn(gspn_path)[0]
        # pn_tool.draw_gspn(mr_gspn)

        print(self.mr_gspn.get_current_marking())
        print(self.mr_gspn.places_to_index)

        sparse_state = self.mr_gspn.get_current_marking(sparse_marking=True)
        location = list(sparse_state.keys())[0]
        print('right_' + location)
        self.mr_gspn.fire_transition('right_' + location)

        sparse_state = self.mr_gspn.get_current_marking(sparse_marking=True)
        location = list(sparse_state.keys())[0]

        token_index = self.mr_gspn.places_to_index[location]
        print(token_index)

    def step(self, action):
        print(action)
        # state = self.mr_gspn.get_current_marking()
        sparse_state = self.mr_gspn.get_current_marking(sparse_marking=True)

        current_location = list(sparse_state.keys())[0]
        reward = self.reward_function(current_location, action)

        if action > 0.5:
            self.mr_gspn.fire_transition('left_'+current_location)
        else:
            self.mr_gspn.fire_transition('right_'+current_location)

        sparse_state = self.mr_gspn.get_current_marking(sparse_marking=True)
        next_location = list(sparse_state.keys())[0]

        next_state = [0]*len(self.mr_gspn.get_current_marking().keys())
        token_index = self.mr_gspn.places_to_index[next_location]
        next_state[token_index] = 1

        episode_done = False

        return True
        # return next_state, reward, episode_done,\
        #        {'timestamp': self.timestamp, 'observation_obj': observation_dict}

    def reset(self):
        return True

    def render(self, mode='human'):
        print('render')
        return True

    def close(self):
        print('Au Revoir Shoshanna!')

    def reward_function(self, state, action):
        reward = 0
        # if state == :


        return reward

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]