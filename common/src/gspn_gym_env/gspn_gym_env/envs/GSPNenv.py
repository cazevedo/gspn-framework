import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

class GSPNenv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        print('GSPN Gym Env')

    def step(self, action):
        print(action)
        return True

    def reset(self):
        return True

    def render(self, mode='human'):
        print('render')
        return True

    def close(self):
        print('Au Revoir Shoshanna!')

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]