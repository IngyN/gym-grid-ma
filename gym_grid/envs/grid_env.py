import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_grid.envs.gridworld import GridWorld

class GridEnv(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, map_name='example', agents= 2):
        # read map + initialize
        gw = GridWorld(map_name,agents)
        pass

    def step(self, action):

        #include collision mechanism.

        pass

    def reset(self):
        #reset to start.

        pass

    def render(self, mode='human'):
        pass

    def close(self):
        # close rendering
        pass