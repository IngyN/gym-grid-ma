import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

terminal = True
#
if terminal:
    from gridworld import GridWorld
else:
    from gym_grid.envs.gridworld import GridWorld

from matplotlib import colors
import matplotlib.pyplot as plt

class GridEnv(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, map_name='example', agents= 2):
        # read map + initialize

        self.gw = GridWorld(map_name,agents)
        self.nrows = self.gw.map.shape[0]
        self.ncols = self.gw.map.shape[1]
        self.nagents = agents
        self.pos = self.gw.init[:agents]
        # update init positions based on padding
        self.pos = np.add(self.pos, self.gw.pads)

        # Rendering :
        # self.map_colors =colors.ListedColormap(['white', 'grey', 'red'])
        self.map_colors = plt.get_cmap('YlGnBu')
        self.norm = colors.BoundaryNorm([-100, 0, 20, 100], self.map_colors.N)
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(self.gw.map)

        print(self.pos)
        print(self.pos[:, 0])

    def step(self, actions):

        #include collision mechanism.
        # TODO: update self.pos
        pass

    def reset(self):
        #reset to start.
        self.pos = self.gw.init[:self.nagents]
        self.ax.clear()

    def render(self, mode='human'):
        print("Rendering...")
        self.ax.clear()
        # plot map
        self.ax.imshow(self.gw.map, cmap=self.map_colors)

        # plot agents
        # 3 = purple, 10 = yellow
        self.ax.scatter(self.pos[:, 1], self.pos[:, 0], c=[3, 10], s=55)

        # plot format
        self.ax.set_title('Map')
        self.ax.xaxis.set_ticks(np.arange(0, self.ncols, 1.0))
        self.ax.yaxis.set_ticks(np.arange(0, self.nrows, 1.0))
        self.ax.xaxis.tick_top()
        # self.fig.subplots_adjust(top=0.85)
        self.fig.canvas.draw()

    def close(self):
        # close any open log file or sth.
        pass


if __name__ == "__main__":
    env = GridEnv('SUNY')
    env.render()

    a = input()
    print(a)
