import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import time

terminal = False
#
if terminal:
    from gridworld import GridWorld
else:
    from gym_grid.envs.gridworld import GridWorld

from matplotlib import colors
import matplotlib.pylab as plt

class GridEnv(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, map_name='example', agents=2, padding=False, debug=False):
        # read map + initialize
        # TODO add version without padding
        self.gw = GridWorld(map_name, agents, padding)
        self.nrows = self.gw.map.shape[0]
        self.ncols = self.gw.map.shape[1]
        self.nagents = agents
        self.pos = self.gw.init[:agents]
        self.targets = self.gw.targets[:agents]
        # update init positions based on padding
        if padding:
            self.pos = np.add(self.pos, self.gw.pads)
            self.targets = np.add(self.targets, self.gw.pads)

        self.start_pos = self.pos
        self.goal_flag = np.zeros(self.nagents)
        self.debug = debug

        # Rendering :
        # self.map_colors =colors.ListedColormap(['white', 'grey'])
        colormap = plt.get_cmap('ocean', 50)
        vircolors = colormap(np.linspace(0, 1, 50))
        self.map_colors = np.zeros([2, 4])
        self.map_colors[1, :] = np.array([0.15, 0.18, 0.25, 1])
        self.map_colors[0, :] = np.array([1, 1, 1, 1])
        self.map_colors = colors.ListedColormap(self.map_colors)
        # self.map_colors = plt.get_cmap('Greys')
        self.norm = colors.BoundaryNorm([0, 0, 1, 1], self.map_colors.N)

        plt.ioff()
        # plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.render(first=True)
        plt.show(block=False)
        plt.ion()

        if self.debug:
            print(self.pos)
            print(self.pos[:, 0])

    def step(self, actions, noop=True, distance=False, share=False, random_priority=True):

        # random priority
        priority = np.arange(self.nagents)
        if random_priority:
            priority = np.random.permutation(priority)

        rev_priority = priority[::-1]
        old_pos = self.pos.astype(int)
        desired = np.zeros([self.nagents, 2], dtype=int)
        visited = np.zeros([self.nagents])
        rewards = np.zeros([self.nagents])

        # find desired state
        # format: map size X #agents +1 : for each map cell : 1 boolean array of nagents + agent that is already there
        temp = np.ones([self.nrows, self.ncols, self.nagents + 1]) * -1
        oob = np.zeros([self.nagents])

        for idx in range(self.nagents):
            i = rev_priority[idx]
            desired[i], oob[i] = self._get_next_state(self.pos[i], actions[i], self.goal_flag[i])
            temp[desired[i][0]][desired[i][1]][i] = 1
            temp[old_pos[i][0]][old_pos[i][1]][self.nagents] = i  # who is already there

        if self.debug:
            print('desired:', desired)

        for idx in range(self.nagents):
            i = priority[idx]

            if not visited[i] and not self.goal_flag[i]:
                visited[i] = 1
                occ = temp[desired[i][0]][desired[i][1]][:-1]  # who wants the spot -> boolean
                wall = self.gw.map[desired[i][0]][desired[i][1]]
                idx_occ = np.where(occ > 0)
                j = int(temp[desired[i][0]][desired[i][1]][self.nagents])
                d_sum = np.sum(occ[occ > 0])

                # check for swap
                swap = False
                if d_sum > 0 and j != i and j >= 0:
                    swap = np.all(desired[j] == old_pos[i])

                # collisions
                if d_sum > 1:
                    if share and j >= 0:
                        if self.goal_flag[j] and (desired[i] == self.targets[i]):
                            self.pos[i] = desired[i]
                            temp[desired[i][0]][desired[i][1]][self.nagents] = i
                            temp[old_pos[i][0]][old_pos[i][1]][self.nagents] = -1

                    else:
                        for k in idx_occ:
                            visited[k] = 1
                            rewards[k] = -10

                elif wall:
                    rewards[i] = -10

                elif swap:
                    rewards[i] = -10
                    rewards[j] = -10

                elif oob[i]:  # out of bounds
                    rewards[i] = -10

                else:
                    rewards[i] = -1
                    temp[desired[i][0]][desired[i][1]][self.nagents] = i
                    temp[old_pos[i][0]][old_pos[i][1]][self.nagents] = -1
                    self.pos[i] = desired[i]

                if noop and np.all(old_pos[i] == self.pos[i]):
                    rewards[i] = -10

                # if distance:
                #     inrange = 1
                #     for k in range(self.nagents):
                #         # TODO distance rewards
                #         pass

            if not self.goal_flag[i] and np.all(self.pos[i] == self.targets[i]):
                rewards[i] = 100
                self.goal_flag[i] = 1

        done = np.all(self.goal_flag)

        return self.pos, rewards, {}, done  # TODO check obs again later

    def _get_next_state(self, pos, action, goal_flag):
        new_p = pos.astype(int)
        oob = True
        # action 0 = nothing
        if not goal_flag:
            if action == 1:  # up
                if pos[0] > 0:
                    new_p[0] -= 1
                    oob = False

            if action == 2:  # down
                if pos[0] < self.nrows - 1:
                    new_p[0] += 1
                    oob = False

            if action == 3:  # left
                if pos[1] > 0:
                    new_p[1] -= 1
                    oob = False

            if action == 4:  # right
                if pos[1] < self.ncols - 1:
                    new_p[1] += 1
                    oob = False

        return new_p, oob

    def reset(self):
        #reset to start.
        self.pos = self.start_pos
        self.goal_flag = np.zeros(self.nagents)
        self.ax.clear()

    def render(self, first=False, mode='human'):
        # print("Rendering...")
        self.ax.clear()
        # plot map
        self.ax.imshow(self.gw.map, cmap=self.map_colors)

        # plot agents
        # 3 = purple, 10 = yellow
        p_colors = [3, 10, 5, 8, 15]
        self.ax.scatter(self.pos[:, 1], self.pos[:, 0], c=p_colors[:self.nagents], s=110)

        # plot format
        self.ax.set_title('Map')
        self.ax.xaxis.set_ticks(np.arange(0, self.ncols, 1.0))
        self.ax.yaxis.set_ticks(np.arange(0, self.nrows, 1.0))
        self.ax.xaxis.tick_top()

        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        # self.fig.subplots_adjust(top=0.85)
        self.fig.canvas.draw()

        # self.fig.canvas.draw_idle()
        time.sleep(2)

    def close(self):
        # close any open log file or sth.
        plt.close('all')


if __name__ == "__main__":
    env = GridEnv()
    env.render()
    # a = input('next:\n')
    obs, rew, _, _ = env.step([4, 3])
    env.render()
    print("Obs: ", obs, "  rew: ", rew)
    # a = input('next:\n')
    obs, rew, _, _ = env.step([4, 2])
    print("Obs: ", obs, "  rew: ", rew)
    env.render()
    # a = input('next:\n')
    obs, rew, _, _ = env.step([4, 4])
    env.render()
    print("Obs: ", obs, "  rew: ", rew)
    # a = input('next:\n')
