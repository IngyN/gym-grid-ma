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
        self.targets = self.gw.targets[:agents]
        # update init positions based on padding
        self.pos = np.add(self.pos, self.gw.pads)
        self.targets = np.add(self.targets, self.gw.pads)
        self.goal_flag = np.zeros(self.nagents)

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

    def step(self, actions, noop=True, distance=False, share=False, random_priority=True):

        # random priority
        priority = np.arange(self.nagents)
        if random_priority:
            priority = np.random.permutation(priority)

        rev_priority = priority[::-1]
        old_pos = self.pos
        desired = np.zeros([self.nagents, 2])
        visited = np.zeros([self.nagents])
        rewards = np.zeros([self.nagents])

        # find desired state
        # format: map size X #agents +1 : for each map cell : 1 boolean array of nagents + agent that is already there
        temp = np.ones([self.nrows, self.ncols, self.nagents + 1]) * -1
        # TODO give negative reward for trying to go out of bounds
        oob = np.zeros([self.nagents])

        for idx in range(self.nagents):
            i = rev_priority[idx]
            desired[i], oob[i] = self._get_next_state(self.pos[i], actions[i], self.goal_flag[i])
            temp[desired[i]][i] = 1
            temp[old_pos[i]][self.nagents] = i  # who is already there

        print(desired)

        for idx in range(self.nagents):
            i = priority[idx]

            if not visited[i] and not self.goal_flag[i]:
                visited[i] = 1
                occ = temp[desired[i]][:-1]  # who wants the spot -> boolean
                wall = self.gw.map[desired[i]]
                idx_occ = np.where(occ > 0)
                j = temp[desired[i]][self.nagents]
                d_sum = np.sum(occ[occ > 0])

                # check for swap
                swap = False
                if d_sum > 0 and j != i and j >= 0:
                    swap = (desired[j] == old_pos[i])

                # collisions
                if d_sum > 1:
                    if share and j >= 0:
                        if self.goal_flag[j] and (desired[i] == self.targets[i]):
                            self.pos[i] = desired[i]
                            temp[desired[i]][self.nagents] = i
                            temp[old_pos[i]][self.nagents] = -1

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
                    temp[desired[i]][self.nagents] = i
                    temp[old_pos[i]][self.nagents] = -1
                    self.pos[i] = desired[i]

                if noop and old_pos == self.pos:
                    rewards = -10

                # if distance:
                #     inrange = 1
                #     for k in range(self.nagents):
                #         # TODO distance rewards
                #         pass

            if not self.goal_flag[i] and self.pos[i] == self.targets[i]:
                rewards[i] = 100
                self.goal_flag[i] = 1

        done = np.all(self.goal_flag)

        return self.pos, rewards, {}, done  # TODO check obs again later

    def _get_next_state(self, pos, action, goal_flag):
        new_p = pos
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
        self.pos = self.gw.init[:self.nagents]
        self.goal_flag = np.zeros(self.nagents)
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
    env.step([4, 3])
    a = input()
    print(a)
