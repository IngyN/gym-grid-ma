import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import time
from copy import deepcopy

terminal = False
#
if terminal:
    from gridworld import GridWorld
else:
    from gym_grid.envs.gridworld import GridWorld

from matplotlib import colors
import matplotlib.pylab as plt
from gym import spaces


class GridEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, map_name='example', nagents=2, padding=False, debug=False, norender=True):
        # read map + initialize
        self.gw = GridWorld(map_name, nagents, padding)
        self.nrows = self.gw.map.shape[0]
        self.ncols = self.gw.map.shape[1]
        self.nagents = nagents
        self.pos = self.gw.init[:nagents]
        self.targets = self.gw.targets[:nagents]
        self.action_space = [spaces.Discrete(6) for _ in range(nagents)]  # 0- 5
        self.observation_space = spaces.MultiDiscrete([self.nrows, self.ncols])
        # update init positions based on padding
        if padding:
            self.pos = np.add(self.pos, self.gw.pads).astype(int)
            self.targets = np.add(self.targets, self.gw.pads).astype(int)

        self.start_pos = deepcopy(self.pos)
        self.goal_flag = np.zeros(self.nagents, dtype=int)
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

        # plt.ion()
        if not norender:
            self.fig = plt.figure(num=0)
            self.ax = self.fig.add_subplot(111)
            # self.render()
            # plt.show(block=False)
            plt.ion()
        self.norender = norender

        if self.debug:
            print(self.pos)
            print(self.pos[:, 0])

    def set_targets(self, targets):  # change preset targets.
        self.targets = np.array([targets])

    def set_start(self, start_pos):
        self.pos = np.array([start_pos])
        self.start_pos = deepcopy(self.pos)

    def step(self, actions, noop=True, distance=False, share=False, random_priority=True, collision_cost = 10):

        # random priority
        priority = np.arange(self.nagents)
        if random_priority:
            priority = np.random.permutation(priority)

        rev_priority = priority[::-1]
        old_pos = self.pos.astype(int)
        desired = np.zeros([self.nagents, 2], dtype=int)
        visited = np.zeros([self.nagents])
        rewards = np.zeros([self.nagents])
        coll = 0

        # find desired state
        # format: map size X #agents +1 : for each map cell : 1 boolean array of nagents + agent that is already there
        temp = np.ones([self.nrows, self.ncols, self.nagents + 1]) * -1
        oob = np.zeros([self.nagents])

        for idx in range(self.nagents):
            i = rev_priority[idx]
            desired[i], oob[i] = self.get_next_state(self.pos[i], actions[i], self.goal_flag[i])
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
                j = int(temp[desired[i][0]][desired[i][1]][self.nagents])  # agent already in the spot
                d_sum = np.sum(occ[occ > 0])

                # check for swap
                swap = False
                if d_sum > 0 and j != i and j >= 0:
                    swap = np.all(desired[j] == old_pos[i])

                # collisions
                if wall:
                    rewards[i] = -10

                elif oob[i]:  # out of bounds
                    rewards[i] = -10

                elif d_sum > 1:  # more than one agent wants it
                    if share and j >= 0:
                        if self.goal_flag[j] and (desired[i] == self.targets[i]):
                            self.pos[i] = desired[i]
                            temp[desired[i][0]][desired[i][1]][self.nagents] = i
                            temp[old_pos[i][0]][old_pos[i][1]][self.nagents] = -1

                    else:
                        for k in idx_occ[0]:
                            visited[k] = 1
                            rewards[k] = -collision_cost
                            temp[desired[i][0]][desired[i][1]][k] = -1
                        coll += 1
                elif swap:
                    rewards[i] = -collision_cost
                    rewards[j] = -collision_cost
                    visited[j] = 1
                    coll += 1

                elif j != i and j >= 0:  # there's already someone there but we're not sure they will move.
                    # if  oob,  or wall -> will not move for sure
                    if self.gw.map[desired[j][0]][desired[j][1]] or oob[j]:
                        # agent will not move
                        rewards[i] = -collision_cost
                        temp[desired[i][0]][desired[i][1]][i] = -1
                        coll += 1

                    elif int(temp[desired[j][0]][desired[j][1]][self.nagents]) > 0:  # if occ
                        # needs to wait TODO: make this recursive to go more than 1 step ahead. rethink tree idea
                        rewards[i] = -collision_cost
                        temp[desired[i][0]][desired[i][1]][i] = -1
                        coll += 1

                    elif visited[j] and temp[desired[j][0]][desired[j][1]][j] == -1:
                        # if this agent was visited and will not move
                        rewards[i] = -collision_cost
                        temp[desired[j][0]][desired[j][1]][i] = -1
                        coll += 1

                    else:
                        # can move
                        self.pos[i] = desired[i]
                        rewards[i] = -1

                else:
                    rewards[i] = -1
                    temp[desired[i][0]][desired[i][1]][self.nagents] = i
                    temp[old_pos[i][0]][old_pos[i][1]][self.nagents] = -1
                    self.pos[i] = desired[i]

                if noop and np.all(desired[i] == self.pos[i]):
                    rewards[i] = -10

                # if distance:
                #     inrange = 1
                #     for k in range(self.nagents):
                #         # TODO distance rewards
                #         pass

        for idx in range(self.nagents):
            i = priority[idx]
            if self.goal_flag[i]:
                rewards[i] = 0

            if not self.goal_flag[i] and np.all(self.pos[i] == self.targets[i]):
                rewards[i] = 100
                self.goal_flag[i] = 1

        done = np.all(self.goal_flag)

        return self.pos, rewards, {'collisions': coll}, self.goal_flag

    def get_next_state(self, pos, action, goal_flag):
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

    def reset(self, debug=False):
        # reset to start.
        if debug:
            print('starting at :', self.start_pos)
        self.pos = deepcopy(self.start_pos)
        self.goal_flag = np.zeros(self.nagents, dtype=int)
        if not self.norender:
            self.ax.clear()

        return [np.array(self.pos[i]) for i in range(self.nagents)]

    def render(self, episode=-1, mode='human', speed=1):
        # print("Rendering...")
        plt.ion()
        if speed != 0:
            self.ax.clear()
            # plot map
            self.ax.imshow(self.gw.map, cmap=self.map_colors)

            # plot agents
            # 3 = purple, 10 = yellow
            p_colors = [3, 10, 5, 8, 15]
            self.ax.scatter(self.pos[:, 1], self.pos[:, 0], c=p_colors[:self.nagents], s=110)

            # plot format
            if episode == -1:
                self.ax.set_title('Map')
            else:
                self.ax.set_title('Map - episode:' + str(episode))
            self.ax.xaxis.set_ticks(np.arange(0, self.ncols, 1.0))
            self.ax.yaxis.set_ticks(np.arange(0, self.nrows, 1.0))
            self.ax.xaxis.tick_top()

            self.ax.relim()
            self.ax.autoscale_view(True, True, True)
            # self.fig.subplots_adjust(top=0.85)
            self.fig.canvas.draw()
            plt.show(block=False)
            # self.fig.show()

            # self.fig.canvas.draw_idle()
            if speed == 1:
                plt.pause(0.05)
            elif speed == 2:
                plt.pause(0.02)
            else:
                plt.pause(0.1)

    def final_render(self):
        plt.ioff()
        plt.show(block=True)

    def close(self):
        # close any open log file or sth.
        plt.close('all')


if __name__ == "__main__":
    env = GridEnv(map_name='Pentagon', nagents=2, norender=False, padding=True)
    # env.render()
    # a = input('next:\n')
    env.pos = np.array([[7, 2], [6, 3]])
    obs, rew, _, _ = env.step([1, 3])
    # env.render()
    print("Obs: ", obs, "  rew: ", rew)
    # a = input('next:\n')
    obs, rew, _, _ = env.step([3, 1])
    # env.render()
    print("Obs: ", obs, "  rew: ", rew)
    # a = input('next:\n')
    obs, rew, _, _ = env.step([2, 1])
    # env.render()
    print("Obs: ", obs, "  rew: ", rew)
    # a = input('next:\n')

    # env.final_render()
