from enum import Enum
import os

terminal = False
if terminal:
    from grid_preprocessing import preprocessing
else:
    from gym_grid.envs.grid_preprocessing import preprocessing
import numpy as np


class Action(Enum):
    stay = 0
    up = 1
    down = 2
    left = 3
    right = 4

    def all_actions(self):
        return [self.stay, self.up, self.down, self.left, self.right]

    def disp(self):
        return [[0, 0], [-1, 0], [1, 0], [0, -1],  [0, 1]]

class GridWorld:
    def __init__(self, map, n, padding):

        self.collision = True
        self.distance = False
        self.coop = False
        self.shared_goal = False
        self.nagents = int(n)
        self.res = [3, 3]  # width, length / columns, rows

        self.states = np.zeros(self.res)
        self.state_to_pos = np.zeros(np.concatenate((self.res, 2), axis=None))
        for i in range(self.res[1]):
            for map_j in range(self.res[0]):
                self.states[i][map_j] = i * self.res[1] + map_j
                self.state_to_pos[i][map_j] = [i, map_j]

        # print(self.state_to_pos)
        self.nactions = 5
        self.obstacle = False

        self.process_map(map)

        # read map
        with open('maps/' + map + ".txt") as textFile:
            self.map = [line.split() for line in textFile]
        self.map = [[int(e) for e in row] for row in self.map]
        self.map = np.asarray(self.map, dtype=np.float32)

        # grid preprocessing
        self.pads = np.zeros([2])
        if padding:
            self.map, self.smap, self.coord, self.pads = preprocessing(self.map, self.res)

        # self.nshields = int(np.max(self.smap))
        # print("Number of Shields: ", self.nshields, '\n')
        # self.shield_max = [2] * int(self.nshields)  # TODO -> make it based on shield morphology

        self.nstates = np.count_nonzero(self.map == 0)
        # print(self.map)
        # print(self.smap)

        # print('Number of reachable states : ', self.nstates)

        # print('Initialization complete')


    def is_target(self, n, x, y):

        if self.targets[n][0] == x and self.targets[n][1] == y:
            return True
        return False

    def get_next(self, state, shield_num, entry_states):
        valid = set()
        obs = set()
        corner = self.coord[shield_num]
        move = [[0, -1], [1, 0], [0, 1], [-1, 0]]  # left/down/right/up
        disp = self.state_to_pos[int(state / self.res[1])][int(state % self.res[0])]
        # print(state, disp)
        pos = [int(corner[0] + disp[0]), int(corner[1] + disp[1])]
        # print(pos)
        if self.smap[corner[0]][corner[1]] != self.smap[pos[0]][pos[1]]:
            print("Error: invalid shield and position combination")
        else:
            # print("valid combination ")
            state = self.states[pos[0] - corner[0]][pos[1] - corner[1]]
            valid.add(state)

            for i in range(len(move)):

                map_i = np.clip(pos[0] + move[i][0], 0, self.map.shape[0] - 1)
                map_j = np.clip(pos[1] + move[i][1], 0, self.map.shape[1] - 1)

                # print(map_i, map_j)
                # print(self.map[map_i][map_j])
                if self.smap[map_i][map_j] != shield_num + 1 and state in entry_states:
                    valid.add(9)  # outside shield shield_num

                elif self.smap[map_i][map_j] != shield_num + 1:  # outside but not a valid exiting state
                    pass

                elif self.map[map_i][map_j]:  # obstacle
                    obs.add(int(self.states[map_i - corner[0]][map_j - corner[1]]))

                elif not self.map[map_i][map_j]:
                    valid.add(int(self.states[map_i - corner[0]][map_j - corner[1]]))

        return valid, obs

    def get_next_req(self, state, shield_num, entry_states, debug=False):
        valid = set()
        corner = self.coord[shield_num]
        move = [[0, -1], [1, 0], [0, 1], [-1, 0]]  # left/down/right/up
        # print(state, shield_num, entry_states)
        disp = self.state_to_pos[int(state / self.res[1])][int(state % self.res[0])]
        pos = [int(corner[0] + disp[0]), int(corner[1] + disp[1])]

        valid.add(state)

        for i in range(len(move)):

            map_i = np.clip(pos[0] + move[i][0], 0, self.map.shape[0] - 1)
            map_j = np.clip(pos[1] + move[i][1], 0, self.map.shape[1] - 1)

            if self.smap[map_i][map_j] != shield_num + 1 and state in entry_states:
                valid.add(9)  # outside shield shield_num

            elif self.smap[map_i][map_j] != shield_num + 1:  # outside but not a valid exiting state
                pass

            else:
                valid.add(int(self.states[map_i - corner[0]][map_j - corner[1]]))

        return valid

    # def find_entries(self, shield_num):
    #     corner = self.coord[shield_num]
    #     entry_states = set()
    #
    #     # check top
    #     if corner[0] > 0:
    #         for i in range(self.res[0]):
    #             if not self.map[corner[0] - 1][corner[1] + i] and not self.map[corner[0]][corner[1] + i]:
    #                 entry_states.add(i)
    #
    #     # check bottom
    #     if corner[0] + self.res[1] < self.map.shape[0]:
    #         for i in range(self.res[0]):
    #             if not self.map[corner[0] + self.res[1]][corner[1] + i] and not self.map[corner[0] + self.res[1] - 1][
    #                 corner[1] + i]:
    #                 entry_states.add(i + 6)
    #
    #     # check right
    #     r = [2, 5, 8]
    #     if corner[1] + self.res[0] < self.map.shape[1]:
    #         for i in range(self.res[1]):
    #             if not self.map[corner[0] + i][corner[1] + self.res[0]] and not self.map[corner[0] + i][
    #                 corner[1] + self.res[0] - 1]:
    #                 entry_states.add(r[i])
    #     # check left
    #     l = [0, 3, 6]
    #     if corner[1] > 0:
    #         for i in range(self.res[1]):
    #             if not self.map[corner[0] + i][corner[1] - 1] and not self.map[corner[0] + i][corner[1]]:
    #                 entry_states.add(l[i])
    #
    #     return entry_states

    def find_obstacles(self, shield_num):
        corner = self.coord[shield_num]
        obs_states = set()

        for i in range(0, self.res[1]):
            for map_j in range(0, self.res[0]):
                if self.map[corner[0] + i][corner[1] + map_j]:
                    obs_states.add(int(self.states[i][map_j]))

        return obs_states

    def process_map(self, map):
        if map == "simple":
            self.targets = [[0, 1], [0, 1], [0, 1], [0, 1]]
            self.init = [[3, 0], [3, 2], [1, 0], [1, 2]]
            self.shared_goal = True
            # self.distance = True
            # self.collision = False
        elif map == "example":
            # example map
            self.targets = [[0, 2], [2, 0]]
            if self.collision:
                self.init = [[2, 0], [2, 4]]
            elif self.distance:
                self.targets = [[0, 2], [0, 2]]
                self.init = [[2, 0], [2, 1]]

        elif map == "ISR":
            # ISR map
            self.targets = [[6, 3], [7, 0], [6, 1], [9, 3]]
            if self.collision:
                self.init = [[7, 0], [6, 3], [7, 4], [6, 1]]
            elif self.distance:
                self.targets = [[0, 5], [0, 5], [0, 5], [0, 5]]
                self.init = [[9, 2], [9, 3], [8, 3], [7, 2]]


        elif map == "MIT":
            # MIT map
            self.targets = [[3, 16], [3, 0], [0, 12], [6, 6]]
            if self.collision:
                self.init = [[3, 0], [3, 16], [6, 6], [0, 12]]
            elif self.distance:
                self.targets = [[3, 16], [3, 16], [3, 16], [3, 16]]
                self.init = [[3, 0], [3, 1], [4, 1], [1, 1]]

        elif map == "Pentagon":
            # Pentagon
            self.targets = [[4, 10], [3, 7], [3, 5], [6, 5]]
            if self.collision:
                self.init = [[3, 7], [4, 10], [6, 5], [3, 5]]
            elif self.distance:
                self.targets = [[4, 10], [4, 10], [4, 10], [4, 10]]
                self.init = [[8, 0], [8, 1], [7, 1], [8, 2]]

        elif map == "SUNY":
            # SUNY map
            self.targets = [[8, 17], [3, 22], [6, 11], [5, 19]]
            if self.collision:
                self.init = [[3, 22], [8, 17], [5, 19], [7, 14]]
            elif self.distance:
                self.targets = [[3, 22], [3, 22], [3, 22], [3, 22]]
                self.init = [[1, 1], [2, 1], [3, 1], [4, 1]]

        elif map == "SUNYvar":
            # SUNY map
            self.targets = [[7, 7], [3, 22], [6, 13], [8, 17]]
            if self.collision:
                self.init = [[3, 22], [7, 7], [8, 17], [6, 13]]
            elif self.distance:
                self.targets = [[3, 22], [3, 22], [3, 22], [3, 22]]
                self.init = [[1, 1], [2, 1], [3, 1], [4, 1]]

        else:
            # unrecognized map
            print("unrecognized map error")
            exit(1)

        self.init = np.asarray(self.init)
        self.targets = np.asarray(self.targets)
