# again ... modified version of Frozen Lake v0.
import gym
from gym import error, spaces, utils
from gym import utils
from gym.envs.toy_text import discrete

import numpy as np
import sys
from six import StringIO, b

# Action
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAP = [
        "SOOOXXXX",
        "OXXOOOOX",
        "OOXOXXOX",
        "OXXOOXOO",
        "OXOOXXOX",
        "OOXXOOOO",
        "OXOOXXXO",
        "OOOOOOXG"
    ]

# MAP = [
#         "SOOO",
#         "OXXO",
#         "OOXO",
#         "OXXG",
#     ]

class Maze2dEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.t = 0

        self.desc = np.asarray(MAP,dtype='c')
        self.nrow, self.ncol = self.desc.shape
        self.reward_range = (0, 1)

        self.nA = 4
        self.nS = self.nrow * self.ncol

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)        

        # initial state distribution
        self.isd = np.array(self.desc == b'S').astype('float64').ravel()
        self.isd /= self.isd.sum()

        self.P = {s : {a : [] for a in range(self.nA)} for s in range(self.nS)}

        def to_s(row, col):
            return row*self.ncol + col
        
        def inc(row, col, a):
            if a==0: # left
                newcol = max(col-1,0)
                if self.desc[row, newcol] == b'X':
                    return (row, col)
                else:
                    col = newcol
            elif a==1: # down
                newrow = min(row+1,self.nrow-1)
                if self.desc[newrow, col] == b'X':
                    return (row, col)
                else:
                    row = newrow
            elif a==2: # right
                newcol = min(col+1,self.ncol-1)
                if self.desc[row, newcol] == b'X':
                    return (row, col)
                else:
                    col = newcol
            elif a==3: # up
                newrow = max(row-1,0)
                if self.desc[newrow, col] == b'X':
                    return (row, col)
                else:
                    row = newrow
            return (row, col)

        for row in range(self.nrow):
            for col in range(self.ncol):
                s = to_s(row, col)
                for a in range(4):
                    letter = self.desc[row, col]
                    if letter == b'X':
                        self.P[s][a] = [(1.0, s, 0, True)]
                    else:
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        newletter = self.desc[newrow, newcol]
                        done = (bytes(newletter) == b'G')
                        if newletter == b'G':
                            rew = 1.0
                        else:
                            rew = -0.1 # at every time t, the reward is -0.1. 
                        self.P[s][a] = [(1.0, newstate, rew, done)]

        # print (self.P)

    def step(self, a):
        chosen_index = np.random.choice(len(self.P[self.s][a]), 
                                        1, p=[e[0] for e in self.P[self.s][a]])[0]
        prob, s, r, d = self.P[self.s][a][chosen_index]
        self.s = s
        self.lastaction=a
        self.t += 1
        return (s, r, d, {"prob" : prob})

    def reset(self):
        self.t = 0
        self.s = np.random.choice(self.nS, p=self.isd)
        self.lastaction=None
        return self.s

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({}, t={})\n".format(["Left","Down","Right","Up"][self.lastaction], self.t))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile
