# simplified version of Frozen Lake v0.
import gym
from gym import error, spaces, utils
from gym import utils
from gym.envs.toy_text import discrete

import numpy as np
import sys
from six import StringIO, b

LEFT = 0
RIGHT = 1

class Maze1dEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = desc = np.asarray("SFFFFFFFFG",dtype='c')
        self.ncol = ncol = len(desc)
        self.reward_range = (0, 1)

        self.nA = 2
        self.nS = ncol

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.isd = np.array(desc == b'S').astype('float64').ravel() # array([1., 0., 0., 0., 0., 0., 0., 0.])
        self.isd /= self.isd.sum() # initial state distribution

        self.P = {s : {a : [] for a in range(self.nA)} for s in range(self.nS)}

        def inc(col, a):
            if a==0: # left
                col = max(col-1,0)
            elif a==1: # right
                col = min(col+1,ncol-1)
            return col

        for col in range(ncol):
            s = col
            for a in range(2):
                letter = desc[col]
                if letter == b'G':
                    self.P[s][a] = (s, 0, True)
                else:
                    newcol = inc(col, a)
                    newstate = newcol
                    newletter = desc[newcol]
                    done = bytes(newletter) == b'G'
                    rew = float(newletter == b'G')
                    self.P[s][a] = (newstate, rew, done)

    def step(self, a):
        s, r, d = self.P[self.s][a]
        self.s = s
        self.lastaction=a
        return (s, r, d, None)

    def reset(self):
        self.s = np.random.choice(self.nS, p=self.isd)
        self.lastaction=None
        return self.s

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        col = self.s % self.ncol
        desc = self.desc.tolist()
        desc = [c.decode('utf-8') for c in desc]
        desc[col] = utils.colorize(desc[col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Right"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write(''.join(desc)+"\n")

        if mode != 'human':
            return outfile
