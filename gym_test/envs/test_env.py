# simplified version of Frozen Lake v0.
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym import utils
from gym.envs.toy_text import discrete

import numpy as np
import sys
from six import StringIO, b

LEFT = 0
RIGHT = 1

class TestEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        desc = "SFFFFFFFFG"
        self.desc = desc = np.asarray(desc,dtype='c')
        self.ncol = ncol = len(desc)
        self.reward_range = (0, 1)

        nA = 2
        nS = ncol

        isd = np.array(desc == b'S').astype('float64').ravel() # array([1., 0., 0., 0., 0., 0., 0., 0.])
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def inc(col, a):
            if a==0: # left
                col = max(col-1,0)
            elif a==1: # right
                col = min(col+1,ncol-1)
            return col

        for col in range(ncol):
            s = col
            for a in range(2):
                li = P[s][a]
                letter = desc[col]
                if letter == b'G':
                    li.append((1.0, s, 0, True))
                else:
                    newcol = inc(col, a)
                    newstate = newcol
                    newletter = desc[newcol]
                    done = bytes(newletter) == b'G'
                    rew = float(newletter == b'G')
                    li.append((1.0, newstate, rew, done))

        super(TestEnv, self).__init__(nS, nA, P, isd)

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
