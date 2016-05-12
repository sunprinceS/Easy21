"""
File: TemporalDifferenceLearner.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Github: https://github.com/sunprinces
Description: Temporal Difference Control for question3 and 4
"""

import numpy as np
import sys
import random
import matplotlib.pyplot as plt
import pickle

import Environment
from marcos import *
from Agent import Agent

MAX_ITER = 100000
N0 = 100

class TemporalDifferenceLearner(Agent):
    def __init__(self,lamda,N0):
        super(TemporalDifferenceLearner,self).__init__(N0)
        self.E = np.zeros(self.qvalue_table.shape)
        self.lamda = lamda

    def sarsa(self,init_state):
        """
        Backward Sarsa
        """
        cur_state = init_state
        self.Ns[cur_state] += 1
        eps_t = self.N0/(self.N0 + self.Ns[cur_state])
        cur_action = super(TemporalDifferenceLearner,self).eps_greedy(cur_state,eps_t)
        reward = 0

        while True:
            cur_idx = cur_state+(cur_action,)
            self.E[cur_idx] += 1
            self.Nsa[cur_idx] += 1
            alpha_t = 1.0/self.Nsa[cur_idx]

            next_state,reward = Environment.step(cur_state,cur_action)
            if next_state == TERMINAL:
                delta = reward - self.qvalue_table[cur_idx]
                self.qvalue_table[cur_idx] += alpha_t*delta*self.E[cur_idx]
                break
            else:
                self.Ns[next_state] += 1
                eps_t = self.N0/(self.N0 + self.Ns[next_state])
                next_action = super(TemporalDifferenceLearner,self).eps_greedy(next_state,eps_t)
                next_idx = next_state + (next_action,)
                delta = reward + (self.qvalue_table[next_idx] - self.qvalue_table[cur_idx])
                self.qvalue_table[cur_idx] += alpha_t*delta*self.E[cur_idx]
                self.E *= self.lamda
                cur_state,cur_action = next_state,next_action


if __name__ == "__main__":
    with open("MC_qtable/episode{}".format(sys.argv[1])) as q_file:
        opt_q = pickle.load(q_file)
        lamdas = 0.1 * np.array(range(11))
        for lamda in lamdas:
            mses = []
            TD_learner = TemporalDifferenceLearner(lamda,N0)

            for episode in range(MAX_ITER):
                sys.stdout.write('\rEpisode {}'.format(episode+1))
                init_state = Environment.init()
                TD_learner.sarsa(init_state)
                if lamda == .0 or lamda == 1.0:
                    mses.append(TD_learner.qvalue_mse(opt_q))
            print(" MSE error of Q value= {} under {}".format(TD_learner.qvalue_mse(opt_q),lamda))
            TD_learner.show_value()
            if lamda == .0 or lamda == 1.0:
                x = range(0,MAX_ITER)
                plt.title('Learning curve of MSE under lambda {} '.format(lamda))
                plt.xlabel("episode number")
                plt.xlim([0, MAX_ITER])
                plt.xticks(range(0, MAX_ITER + 1,10000))
                plt.ylabel("Mean-Squared Error")
                plt.plot(x, mses)
                plt.show()
