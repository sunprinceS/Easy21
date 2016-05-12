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

MAX_ITER = 33000
N0 = 100

class TemporalDifferenceLearner(Agent):
    def __init__(self,lamda,N0,approx):
        super(TemporalDifferenceLearner,self).__init__(N0,approx)
        self.approx = approx

        if self.approx:
            self.E = np.zeros(self.w.shape)
        else:
            self.E = np.zeros(self.qvalue_table.shape)

        self.lamda = lamda

    def learn(self,init_state):
        if self.approx:
            return self.lstd(init_state)
        else:
            return self.sarsa(init_state)

    def build_q(self):
        if self.approx:
            for i in range(10):
                for j in range(21):
                    for k in range(2):
                        self.qvalue_table[i][j][k] = np.dot(self.w,self.get_feat((i,j),k))
        else:
            pass

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

    def lstd(self,init_state):
        """
        Linear Least Squares Prediction (v^{pi}_t use G^{lamda}_t) to computer w
        """
        eps = 0.05
        alpha = 0.01
        cur_state = init_state
        cur_action = super(TemporalDifferenceLearner,self).eps_greedy(cur_state,eps)
        reward = 0

        while True:
            approx_x = self.get_feat(cur_state,cur_action)
            cur_approx_q = np.dot(self.w,approx_x)
            self.E[approx_x == 1] += 1
            next_state,reward = Environment.step(cur_state,cur_action)

            if next_state == TERMINAL:
                delta = reward - cur_approx_q
                self.w += alpha * delta * self.E
                break
            else:
                next_action = super(TemporalDifferenceLearner,self).eps_greedy(next_state,eps)
                next_approx_q = np.dot(self.w,self.get_feat(next_state,next_action))
                delta = reward + (next_approx_q - cur_approx_q)
                self.w += alpha * delta * self.E
                self.E *= self.lamda
                cur_state,cur_action = next_state,next_action

if __name__ == "__main__":
    approx = (len(sys.argv) == 3) and (sys.argv[2] == 'approx')

    with open("MC_qtable/episode{}".format(sys.argv[1])) as q_file:
        opt_q = pickle.load(q_file)
        lamdas = 0.1 * np.array(range(11))
        # lamdas = [0,1]
        # lamdas = 0.1 * np.array(range(1,10))
        for lamda in lamdas:
            mses = []
            TD_learner = TemporalDifferenceLearner(lamda,N0,approx)

            for episode in range(MAX_ITER):
                sys.stdout.write('\rEpisode {}'.format(episode+1))
                init_state = Environment.init()
                TD_learner.learn(init_state)
                # if lamda == .0 or lamda == 1.0:
                    # TD_learner.build_q()
                    # mses.append(TD_learner.qvalue_mse(opt_q))
            TD_learner.build_q()
            print(" MSE error of Q value= {} under {}".format(TD_learner.qvalue_mse(opt_q),lamda))

            # TD_learner.show_value()

            # TD_learner.show_policy()
            # if lamda == .0 or lamda == 1.0:
                # x = range(0,MAX_ITER)
                # plt.xlabel("episode number")
                # plt.xlim([0, MAX_ITER])
                # plt.xticks(range(0, MAX_ITER + 1,MAX_ITER/10))
                # plt.ylabel("Mean-Squared Error")
                # plt.plot(x, mses)
                # plt.show()
