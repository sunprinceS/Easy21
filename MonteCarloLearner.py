"""
File: MonteCarloLearner.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Github: https://github.com/sunprinces
Description: Monte Carlo Control for question2
"""

import numpy as np
import sys
import random

import Environment
from marcos import *
from Agent import Agent


MAX_ITER = 10000000

class MonteCarloLearner(Agent):
    def __init__(self,N0):
        super(MonteCarloLearner,self).__init__()
        self.N0 = N0
        self.Ns = np.zeros((10,21))
        self.Nsa = np.zeros(self.qvalue_table.shape)

    def simulate_one_episode(self,init_state):
        state_seq = []
        action_seq = []
        reward = 0

        state = init_state
        while True:
            self.Ns[state] += 1
            eps_t = self.N0/(self.N0 + self.Ns[state])
            action  = self.eps_greedy(state,eps_t)
            state_seq.append(state)
            action_seq.append(action)
            state,reward = Environment.step(state,action)

            if state == TERMINAL:
                return state_seq,action_seq,reward

    def learn(self,init_state):
        state_seq,action_seq,reward = self.simulate_one_episode(init_state)
        for state,action in zip(state_seq,action_seq):
            idx = state+(action,)
            self.Nsa[idx] += 1
            alpha_t = 1.0/self.Nsa[idx]
            self.qvalue_table[idx] += alpha_t * (reward - self.qvalue_table[idx])
        return reward

    def eps_greedy(self,state,eps):
        if random.uniform(0,1) <= eps:
            return random.randint(0,1)
        else:
            return np.argmax(self.qvalue_table,axis=2)[(state)]

    def show_value(self):
        print("\nN0 is {}".format(self.N0))
        super(MonteCarloLearner,self).show_value()

if __name__ == "__main__":
    MC_learner = MonteCarloLearner(100)
    reward_record = []
    for episode in range(MAX_ITER):
        sys.stdout.write('\r{}'.format((episode+1)))
        init_state =  Environment.init()
        reward_record.append(MC_learner.learn(init_state))
    MC_learner.show_value()
