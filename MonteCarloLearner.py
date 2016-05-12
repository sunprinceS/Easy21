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
import pickle

import Environment
from marcos import *
from Agent import Agent


MAX_ITER = 100000

class MonteCarloLearner(Agent):
    def __init__(self,N0):
        super(MonteCarloLearner,self).__init__(N0)

    def simulate_one_episode(self,init_state):
        state_seq = []
        action_seq = []
        reward = 0

        state = init_state
        while True:
            self.Ns[state] += 1
            eps_t = self.N0/(self.N0 + self.Ns[state])
            action  = super(MonteCarloLearner,self).eps_greedy(state,eps_t)
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

    def show_value(self):
        print("\nN0 is {}".format(self.N0))
        super(MonteCarloLearner,self).show_value()

if __name__ == "__main__":
    MC_learner = MonteCarloLearner(100)
    with open("MC_qtable/episode{}".format(MAX_ITER),'w') as q_file:
        for episode in range(MAX_ITER):
            sys.stdout.write('\rEpisode {}'.format((episode+1)))
            init_state =  Environment.init()
            MC_learner.learn(init_state)
        MC_learner.show_value()
        pickle.dump(MC_learner.get_q(),q_file)
    with open("MC_qtable/episode{}".format(MAX_ITER)) as q_file:
        test = pickle.load(q_file)
