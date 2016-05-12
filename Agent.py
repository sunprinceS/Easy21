"""
File: Agent.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Github: https://github.com/sunprinces
Description: Abstract class for various control method
"""

import numpy as np
import random
import sys
import pickle
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from marcos import *

class Agent(object):
    def __init__(self,N0,approx=False):
        self.qvalue_table = np.zeros((10,21,2))
        self.approx = approx

        #for linear approximation
        if self.approx:
            self.w = np.random.randn(3*6*2)
            self.d_boundary = [[0,3],[3,6],[6,9]]
            self.p_boundary = [[0,5],[3,8],[6,11],[9,14],[12,17],[15,20]]
        else:
            self.N0 = N0
            self.Ns = np.zeros((10,21))
            self.Nsa = np.zeros(self.qvalue_table.shape)

    def get_q(self):
        return self.qvalue_table

    def get_feat(self,state,action):
        feat=[]
        dealer_sum,player_sum = state
        for d in self.d_boundary:
            for p in self.p_boundary:
                if (p[0] <= player_sum <= p[1]) and (d[0] <= dealer_sum <= d[1]):
                    feat.append(1)
                else:
                    feat.append(0)

        if action == HIT:
            feat = feat + [0]*18
        else:
            feat = [0]*18 + feat
        return np.array(feat)

    def qvalue_mse(self,optimal_qvalue_table):
        err_table = optimal_qvalue_table - self.qvalue_table
        err_table = np.power(err_table,2)
        return np.sum(err_table)

    def eps_greedy(self,state,eps):
        if random.uniform(0,1) <= eps:
            return random.randint(0,1)
        else:
            if self.approx:
                if np.dot(self.w,self.get_feat(state,HIT)) > \
                        np.dot(self.w,self.get_feat(state,STICK)):
                    return HIT
                else:
                    return STICK

            else:
                return np.argmax(self.qvalue_table,axis=2)[(state)]

    def show_value(self):

        # plot Value function v
        optimal_v_table = np.max(self.qvalue_table, axis=2)
        optimal_v_table = optimal_v_table[:,range(11,21)]

        fig = plt.figure(1)
        ax = fig.gca(projection='3d')
        X = np.arange(1, 11)
        Y = np.arange(12, 22)
        X, Y = np.meshgrid(X, Y)
        surf = ax.plot_surface(X, Y, optimal_v_table.T, rstride=1, cstride=1, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
        ax.set_xlim(1, 10)
        ax.set_xlabel('Dealer showing')
        ax.set_ylim(12, 21)
        ax.set_ylabel('Player sum')
        ax.set_zlim(-1, 1)
        ax.zaxis.set_major_locator(LinearLocator(10))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.title('V*(state)')
        plt.show()

    def show_policy(self):

        # plot Policy pi
        fig = plt.figure(2)
        X = np.arange(1, 11)
        Y = np.arange(1, 22)
        X, Y = np.meshgrid(X, Y)
        pi = np.argmax(self.qvalue_table, axis=2)
        CS = plt.contourf(X, Y, pi.T, [0, 0.5, 1], cmap=cm.bone, origin='lower')
        cbar = plt.colorbar(CS)
        plt.title('Policy')
        plt.show()
