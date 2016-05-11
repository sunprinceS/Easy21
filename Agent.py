"""
File: Agent.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Github: https://github.com/sunprinces
Description: Abstract class for various control method
"""

import numpy as np
import sys
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class Agent(object):
    def __init__(self):
        self.qvalue_table = np.zeros((10,21,2))

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
        plt.show()
