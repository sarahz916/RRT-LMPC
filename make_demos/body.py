# -*- coding: utf-8 -*-
"""
Created on Sat May 15 21:25:54 2021

@author: zousa
"""
# Body Class for CS 159 Final Project

import numpy as np
import math

class Body():
    def __init__(self, obs_list: list, start_state: tuple, end_state: tuple, max_grid: tuple,
                 max_acc = 10, max_theta_dot = 10 * math.pi):
        '''
        

        Parameters
        ----------
        obs_list : list
            assumes obstacles are circular (x, y, radius)
        start_state : tuple
            (x, y, v, theta)
        end_state : tuple
            (x, y, v, theta)
        max_grid : tuple
            assume workspace is [0, max_x] x [0, max_y]
            (max_x, max_y)
        max_acc : TYPE, optional
            The max acceleration in unit/sec^2. The default is 10.
        max_theta_dot : TYPE, optional
            The max theta_do in rad/sec^2. The default is 10 * math.pi.

        Returns
        -------
        None.

        '''
        self.obs_list = obs_list
        self.start = start_state
        self.end = end_state
        self.max_x = max_grid[0]
        self.max_y = max_grid[1]
        self.max_acc = max_acc
        self.max_theta_dot = max_theta_dot
