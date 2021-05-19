# -*- coding: utf-8 -*-
"""
Created on Sat May 15 21:25:54 2021

@author: zousa
"""
# Body Class for CS 159 Final Project

import numpy as np
import math

class Body():
    def __init__(self, obs_list: list, start: tuple, end: tuple, max_grid: tuple,
                 max_velocity = 10, heading_ang = math.pi/2):
        """
        
        Parameters
        ----------
        obs_list : list
            List of obstacles
        start : tuple
            Start coordinate
        end : tuple
            End coordinate
        max_grid : tuple
            Maximium x value and Maximum y value of workspace

        Returns
        -------
        None.

        """
        self.obs_list = obs_list
        self.start = start
        self.end = end
        self.max_x = max_grid[0]
        self.max_y = max_grid[1]
        #should we have a velocity range
        self.max_velocity = max_velocity
        self.heading_ang = heading_ang
