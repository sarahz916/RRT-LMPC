# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:26:42 2021

@author: zousa
"""
import matplotlib.pyplot as plt
import math
import numpy as np

# File of different obstacles and corresponding bodies
def plot_circle(x, y, size, color="-b"):  # pragma: no cover
    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
    yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
    plt.plot(xl, yl, color)

obstacle_list_1 = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                (9, 5, 2), (8, 10, 1)]  # [x, y, radius]

obstacle_list_2 = [(3, 8, 2), (3, 10, 2), (7, 5, 2),
                (9, 5, 2), (8, 10, 1), (12, 12, 5)]  # [x, y, radius]

obstacle_list = obstacle_list_2

for (ox, oy, size) in obstacle_list:
    plot_circle(ox, oy, size)

plt.axis([0, 15, 0, 15])

