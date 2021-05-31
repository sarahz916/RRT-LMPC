# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:19:26 2021

@author: zousa
"""
from rrt import RRT
from body import Body 
from make_demo import make_demo
import matplotlib.pyplot as plt
import numpy as np
from cubic_spline_planner import fit_path
import pdb
import environments
import math

def show_path_and_demos(path, demos: list, j: int):
    plt.plot([x for (x, y) in path], [y for (x, y) in path], 'or')
    for i in range(j):
        # plot demos
        states = demos[i][1]
        plt.plot([state[0] for state in states], [state[1] for state in states])
    plt.grid(True)
    plt.pause(0.01)  # Need for Mac
    plt.show()
    # TODO: add in legend

def main(j: int):

    # ====Search Path with RRT====
    # obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
    #                 (9, 5, 2), (8, 10, 1)]  # [x, y, radius]
    
    dt = .1
    obstacleList = environments.obstacle_list_1
    # Set Initial parameters
    body = Body(obstacleList,  start_state=(0, 0, 0, 0), end_state=(2, 15, 0, 0), max_grid = (15, 15))
    rrt = RRT(body, 1000, 50, 1, 0.01, 0.2, math.pi/2) # body, max_iter, goal_sample_rate, expand_dis, path_resolution, bubbleDist
    path = rrt.planning()
    if path is None:
        print("Cannot find path")
        return 
    else:
        print("found path!!")
        #need to create list of states for path
        fitted_path = fit_path(np.array(path), ds = 0.1)
        demos = []
        for i in range(j):
            inputs, states, f = make_demo(body, fitted_path, dt)
            if f == 1:
                demos.append([inputs, states])
        # Draw final path
        rrt.draw_graph()
        show_path_and_demos(fitted_path, demos, j)
    
    return demos

if __name__ == '__main__':
    demos = main(10)
    # for demo in demos:
    #     print(demo[1][-1] - [2, 15, 0, 0]) #notice we're quite close to desired
    #     # end state