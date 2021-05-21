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


def show_path_and_demos(path, demos, j):
    plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
    for i in range(j):
        # plot demos
        states = demos[i][1]
        plt.plot([state[0] for state in states], [state[1] for state in states])
    plt.grid(True)
    plt.pause(0.01)  # Need for Mac
    plt.show()

def main(j: int):

    # ====Search Path with RRT====
    obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                    (9, 5, 2), (8, 10, 1)]  # [x, y, radius]
    # Set Initial parameters
    body = Body(obstacleList,  start=(0, 0), end=(2, 15), max_grid = (20, 20))
    rrt = RRT(body, 1000, 50, 1, 0.01, 0.2) # body, max_iter, goal_sample_rate, expand_dis, path_resolution, bubbleDist
    path = rrt.planning()

    if path is None:
        print("Cannot find path")
        return 
    else:
        print("found path!!")
        #need to create list of states for path
        demos = []
        for i in range(j):
            inputs, states = make_demo(body, path, .01)
            demos.append([inputs, states])
        # Draw final path
        rrt.draw_graph()
        show_path_and_demos(path, demos, j)
    
    return demos

if __name__ == '__main__':
    main(100)