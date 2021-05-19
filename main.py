# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:19:26 2021

@author: zousa
"""
from rrt import RRT
from body import Body 
from pid import pid
import matplotlib.pyplot as plt
import numpy as np

show_animation = True

def main(gx=6.0, gy=10.0):
    print("start " + __file__)

    # ====Search Path with RRT====
    obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                    (9, 5, 2), (8, 10, 1)]  # [x, y, radius]
    # Set Initial parameters
    body = Body(obstacleList,  start=(0, 0), end=(2, 15), max_grid = (20, 20))
    rrt = RRT(body, 1000, 50, 1, 0.01, 0.2) # body, max_iter, goal_sample_rate, expand_dis, path_resolution, bubbleDist
    path = rrt.planning()

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")
        #need to create list of states for path
        inputs, states = pid(body, path, .1)
        print(states)
        #pos = np.array(states)
        # Draw final path
        if show_animation:
            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.plot([state[0][0] for state in states], [state[0][1] for state in states], '-b')
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            plt.show()

        #pdb.set_trace()
    
    return inputs, states

if __name__ == '__main__':
    main()