import sys
sys.path.insert(0, './make_demos')
from rrt import RRT
from body import Body 
from make_demo import make_demo
import matplotlib.pyplot as plt
import numpy as np
from cubic_spline_planner import fit_path, Spline2D
from lmpc import LMPC
import pdb

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
    body = Body(obstacleList,  start_state=(0, 0, 0, 0), end_state=(2, 15, 0, 0), max_grid = (20, 20))
    
    
    ### User-defined constants for LMPC ###
    goal = np.array(body.end)
    N  = 40
    K = 30
    n = 4
    d = 2
    Q  = 1*np.eye(n)
    R  = 0.01*np.eye(d)
    Qf = 1000*np.eye(n)
    dt = 0.1
    printLevel = 2
    width = 0.2
    amax = body.max_acc
    amin = -body.max_acc
    theta_dotMax = body.max_theta_dot
    theta_dotMin = -body.max_theta_dot    
    ######

    rrt = RRT(body, 1000, 50, 1, 0.01, width) # body, max_iter, goal_sample_rate, expand_dis, path_resolution, width
    
    path = rrt.planning()
    if path is None:
        print("Cannot find path")
        return 
    else:
        print("found path!!")
        
        # Create the spline
        path = np.array(path)
        spline = Spline2D(path[:,0], path[:,1])
                
        # Need to create list of states for path
        fitted_path = fit_path(path, ds = 0.1)
        demos = []
        for i in range(j):
            inputs, states, f = make_demo(body, fitted_path, .1)
            if f == 1:
                demos.append([inputs, states])
        # Draw final path
        rrt.draw_graph()
        show_path_and_demos(path, demos, j)
    
    lmpcSolver = LMPC(N, K, Q, Qf, R, [], [], spline, dt, width, amax, amin, theta_dotMax, theta_dotMin, printLevel)    
    
    # Add all the trajectories
    for demo in demos:
        # Need to convert from x,y to s,y representation
        xTraj = demo[1][1:] # xTraj right now includes x(0) which remove
        for i,x in enumerate(xTraj):
            xTraj[i][:2] = spline.calcSY(xTraj[i][0], xTraj[i][1])
            if xTraj[i][0] < 0:
                pdb.set_trace()
        uTraj = demo[0]
        lmpcSolver.updateSSandValueFunction(xTraj, uTraj)
        
    # Pass in the N'th point of the last demonstrated trajectory
    xTraj, uTraj = lmpcSolver.runTrajectory(xTraj[N])
    
    return demos

if __name__ == '__main__':
    demos = main(10)