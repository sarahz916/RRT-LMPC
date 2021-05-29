import sys
sys.path.insert(0, './make_demos')
from rrt import RRT
from body import Body 
from make_demo import make_demo
import matplotlib.pyplot as plt
import numpy as np
from cubic_spline_planner import fit_path, Spline2D
from lmpc import LMPC
import math
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
    
def xDynamics(curr_state, acc, theta_dot, dt):
    new_ang = theta_dot * dt + curr_state[3]
    new_vel = acc*dt + curr_state[2]
    new_x = curr_state[2] * dt * math.cos(curr_state[3]) + curr_state[0]
    new_y = curr_state[2] * dt * math.sin(curr_state[3]) + curr_state[1]
    
    return [new_x, new_y, new_vel, new_ang]

def main(j: int):
    # ====Search Path with RRT====
    obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                    (9, 5, 2), (8, 10, 1)]  # [x, y, radius]
    # Set Initial parameters
    body = Body(obstacleList, start_state=(0, 0, 0, 0), end_state=(2, 15, 0, 0), max_grid = (20, 20))
    
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
        xTraj = np.copy(demo[1][1:]) # xTraj right now includes x(0) which remove
        for i,x in enumerate(xTraj):
            xTraj[i][:2] = spline.calcSY(xTraj[i][0], xTraj[i][1])
            if xTraj[i][0] < 0:
                pdb.set_trace()
            # origX = demo[1][1:][i][:2]
            # backAgainX = np.array(spline.calcXY(xTraj[i][0], xTraj[i][1])) 
            # if np.linalg.norm(origX - backAgainX) > 0.1:
            #     pdb.set_trace()
        uTraj = demo[0]
        lmpcSolver.updateSSandValueFunction(xTraj, uTraj)
        
        # Compute the expected path using x-parameterized nonlinear dynamics
        xExp = [demo[1][0]]
        for i, ut in enumerate(uTraj):
            xExp.append(xDynamics(xExp[-1], ut[0], ut[1], dt))
        xExp = np.array(xExp)
        
        # Compute the expected path using s-parameterized nonlinear dynamics
        
        # Let's visualize the demos in both x and s representations and make sure
        # they agree            
        xyCoords = lmpcSolver.convertToXY(xTraj)
        
        splineLine = []
        for s in np.linspace(0, spline.end, 1000, endpoint=False):
            splineLine.append(spline.calc_position(s))
        splineLine = np.array(splineLine)

        plt.figure()
        demoX = demo[1][1:]
        plt.plot(demoX[:,0], demoX[:,1], '--og', label='Original Demo')
        plt.plot(xyCoords[:,0], xyCoords[:,1], '--ob', label='Converted to S and Back')
        # plt.plot(xExp[:,0], xExp[:,1], '--or', label='Predicted from uDemo')
        plt.plot(splineLine[:,0], splineLine[:,1], '--oy', label='Spline')
        plt.legend()
        
        plt.figure()
        plt.plot(np.array(xTraj)[:,1])
        plt.title('y component in s-space')
        plt.xlabel('Iteration')
        plt.ylabel('y')
        plt.figure()
        plt.plot(demo[0][:,1])
        plt.title('ThetaDot against Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('ThetaDot')
    
    # Pass in the N'th point of the last demonstrated trajectory
    xTraj, uTraj = lmpcSolver.runTrajectory(xTraj, uTraj)
    
    return demos

if __name__ == '__main__':
    demos = main(10)