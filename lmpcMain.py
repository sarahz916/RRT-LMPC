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
    plt.close('all')
    # ====Search Path with RRT====
    obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                    (9, 5, 2), (8, 10, 1)]  # [x, y, radius]
    # Set Initial parameters
    body = Body(obstacleList, start_state=(0, 0, 0, 0), end_state=(2, 15, 0, 0), max_grid = (20, 20))
    
    ### User-defined constants for LMPC ###
    goal = np.array(body.end)
    N  = 20
    K = 100
    n = 4
    d = 2
    Q  = 1*np.eye(n)
    R  = 0.1*np.eye(d)
    Qf = 1000*np.eye(n)
    regQ = 100*np.eye(n)
    regR = 100*np.eye(n)
    dt = 0.05
    printLevel = 2
    width = 1
    numDemos = 10
    amax = body.max_acc
    amin = -body.max_acc
    theta_dotMax = body.max_theta_dot
    theta_dotMin = -body.max_theta_dot
    path_length = 100    
    ######

    create = True
    
    if create:
        # body, max_iter, goal_sample_rate, expand_dis, path_resolution, width,
        # tightest branch turn angle, how many skips to allow 
        # (1 = no skips, 2 = 1 skip, etc.)
        rrt = RRT(body, 2000, 50, 1, 0.01, width, math.pi/2, 3)

        path = rrt.planning()
                
        if path is None:
            print("Cannot find path")
            return 
        else:
            print("found path!!")
            
         # Create the spline
        path = np.array(path)
        
    else:
        path = np.load('path.npy')
                
    spline = Spline2D(path[:,0], path[:,1], ds=0.001)
    
    # Need to create list of states for path
    fitted_path = fit_path(path, ds = 0.01)
    demos = []
    for i in range(j):
        inputs, states, f = make_demo(body, fitted_path, dt, path_length)
        if f == 1:
            demos.append([inputs, states])
    
    if create:
        # Draw final path
        rrt.draw_graph()
        show_path_and_demos(path, demos, j)
        
    lmpcSolver = LMPC(N, K, Q, Qf, R, regQ, regR, [], [], spline, dt, width, amax, amin, theta_dotMax, theta_dotMin, printLevel)    
    
    # plt.figure()
           
    # splineLine = []
    # for s in np.linspace(0, spline.end, 1000, endpoint=False):
    #     splineLine.append(spline.calc_position(s))
    # splineLine = np.array(splineLine)

    # plt.plot(splineLine[:,0], splineLine[:,1], '--oy', label='Spline')
    # plt.legend()        
    
    # for i in range(numDemos):
    #     xDemo, uDemo = lmpcSolver.createDemo([1, 0.2, 0.2, 0], 1)
        
    #     xyCoords = lmpcSolver.convertToXY(xDemo)
    #     plt.plot(xyCoords[:,0], xyCoords[:,1], '--o', label='Demo ' + str(i))
    
    #     pdb.set_trace()
    
    # Add all the trajectories
    for k, demo in enumerate(demos):
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
        uTraj = np.copy(demo[0])
        lmpcSolver.updateSSandValueFunction(xTraj, uTraj)
        
        if k == 0:
            # Compute the expected path using x-parameterized nonlinear dynamics
            xExp = [demo[1][0]]
            for i, ut in enumerate(uTraj):
                xExp.append(xDynamics(xExp[-1], ut[0], ut[1], dt))
            xExp = np.array(xExp)
            
            # Compute the expected path using s-parameterized nonlinear dynamics
            x0 = np.copy(demo[1][0])
            x0[0], x0[1] = spline.calcSY(x0[0], x0[1])
            xSExp = [x0]
            for i, ut in enumerate(uTraj):
                # print('i = ' + str(i))
                # print('ut = ', ut)
                try:
                    xNext = np.array(lmpcSolver.dynamics(xSExp[-1], ut))
                    temp = np.copy(xNext)
                    temp[0], temp[1] = spline.calcXY(temp[0], temp[1])
                    xSExp.append(xNext)
                    # if i < len(uTraj)-1 and np.linalg.norm(temp - xExp[i+1]) > 1e-2:
                    #       pdb.set_trace()
                except:
                    print('Error')
                    pdb.set_trace()
            xSExp = lmpcSolver.convertToXY(xSExp)
            
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
            plt.plot(xExp[:,0], xExp[:,1], '--or', label='Predicted from control x')
            plt.plot(xSExp[:,0], xSExp[:,1], '--ok', label='Predicted from control s')
            plt.plot(splineLine[:,0], splineLine[:,1], '--oy', label='Spline')
            plt.legend()
            
            # plt.figure()
            # plt.plot(np.array(xTraj)[:,1])
            # plt.title('y component in s-space')
            # plt.xlabel('Iteration')
            # plt.ylabel('y')
            # plt.figure()
            # plt.plot(demo[0][:,1])
            # plt.title('ThetaDot against Iterations')
            # plt.xlabel('Iteration')
            # plt.ylabel('ThetaDot')
    
            print('Finished')
            # pdb.set_trace()
    
    # Pass in the N'th point of the last demonstrated trajectory
    costs = []
    for i in range(10):
        xTraj, uTraj = lmpcSolver.runTrajectory(xTraj, uTraj)
        
        print('Trajectory ' + str(i) + ' Completed!')
        
        # Visualize the resulting trajectory
        
        xyTraj = lmpcSolver.convertToXY(xTraj)
        
        plt.figure()
        plt.title('Closed-Loop Trajectory: Iter = ' + str(i))
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.plot(xyTraj[:,0], xyTraj[:,1], '--ob', label='Closed-Loop')
        plt.plot(splineLine[:,0], splineLine[:,1], '--oy', label='Spline')
        plt.legend()
        
        costs.append(lmpcSolver.updateSSandValueFunction(xTraj[1:], uTraj))
    return costs

if __name__ == '__main__':
    costs = main(10)