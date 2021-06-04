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
from environments import plot_circle
import os
import pickle

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

def main():
    plt.close('all')
    # ====Search Path with RRT====
    obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                    (9, 5, 2), (8, 10, 1)]  # [x, y, radius]
    # Set Initial parameters
    body = Body(obstacleList, start_state=(0, 0, 0, 0), end_state=(2, 15, 0, 0), max_grid = (20, 20))
    
    ### User-defined constants for LMPC ###
    goal = np.array(body.end)
    N  = 6
    K = 200
    n = 4
    d = 2
    Q  = 1 * np.eye(n)
    R  = 1e-2 * np.eye(d)
    Qf = 1 * np.eye(n) # Qf should just use Q
    regQ = 150 * np.eye(n) # I should try to decrease this if possible
    regR = 150 * np.eye(n)
    dt = 0.05
    printLevel = 2
    width = 1.5 # Was 0.3
    numDemos = 10
    amax = body.max_acc
    amin = -body.max_acc
    theta_dotMax = body.max_theta_dot
    theta_dotMin = -body.max_theta_dot
    path_length = 50
    radius = 0.5
    numTrajectories = 1
    slackCost = 1e5 * np.eye(n)
    # memory = np.inf
    ######

    create = True
    
    if create:
        # body, max_iter, goal_sample_rate, expand_dis, path_resolution, width,
        # tightest branch turn angle, how many skips to allow 
        # (1 = no skips, 2 = 1 skip, etc.)
        # Use a multiplying factor here since also have spline coming
        rrt = RRT(body, 10000, 50, 1, 0.01, 1.2 * width, math.pi/2, 3)

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
                
    spline = Spline2D(path[:,0], path[:,1], ds=1e-3)
    
    print('---> The maximum curvature along the spline is: ')
    maxK = max([spline.calc_curvature(s) for s in np.linspace(0, spline.end, 1000, endpoint=False)])
    print(maxK)
    
    # Need to create list of states for path
    fitted_path = fit_path(path, ds = 0.01)
    demos = []
    for i in range(numDemos):
        inputs, states, f = make_demo(body, fitted_path, dt, path_length)
        if f == 1:
            demos.append([inputs, states])
    
    if create:
        # Draw final path
        rrt.draw_graph()
        show_path_and_demos(path, demos, numDemos)
        
    lmpcSolver = LMPC(N, K, Q, Qf, R, regQ, regR, [], [], spline, dt, width, \
                      amax, amin, theta_dotMax, theta_dotMin, radius, printLevel, slackCost) # , memory)    
    
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
        xTraj = np.copy(demo[1])
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
        
        # if k == 0:
        #     # Compute the expected path using x-parameterized nonlinear dynamics
        #     xExp = [demo[1][0]]
        #     for i, ut in enumerate(uTraj):
        #         xExp.append(xDynamics(xExp[-1], ut[0], ut[1], dt))
        #     xExp = np.array(xExp)
            
        #     # Compute the expected path using s-parameterized nonlinear dynamics
        #     x0 = np.copy(demo[1][0])
        #     x0[0], x0[1] = spline.calcSY(x0[0], x0[1])
        #     xSExp = [x0]
        #     for i, ut in enumerate(uTraj):
        #         # print('i = ' + str(i))
        #         # print('ut = ', ut)
        #         try:
        #             xNext = np.array(lmpcSolver.dynamics(xSExp[-1], ut))
        #             temp = np.copy(xNext)
        #             temp[0], temp[1] = spline.calcXY(temp[0], temp[1])
        #             xSExp.append(xNext)
        #             # if i < len(uTraj)-1 and np.linalg.norm(temp - xExp[i+1]) > 1e-2:
        #             #       pdb.set_trace()
        #         except:
        #             print('Error')
        #             pdb.set_trace()
        #     xSExp = lmpcSolver.convertToXY(xSExp)
            
        #     # Let's visualize the demos in both x and s representations and make sure
        #     # they agree            
        #     xyCoords = lmpcSolver.convertToXY(xTraj)
            
        #     splineLine = []
        #     for s in np.linspace(0, spline.end, 1000, endpoint=False):
        #         splineLine.append(spline.calc_position(s))
        #     splineLine = np.array(splineLine)
    
            # plt.figure()
            # demoX = demo[1]
            # plt.plot(demoX[:,0], demoX[:,1], '--og', label='Original Demo')
            # plt.plot(xyCoords[:,0], xyCoords[:,1], '--ob', label='Converted to S and Back')
            # plt.plot(xExp[:,0], xExp[:,1], '--or', label='Predicted from control x')
            # plt.plot(xSExp[:,0], xSExp[:,1], '--ok', label='Predicted from control s')
            # plt.plot(splineLine[:,0], splineLine[:,1], '--oy', label='Spline')
            # plt.legend()
            
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
    
            # print('Finished')
            # pdb.set_trace()
    
    # Make a new folder to save the results
    suffix = 0
    while True:
        foldername = 'Results/results' + str(suffix)
        if not os.path.isdir(foldername):
            os.mkdir(foldername)
            break
        suffix += 1
    
    # Pass in the N'th point of the last demonstrated trajectory
    costs = []
    maxVelocities = []
    
    trajectories = []
    
    for i in range(numTrajectories):
        xTraj, uTraj, xTrajList, terminalPointsList, counts, mismatchFirst, mismatchLast, \
        openLoopCosts, slacks = lmpcSolver.runTrajectory(xTraj, uTraj)
        
        maxVelocities.append(max(np.array(xTraj)[:,2]))
        trajectories.append(xTraj)
        
        # Save the results
        np.save(foldername + '/xTraj' + str(i), xTraj)
        np.save(foldername + '/uTraj' + str(i), uTraj)
        np.save(foldername + '/xTrajList' + str(i), xTrajList)
        np.save(foldername + '/terminalPointsList' + str(i), terminalPointsList)
        np.save(foldername + '/counts' + str(i), counts)
        np.save(foldername + '/mismatchFirst' + str(i), mismatchFirst)
        np.save(foldername + '/mismatchLast' + str(i), mismatchLast)
        np.save(foldername + '/openLoopCosts' + str(i), openLoopCosts)
        
        print('Trajectory ' + str(i) + ' Completed!')
                
        # Visualize the resulting trajectory
        fig, axes = visualizeElapsedTrajectory(spline, xTrajList, terminalPointsList, counts)        
        fig.suptitle('Closed-Loop Trajectory: Iter = ' + str(i))
        plt.pause(0.001)
        fig.savefig(foldername + '/elapsed' + str(i) + '.png')
        plt.close()

        # xyTraj = lmpcSolver.convertToXY(xTraj)
        
        # plt.figure()
        # plt.title('Closed-Loop Trajectory: Iter = ' + str(i))
        # plt.xlabel(r'$x_1$')
        # plt.ylabel(r'$x_2$')
        # plt.plot(xyTraj[:,0], xyTraj[:,1], '--ob', label='Closed-Loop')
        # plt.plot(splineLine[:,0], splineLine[:,1], '--oy', label='Spline')
        # plt.legend()
        
        # Visualize the open-loop costs
        plt.figure(figsize=(8,6))
        plt.plot(range(len(openLoopCosts)), openLoopCosts)
        plt.xlabel('Trajectory Steps', fontsize=14)
        plt.ylabel('Open-Loop Cost (Constant Terms Dropped)', fontsize=14)
        plt.title('Open-Loop Costs across Trajectory: Iter = ' + str(i), fontsize=14)
        plt.pause(0.001)
        plt.savefig(foldername + '/openLoopCosts' + str(i) + '.png')
        plt.close()
        
        # Visualize the slack norm
        plt.figure(figsize=(8,6))
        plt.plot(range(len(slacks)), slacks)
        plt.xlabel('Trajectory Steps', fontsize=14)
        plt.ylabel('Slack norm', fontsize=14)
        plt.title('Slacks across Trajectory: Iter = ' + str(i), fontsize=14)
        plt.pause(0.001)
        plt.savefig(foldername + '/slack' + str(i) + '.png')
        plt.close()
        
        # Visualize the model mismatch due to linearization
        
        # Against steps
        plt.figure(figsize=(8,6))
        plt.title('Linearization-Induced Model Mismatch', fontsize=14)
        plt.ylabel(r'|Predicted $x_{t+1}$ - Actual $x_{t+1}$|', fontsize=14)
        plt.xlabel('Trajectory Steps', fontsize=14)
        plt.plot(np.array(mismatchFirst))
        plt.legend([r'$s$',r'$e_y$', r'$v$', r'$\theta$'])
        plt.pause(0.001)
        plt.savefig(foldername + '/mismatchFirst' + str(i) + '.png')
        plt.close()
        
        plt.figure(figsize=(8,6))
        plt.title('Linearization-Induced Model Mismatch', fontsize=14)
        plt.ylabel(r'|Predicted $x_{t+N}$ - Actual $x_{t+N}$|', fontsize=14)
        plt.xlabel('Trajectory Steps', fontsize=14)
        plt.plot(np.array(mismatchLast))
        plt.legend([r'$s$',r'$e_y$', r'$v$', r'$\theta$'])
        plt.pause(0.001)
        plt.savefig(foldername + '/mismatchLast' + str(i) + '.png')
        plt.close()
        
        # Against velocity
        plt.figure(figsize=(8,6))
        plt.title('Linearization-Induced Model Mismatch', fontsize=14)
        plt.ylabel(r'|Predicted $x_{t+1}$ - Actual $x_{t+1}$|', fontsize=14)
        plt.xlabel('Current Velocity', fontsize=14)
        plt.scatter(np.array(xTraj[:-1])[:,2], np.array(mismatchFirst)[:,0])
        plt.scatter(np.array(xTraj[:-1])[:,2], np.array(mismatchFirst)[:,1])
        # plt.scatter(np.array(xTraj[:-1])[:,2], np.array(mismatchFirst)[:,2])
        # plt.scatter(np.array(xTraj[:-1])[:,2], np.array(mismatchFirst)[:,3])
        plt.legend([r'$s$',r'$e_y$'])
        plt.pause(0.001)
        plt.savefig(foldername + '/mismatchFirstVel' + str(i) + '.png')
        plt.close()
        
        plt.figure(figsize=(8,6))
        plt.title('Linearization-Induced Model Mismatch', fontsize=14)
        plt.ylabel(r'|Predicted $x_{t+N}$ - Actual $x_{t+N}$|', fontsize=14)
        plt.xlabel('Current Velocity', fontsize=14)
        plt.scatter(np.array(xTraj[:-1])[:,2], np.array(mismatchLast)[:,0])
        plt.scatter(np.array(xTraj[:-1])[:,2], np.array(mismatchLast)[:,1])
        # plt.scatter(np.array(xTraj[:-1])[:,2], np.array(mismatchLast)[:,2])
        # plt.scatter(np.array(xTraj[:-1])[:,2], np.array(mismatchLast)[:,3])
        plt.legend([r'$s$',r'$e_y$'])
        plt.pause(0.001)
        plt.savefig(foldername + '/mismatchLastVel' + str(i) + '.png')
        plt.close()
        
        # Against curvature
        curvatures = [spline.calc_curvature(s) for s in np.array(xTraj[:-1])[:,0]]
        plt.figure(figsize=(8,6))
        plt.title('Linearization-Induced Model Mismatch', fontsize=14)
        plt.ylabel(r'|Predicted $x_{t+1}$ - Actual $x_{t+1}$|', fontsize=14)
        plt.xlabel('Current Curvature', fontsize=14)
        plt.scatter(curvatures, np.array(mismatchFirst)[:,0])
        plt.scatter(curvatures, np.array(mismatchFirst)[:,1])
        # plt.scatter(curvatures, np.array(mismatchFirst)[:,2])
        # plt.scatter(curvatures, np.array(mismatchFirst)[:,3])
        plt.legend([r'$s$',r'$e_y$'])
        plt.pause(0.001)
        plt.savefig(foldername + '/mismatchFirstCurve' + str(i) + '.png')
        plt.close()
        
        plt.figure(figsize=(8,6))
        plt.title('Linearization-Induced Model Mismatch', fontsize=14)
        plt.ylabel(r'|Predicted $x_{t+N}$ - Actual $x_{t+N}$|', fontsize=14)
        plt.xlabel('Current Curvature', fontsize=14)
        plt.scatter(curvatures, np.array(mismatchLast)[:,0])
        plt.scatter(curvatures, np.array(mismatchLast)[:,1])
        # plt.scatter(curvatures, np.array(mismatchLast)[:,2])
        # plt.scatter(curvatures, np.array(mismatchLast)[:,3])
        plt.legend([r'$s$',r'$e_y$'])
        plt.pause(0.001)
        plt.savefig(foldername + '/mismatchLastCurve' + str(i) + '.png')
        plt.close()
        
        # Against theta
        plt.figure(figsize=(8,6))
        plt.title('Linearization-Induced Model Mismatch', fontsize=14)
        plt.ylabel(r'|Predicted $x_{t+1}$ - Actual $x_{t+1}$|', fontsize=14)
        plt.xlabel(r'Current $\theta$', fontsize=14)
        plt.scatter(np.array(xTraj[:-1])[:,3], np.array(mismatchFirst)[:,0])
        plt.scatter(np.array(xTraj[:-1])[:,3], np.array(mismatchFirst)[:,1])
        # plt.scatter(np.array(xTraj[:-1])[:,3], np.array(mismatchFirst)[:,2])
        # plt.scatter(np.array(xTraj[:-1])[:,3], np.array(mismatchFirst)[:,3])
        plt.legend([r'$s$',r'$e_y$'])
        plt.pause(0.001)
        plt.savefig(foldername + '/mismatchFirstTheta' + str(i) + '.png')
        plt.close()
        
        plt.figure(figsize=(8,6))
        plt.title('Linearization-Induced Model Mismatch', fontsize=14)
        plt.ylabel(r'|Predicted $x_{t+N}$ - Actual $x_{t+N}$|', fontsize=14)
        plt.xlabel(r'Current $\theta$', fontsize=14)
        plt.scatter(np.array(xTraj[:-1])[:,3], np.array(mismatchLast)[:,0])
        plt.scatter(np.array(xTraj[:-1])[:,3], np.array(mismatchLast)[:,1])
        # plt.scatter(np.array(xTraj[:-1])[:,3], np.array(mismatchLast)[:,2])
        # plt.scatter(np.array(xTraj[:-1])[:,3], np.array(mismatchLast)[:,3])
        plt.legend([r'$s$',r'$e_y$'])
        plt.pause(0.001)
        plt.savefig(foldername + '/mismatchLastTheta' + str(i) + '.png')
        plt.close()
        
        # Against 1/(1- y k)
        yVals = np.array(xTraj[:-1])[:,1]
        denoms = np.divide(1, 1 - yVals * np.array(curvatures))
        plt.figure(figsize=(8,6))
        plt.title('Linearization-Induced Model Mismatch', fontsize=14)
        plt.ylabel(r'|Predicted $x_{t+1}$ - Actual $x_{t+1}$|', fontsize=14)
        plt.xlabel(r'Current $1/(1 - e_y \kappa)$', fontsize=14)
        plt.scatter(denoms, np.array(mismatchFirst)[:,0])
        plt.scatter(denoms, np.array(mismatchFirst)[:,1])
        # plt.scatter(np.array(xTraj[:-1])[:,3], np.array(mismatchFirst)[:,2])
        # plt.scatter(np.array(xTraj[:-1])[:,3], np.array(mismatchFirst)[:,3])
        plt.legend([r'$s$',r'$e_y$'])
        plt.pause(0.001)
        plt.savefig(foldername + '/mismatchFirstDen' + str(i) + '.png')
        plt.close()
        
        plt.figure(figsize=(8,6))
        plt.title('Linearization-Induced Model Mismatch', fontsize=14)
        plt.ylabel(r'|Predicted $x_{t+N}$ - Actual $x_{t+N}$|', fontsize=14)
        plt.xlabel(r'Current $1/(1 - e_y \kappa)$', fontsize=14)
        plt.scatter(denoms, np.array(mismatchLast)[:,0])
        plt.scatter(denoms, np.array(mismatchLast)[:,1])
        # plt.scatter(np.array(xTraj[:-1])[:,3], np.array(mismatchLast)[:,2])
        # plt.scatter(np.array(xTraj[:-1])[:,3], np.array(mismatchLast)[:,3])
        plt.legend([r'$s$',r'$e_y$'])
        plt.pause(0.001)
        plt.savefig(foldername + '/mismatchLastDen' + str(i) + '.png')
        plt.close()
        
        # Against yawPrime
        yawPrimes = [spline.calc_yawPrime(s) for s in np.array(xTraj[:-1])[:,0]]
        plt.figure(figsize=(8,6))
        plt.title('Linearization-Induced Model Mismatch', fontsize=14)
        plt.ylabel(r'|Predicted $x_{t+1}$ - Actual $x_{t+1}$|', fontsize=14)
        plt.xlabel(r"Current $\gamma'(s)$", fontsize=14)
        plt.scatter(yawPrimes, np.array(mismatchFirst)[:,0])
        plt.scatter(yawPrimes, np.array(mismatchFirst)[:,1])
        # plt.scatter(np.array(xTraj[:-1])[:,3], np.array(mismatchFirst)[:,2])
        # plt.scatter(np.array(xTraj[:-1])[:,3], np.array(mismatchFirst)[:,3])
        plt.legend([r'$s$',r'$e_y$'])
        plt.pause(0.001)
        plt.savefig(foldername + '/mismatchFirstYawPrime' + str(i) + '.png')
        plt.close()

        plt.figure(figsize=(8,6))
        plt.title('Linearization-Induced Model Mismatch', fontsize=14)
        plt.ylabel(r'|Predicted $x_{t+N}$ - Actual $x_{t+N}$|', fontsize=14)
        plt.xlabel(r"Current $\gamma'(s)$", fontsize=14)
        plt.scatter(yawPrimes, np.array(mismatchLast)[:,0])
        plt.scatter(yawPrimes, np.array(mismatchLast)[:,1])
        # plt.scatter(np.array(xTraj[:-1])[:,3], np.array(mismatchFirst)[:,2])
        # plt.scatter(np.array(xTraj[:-1])[:,3], np.array(mismatchFirst)[:,3])
        plt.legend([r'$s$',r'$e_y$'])
        plt.pause(0.001)
        plt.savefig(foldername + '/mismatchLastYawPrime' + str(i) + '.png')
        plt.close()
        
        costs.append(lmpcSolver.updateSSandValueFunction(xTraj, uTraj))
        
    with open(foldername + '/spline.pkl', 'wb') as output:
        pickle.dump(spline, output, pickle.HIGHEST_PROTOCOL)
        
    np.save(foldername + '/costs', costs)
    np.save(foldername + '/maxVelocities', maxVelocities)
    
    for i, xTraj in enumerate(trajectories):
        fig, ax = visualizeHeatmapTrajectory(spline, xTraj, obstacleList, vMax = max(maxVelocities))
        fig.suptitle('Velocity Heatmap: Iter = ' + str(i))
        plt.pause(0.001)
        fig.savefig(foldername + '/heatmap' + str(i) + '.png')    
        plt.close()
    
    fig, ax = plt.subplots()
    plt.title('Trajectory Cost across Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.plot(costs)
    plt.savefig(foldername + '/overallCost.png')
    
    return costs

def convertToXY(spline, states):
    xyStates = []
    for state in states: 
        try:    
            x,y = spline.calcXY(state[0], state[1])
        except:
            pdb.set_trace()
        xyStates.append(np.array([x, y, state[2], state[3]]))
    return np.array(xyStates)

def visualizeElapsedTrajectory(spline, xTrajList, terminalPointsList, counts, obstacleList=None):
    n = len(xTrajList)
    fig, axes = plt.subplots(n) 
    
    splineLine = []
    for s in np.linspace(0, spline.end, 1000, endpoint=False):
        splineLine.append(spline.calc_position(s))
    splineLine = np.array(splineLine)
    
    for i in range(n):
        ax = axes[i]
        xTraj = np.copy(xTrajList[i])
        terminalPoints = np.copy(terminalPointsList[i])
        
        # Plot the spline
        ax.plot(splineLine[:,0], splineLine[:,1], '--r', label='Spline')
    
        # Plot the terminal points
        terminalXY = convertToXY(spline, terminalPoints.T)
        ax.scatter(terminalXY[:,0], terminalXY[:,1], color='k', s=50, label='Terminal Points')
            
        # Plot the elapsed trajectory
        xyCoords = convertToXY(spline, xTraj)
        ax.plot(xyCoords[:,0], xyCoords[:,1], '--ob', label='Elapsed')
        
        if obstacleList is not None:
            for (ox, oy, size) in obstacleList:
                plot_circle(ox, oy, size)
                    
        if i == 0:
            ax.legend()
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_title('Count = ' + str(counts[i]))
    
    fig.suptitle('Closed-Loop Trajectory')
    
    return fig, axes

def visualizeHeatmapTrajectory(spline, xTraj, obstacleList=None, vMax=None):
    fig, ax = plt.subplots() 

    splineLine = []
    for s in np.linspace(0, spline.end, 1000, endpoint=False):
        splineLine.append(spline.calc_position(s))
    splineLine = np.array(splineLine)

    # Plot the spline
    ax.plot(splineLine[:,0], splineLine[:,1], '--r', label='Spline')
    
    velocities = np.array(xTraj)[:,2]
    
    if vMax is None:
        vMax = max(velocities)
        
    # Plot the elapsed trajectory
    xyCoords = convertToXY(spline, xTraj)
    cm = plt.cm.get_cmap('viridis')
    sc = ax.scatter(xyCoords[:,0], xyCoords[:,1], c=velocities, cmap=cm, label='Traj')
    plt.colorbar(sc)
    sc.set_clim([0,vMax])

    if obstacleList is not None:
        for (ox, oy, size) in obstacleList:
            plot_circle(ox, oy, size)
              
    ax.legend()
        
    return fig, ax

if __name__ == '__main__':
    costs = main()
    
    # After all trajectories are generated
    