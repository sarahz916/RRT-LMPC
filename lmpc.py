# Author: Aaron Feldman
# Code which does LMPC, including wrapping of ftocp object to implement an SQP
# solver given terminal points and value function

import numpy as np
import matplotlib.pyplot as plt
from ftocpLMPC import FTOCP
from scipy import linalg
from casadi import sin, cos
import pdb

class LMPC(object):
    
    # Assume some demonstrations are given so that before calling 
    # runSQP SS is non-empty
    # SS should be a list of states and values contains corresponding value
    # for them
    def __init__(self, N, K, Q, Qf, R, SS, values, spline, dt, width, amax, amin, theta_dotMax, theta_dotMin, printLevel):
        self.printLevel = printLevel
        self.N = N
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.SS = SS
        self.dt = dt
        self.Fx = np.array([[0, 1, 0, 0],[0, -1, 0, 0]])
        self.bx = np.array([width, width])
        self.Fu = np.vstack([np.eye(2), -np.eye(2)])
        self.bu = np.array([amax, theta_dotMax, -amin, -theta_dotMin])
        self.spline = spline
        # Set as endpoint of the spline with 0 angle and 0 velocity
        self.goal = np.array([spline.end, 0, 0, 0])
        # Start with s = 0, on centerline, 0 angle, 0 velocity
        self.start = np.array([0, 0, 0, 0])
        self.values = values
        self.K = K # Number of nearest neighbors to use
        self.amax = amax
        self.amin = amin
        self.theta_dotMax = theta_dotMax
        self.theta_dotMin = theta_dotMin
        
    # Given a current location, run one open-loop trajectory using SQP
    # numIters controls how many times to run batch approach
    # x0 is the start state, goal is the target state
    # Ultimately, when are repeatedly calling this function can use previous
    # uPred with offset of one as uGuess
    def runSQP(self, x0, goal, terminalPoints, valuePoints, uGuess = None, numIters=2):
        if uGuess is None:
            uGuess = [np.array([self.amax / 100, 0])]*self.N
        
        ftocp = FTOCP(self.N, self.Q, self.R, self.Fx, self.bx, self.Fu, self.bu, terminalPoints, valuePoints, self.spline, self.dt, uGuess, goal, self.printLevel)
        if self.printLevel >= 2:
            fig, ax = plt.subplots()
            ax.set_title('Predicted trajectory')
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
                
        # TODO: TEMPORARY
        # SHOULD PUT BACK TO using ftocp.xPred instead of xDemo
        # Let's see what would happen if used uGuess to propogate solution
        xDemo = [x0]
        for i, ut in enumerate(uGuess):
            xDemo.append(self.dynamics(xDemo[-1], ut))
        
        for i in range(numIters):
            ftocp.solve(x0)
            
            if self.printLevel >= 2:
                # Need to convert back to x,y
                xyCoords = []
                # for state in ftocp.xPred:
                for state in ftocp.xPred:
                    try:
                        x,y = self.spline.calcXY(state[0], state[1])
                    except:
                        pdb.set_trace()
                    xyCoords.append((x,y))
                xyCoords = np.array(xyCoords)
                ax.plot(xyCoords[:,0], xyCoords[:,1], '--ob', label='SQP iter = ' + str(i))
                
                # Visualize the safe set and target x,y
                terminalXY = self.convertToXY(terminalPoints.T)
                ax.scatter(terminalXY[:,0], terminalXY[:,1], label='Terminal Points')
                ax.legend()
                
                pdb.set_trace()
            
            # To update the linearization, use as uGuess the uPred from
            # previous iteration (will then also update xGuess internally)
            ftocp.uGuessUpdate() 
        
        # Return the resulting predicted trajectory and control sequence
        return ftocp.xPred, ftocp.uPred
                
    # Run a full trajectory going from start to end of the spline using
    # closed-loop receding horizon updating
    # Takes in a previous (demonstration) trajectory
    def runTrajectory(self, xDemo, uDemo, eps = 1, maxIter = 1e3):
        # Placeholders
        distLeft = np.inf
        xPred = []
        uPred = []
        
        count = 0
        x0 = self.start
        xTraj = []
        uTraj = []
        
        # while not sufficiently close to goal state and < max iterations:
        while distLeft > eps and count < maxIter:
            print('Count is ' + str(count))
            # 1. Set target
            # If this is first iteration, use the N'th point from the last
            # trajectory
            # Else, use the final state predicted by the last SQP run
            if count == 0:
                target = xDemo[self.N]
                # If this is the first iteration also set uGuess
                # based on the first N steps of the demonstration trajectory
                uGuess = []
                for i in range(self.N):
                    uGuess.append(uDemo[i])
            else:
                target = xPred[-1]
            
            # 2. Determine terminal region
            # Select new terminalPoints and get corresponding valuePoints 
            # using nearest neighbors
            safeIndices = self.computeKnearest(self.K, target)
            # Should be a matrix of form n x K
            terminalPoints = np.array([self.SS[ind] for ind in safeIndices]).T
            valuePoints = np.array([self.values[ind] for ind in safeIndices])
            
            if self.printLevel >= 2:
                # Visualize the safe set and target x,y
                terminalXY = self.convertToXY(terminalPoints.T)
                targetXY = self.spline.calcXY(target[0], target[1])
                
                # Visualize the full set of demo points
                demoXY = self.convertToXY(xDemo)
                
                plt.figure()
                plt.scatter(demoXY[:,0], demoXY[:,1], label='Full Demo')
                plt.scatter(terminalXY[:,0], terminalXY[:,1], label='Terminal Points')
                plt.scatter(targetXY[0], targetXY[1], label='Target')
                
                plt.legend()
                plt.title('Visualizing Safe Set Points')
                plt.xlabel(r'$x_1$')
                plt.ylabel(r'$x_2$')
                plt.xlim([0,20])
                plt.ylim([0,20])
                
            # 3. If this is not the first iteration, set uGuess using
            # one-offset from past SQP uPred as in HW2 problem 1 ftocp 
            # uGuessUpdate code
            # Else, leave uGuess as none
            # uGuess = None
            if count > 0:
                uGuess = []
                for i in range(self.N):
                    # So, if i < N-1 get uPred[i+1]
                    # and then uGuess[N-1] = uPred[N-1]
                    uGuess.append(uPred[min(i+1, self.N-1)])
            
            # 4. runSQP using the current state. Save uPred for step 3 and 
            # last state in xPred for step 1.
            xPred, uPred = self.runSQP(x0, self.goal, terminalPoints, valuePoints, uGuess)
            
            # 5. Execute the first control action:
            # compute the updated state using the *nonlinear* dynamics and this
            # control action.
            xNext = self.dynamics(x0, uPred[0])
            
            # 6. Record the executed control action and updated state  
            xTraj.append(xNext)
            uTraj.append(uPred[0])
            
            # 7. Compute the new distance to goal and update count
            deltaX = xNext - self.goal
            distLeft = deltaX.T @ self.Q @ deltaX
            count += 1
            
        # return the list of executed control actions and corresponding states
        xTraj, uTraj
    
    # Takes in a state, determines K-nearest neighbors in SS, K = numNearest
    # and returns the relevant indices into SS
    # Uses inner product defined by self.Q
    def computeKnearest(self, numNearest, target):
        # Each state in self.SS will become a row
        # Should be numStates x n 
        deltaX = target - np.array(self.SS) 
        numStates = deltaX.shape[0]
        # listQ = [self.Q] * numStates
        # barQ = linalg.block_diag(linalg.block_diag(*listQ))
        
        # Diagonal elements of this matrix give the squared Q-norms
        distance = np.diag(deltaX @ self.Q @ deltaX.T)
                
        # Gives generalized distance
        # distance = deltaX.T @ barQ @ deltaX
        return np.argsort(distance)[:numNearest]
    
    # Given a new, full trajectory defined by xTraj, uTraj update the SS
    # and value function
    # xTraj, uTraj should be lists where each element is an array
    def updateSSandValueFunction(self, xTraj, uTraj):
        # Start at trajectory end and work backwards adding costs
        M = len(xTraj)
        listQ = [self.Q] * (M-1) + [self.Qf]
        listR = [self.R] * M
        
        pointValues = []
        for i in range(M-1,-1,-1):
            deltaX = xTraj[i] - self.goal
            stageCost = deltaX.T @ listQ[i] @ deltaX + uTraj[i].T @ listR[i] @ uTraj[i]
            if len(pointValues):
                costToCome = pointValues[-1]
            else:
                costToCome = 0
            pointValues.append(stageCost + costToCome)
        
        pointValues = pointValues[::-1]
        self.SS.extend(xTraj)
        self.values.extend(pointValues)
    
    def dynamics(self, x, u):
        # state = [s, y, v, theta]
        # input = [acc, theta_dot]
        # use Euler discretization
        gamma = self.spline.calc_yaw(x[0])
        curvature = self.spline.calc_curvature(x[0])
        deltaS = x[2] * cos(x[3] - gamma) / (1 - gamma * curvature)
        deltaY = x[2] * sin(x[3] - gamma)
        s_next      = x[0] + self.dt * deltaS
        y_next      = x[1] + self.dt * deltaY
        v_next      = x[2] + self.dt * u[0]
        theta_next  = x[3] + self.dt * u[1]

        state_next = [s_next, y_next, v_next, theta_next]

        return state_next
    
    # Assume states is a list of states (or array where each row is a state)
    def convertToXY(self, states):
        xyStates = []
        for state in states: 
            try:    
                x,y = self.spline.calcXY(state[0], state[1])
            except:
                pdb.set_trace()
            xyStates.append(np.array([x, y, state[2], state[3]]))
        return np.array(xyStates)