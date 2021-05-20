# Author: Aaron Feldman
# Code which does LMPC, including wrapping of ftocp object to implement an SQP
# solver given terminal points and value function

import numpy as np
import matplotlib.pyplot as plt
from ftocp import FTOCP

class LMPC(object):
    
    # Assume some demonstrations are given so that SS is non-empty
    # SS should be a list of states
    def __init__(self, N, Q, R, SS, spline, dt, width, amax, amin, theta_dotMax, theta_dotMin, printLevel):
        self.printLevel = printLevel
        self.N = N
        self.Q = Q
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
        # TODO: Need to compute the value function approximation for each point in SS
        
    # Given a current location, run one open-loop trajectory using SQP
    # numIters controls how many times to run batch approach
    # x0 is the start state, goal is the target state
    # Ultimately, when are repeatedly calling this function can use previous
    # uPred with offset of one as uGuess
    def runSQP(self, x0, goal, terminalPoints, valuePoints, uGuess = None, numIters=2):
        if uGuess is None:
            uGuess = [np.array([self.amax / 10, self.theta_dotMax / 10])]*self.N
        
        ftocp = FTOCP(self.N, self.Q, self.R, self.Fx, self.bx, self.Fu, self.bu, terminalPoints, valuePoints, self.spline, self.dt, uGuess, goal, self.printLevel)
        if self.printLevel >= 2:
            fig, ax = plt.subplots()
            ax.title('Predicted trajectory')
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
            ax.xlim(-1,12)
            ax.ylim(-1,10)
            ax.legend()
            
        for i in range(numIters):
            ftocp.solve(x0)
            
            if self.printLevel >= 2:
                ax.plot(ftocp.xPred[:,0], ftocp.xPred[:,1], '--ob', label='SQP iter = ' + str(i))
                ax.show()
        
            # To update the linearization, use as uGuess the uPred from
            # previous iteration (will then also update xGuess internally)
            ftocp.uGuessUpdate() 
        
        # Return the resulting predicted trajectory and control sequence
        return ftocp.xPred, ftocp.uPred
                
    # Run a full trajectory going from start to end of the spline using
    # closed-loop receding horizon updating
    def runTrajectory(self):
        # while not sufficiently close to goal state and < max iterations:
            # 1. Set target
            # If this is first iteration, use the N'th point from the last
            # trajectory
            # Else, use the final state predicted by the last SQP run
            
            # 2. Determine terminal region
            # Compute new terminalPoints and valuePoints using nearest neighbors
            
            # 3. If this is not the first iteration, set uGuess using
            # one-offset from past SQP uPred as in HW2 problem 1 ftocp 
            # uGuessUpdate code
            # Else, leave uGuess as none
            
            # 4. runSQP using the current state. Save uPred for step 3 and 
            # last state in xPred for step 1. Execute the first control action:
            # compute the updated state using the *nonlinear* dynamics and this
            # control action.
            
            # 5. Record the executed control action and updated state  
        
        # return the list of executed control actions and corresponding states
        pass
    
    # Takes in a state, determines K-nearest neighbors in SS, K = numNearest
    # and returns the relevant indices into SS
    def computeKnearest(self, numNearest, target):
        # Each state in self.SS will become a row so want axis=1 to
        # take norm across each row
        return np.argsort(np.linalg.norm(target - np.array(self.SS), axis=1))[:numNearest]
    
    # Given a new, full trajectory defined by xTraj, uTraj update the SS
    # and value function
    def updateSSandValueFunction(self, xTraj, uTraj):
        pass
    
    
