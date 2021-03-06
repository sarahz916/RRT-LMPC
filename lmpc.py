# Author: Aaron Feldman
# Code which does LMPC, including wrapping of ftocp object to implement an SQP
# solver given terminal points and value function

import numpy as np
import matplotlib.pyplot as plt
from ftocpLMPC import FTOCP
from scipy import linalg
# from casadi import sin, cos
import math
import pdb
from nlpSdynamics import NLP

class LMPC(object):
    
    # Assume some demonstrations are given so that before calling 
    # runSQP SS is non-empty
    # SS should be a list of states and values contains corresponding value
    # for them
    def __init__(self, N, K, Q, Qf, R, regQ, regR, SS, values, spline, dt, \
                 width, amax, amin, theta_dotMax, theta_dotMin, radius, printLevel, slackCost): #, memory):
        self.printLevel = printLevel
        self.N = N
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.SS = SS
        self.dt = dt
        self.Fx = np.array([[1,0,0,0],[-1,0,0,0],[0, 1, 0, 0],[0, -1, 0, 0],[0,0,-1,0]])
        self.bx = np.array([spline.end, 0, width, width, 0])
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
        self.regQ = regQ
        self.regR = regR
        self.radius = radius
        self.slackCost = slackCost
        self.trajLengths = []
        # self.memory = memory
    # Given a current location, run one open-loop trajectory using SQP
    # numIters controls how many times to run batch approach
    # x0 is the start state, goal is the target state
    # Ultimately, when are repeatedly calling this function can use previous
    # uPred with offset of one as uGuess
    def runSQP(self, x0, goal, terminalPoints, valuePoints, uGuess, disp=False, maxIters=8, eps=0.05):
        # if uGuess is None:
        #     uGuess = [np.array([self.amax / 100, 0])]*self.N
        
        ftocp = FTOCP(self.N, self.Q, self.R, self.Fx, self.bx, self.Fu, self.bu, \
                      self.regQ, self.regR, terminalPoints, valuePoints, self.spline, \
                          self.dt, uGuess, goal, 0, self.slackCost)
        
        # Let's see what would happen if used uGuess to propogate solution
        xDemo = [x0]
        for i, ut in enumerate(uGuess):
            xDemo.append(self.dynamics(xDemo[-1], ut))
        demoXY = self.convertToXY(xDemo)
        
        converged = False
                
        count = 0
        while count < maxIters and not converged:
            # print('Entering solver!!!!!!!!')
            
            if count != 0:
                currX = np.copy(ftocp.xGuess)
            
            feasibleFlag = ftocp.solve(x0)
            
            if type(feasibleFlag) != type(np.zeros(1)) or None in ftocp.xPred:
                plt.figure()
                xyTerminal = self.convertToXY(ftocp.terminalPoints.T)
                xyGuess = self.convertToXY(ftocp.xGuess)
                plt.scatter(xyTerminal[:,0], xyTerminal[:,1], label='Terminal')
                plt.scatter(xyGuess[:,0], xyGuess[:,1], label='Guess')
                plt.legend()
                
                pdb.set_trace()
            
            # To update the linearization, use as uGuess the uPred from
            # previous iteration (will then also update xGuess internally)
            ftocp.uGuessUpdate() 
            
            # Look at the difference between past first action and this action
            nextX = np.copy(ftocp.xGuess) 
            
            if count != 0:
                deltaX = nextX - currX
                distances = np.sqrt(np.diag(deltaX @ self.Q @ deltaX.T)[1:])
                if np.mean(distances) < eps:
                    converged = True
                    # print('GUESS = ', nextX[-1])
                    # print('PRED = ', ftocp.xPred[-1])
                    
                # print('Iteration = ' + str(count))
                # print('deltaX = ', deltaX)
                # print('Converged = ', converged)
                
            if disp:
                if self.printLevel >= 2 and count == 0:
                    fig, ax = plt.subplots()
                    ax.set_title('Predicted trajectory', fontsize=14)
                    ax.set_xlabel('x', fontsize=14)
                    ax.set_ylabel('y', fontsize=14)
                        
                # Need to convert back to x,y
                xyCoords = []
                for j, state in enumerate(ftocp.xPred):
                    try:
                        # Account for numerical issues in solver
                        # if state[0] < 0 and np.abs(state[0]) < 1e-4:
                        #     x, y = self.spline.calcXY(0, state[1])
                        # else:
                        x, y = self.spline.calcXY(state[0], state[1])
                    except:
                        print('Error')
                        pdb.set_trace()
                    xyCoords.append((x,y))
                xyCoords = np.array(xyCoords)
                
                xTrue = [x0]
                for j, ut in enumerate(ftocp.uPred):
                    xTrue.append(self.dynamics(xTrue[-1], ut))
                trueXY = self.convertToXY(xTrue)
                
                # Propogate the dynamics using the linear approximation and
                # see what happens
                xProp = [x0]
                for j, ut in enumerate(ftocp.uPred):
                    xBar = ftocp.xGuess[j]
                    uBar = ftocp.uGuess[j]
                    A, B, C = ftocp.buildLinearizedMatrices(xBar, uBar)
                    xNext = A @ xProp[-1] + B @ ut + C
                    xProp.append(xNext)
                
                xyProp = self.convertToXY(xProp)
                
                # ax.plot(xyCoords[:,0], xyCoords[:,1], '--o', label='FTOCP result ' + str(count))
                ax.plot(trueXY[:,0], trueXY[:,1], '--o', label='Nonlinear result ' + str(count))
                ax.plot(xyProp[:,0], xyProp[:,1], '--o', label='Linear result ' + str(count))
                # Visualize the safe set
                if count == 0: 
                    terminalXY = self.convertToXY(terminalPoints.T)
                    ax.scatter(terminalXY[:,0], terminalXY[:,1], color='k', s=50, label='Terminal Points')
                    ax.plot(demoXY[:,0], demoXY[:,1], '--o', label='Guess ' + str(count))
                ax.legend()
                plt.pause(0.001)
                
                # print('Lambdas sum to: ', sum(ftocp.lambdas))
                # print('Specify combination: ', sum([ftocp.lambdas[count] * ftocp.terminalPoints.T[count] for count in range(30)]))
                # print('ftocp final: ', ftocp.xPred[-1])
                # if ftocp.xPred[-1][2] - 1 > 1e-2:
                #     pdb.set_trace()
                    
            count += 1

        # pdb.set_trace()
        
        slackContr = ftocp.slack.T @ self.slackCost @ ftocp.slack
        
        print('slackContr = ', slackContr)
        # Return the resulting predicted trajectory and control sequence
        return ftocp.xPred, ftocp.uPred, ftocp.xGuess, ftocp.Cost, slackContr
    
    # Run a full trajectory going from start to end of the spline using
    # closed-loop receding horizon updating
    # Takes in a previous (demonstration) trajectory
    def runTrajectory(self, xDemo, uDemo, eps = 1, maxIter = 500):
        # Placeholders
        distLeft = np.inf
        xPred = []
        uPred = []
        
        # Never used, just silence error
        xGuess = []
        
        count = 0
        xTraj = [self.start]
        uTraj = []
        
        # if self.printLevel >= 1:
        #     plt.ion()
        #     fig, ax = plt.subplots()
        
        # For display purposes
        xTrajList = []
        terminalPointsList = []
        counts = []
        mismatchFirst = []
        mismatchLast = []
        openLoopCosts = []
        slacks = []
        
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
                # target = xGuess[-1]
            
            # 2. Determine terminal region
            # Select new terminalPoints and get corresponding valuePoints 
            # using nearest neighbors
            safeIndices = self.computeKnearest(self.K, target)
            if len(xGuess):
                guessSafeIndices = self.computeKnearest(self.K, xGuess[-1])
                safeIndices = np.concatenate([safeIndices, guessSafeIndices])
                
            # safeIndices = self.getTerminalPoints(self.K, target)
            # if len(xGuess):
            #     guessSafeIndices = self.getTerminalPoints(self.K, xGuess[-1])
            #     safeIndices = np.concatenate([safeIndices, guessSafeIndices])   
                
            # safeIndices = self.getTerminalPoints(self.K, target)
            # Should be a matrix of form n x K
            terminalPoints = np.array([self.SS[ind] for ind in safeIndices]).T
            valuePoints = np.array([self.values[ind] for ind in safeIndices])
            
            # if self.printLevel >= 1:
            #     x_1, y_1 = self.spline.calcXY(*xTraj[-1][:2])
            #     ax.scatter(x_1, y_1, color='r')
                
            # if self.printLevel >= 1:
            #     plt.figure()
            #     # Visualize the safe set and target x,y
            #     terminalXY = self.convertToXY(terminalPoints.T)
            #     targetXY = self.spline.calcXY(target[0], target[1])
            #     plt.scatter(terminalXY[:,0], terminalXY[:,1])
            #     plt.scatter(targetXY[0], targetXY[1])
            # pdb.set_trace()
            
            #     # Visualize the full set of demo points
            #     demoXY = self.convertToXY(xDemo)
                
            #     ax.scatter(terminalXY[:,0], terminalXY[:,1], label='Terminal Points')
            #     ax.scatter(targetXY[0], targetXY[1], label='Target')
                
            #     # plt.figure()
            #     # plt.scatter(demoXY[:,0], demoXY[:,1], label='Full Demo')
            #     # plt.scatter(terminalXY[:,0], terminalXY[:,1], label='Terminal Points')
            #     # plt.scatter(targetXY[0], targetXY[1], label='Target')
                
            #     plt.legend()
            #     plt.pause(0.001)
            #     # plt.title('Visualizing Safe Set Points')
            #     # plt.xlabel(r'$x_1$')
            #     # plt.ylabel(r'$x_2$')
            #     # plt.xlim([0,20])
            #     # plt.ylim([0,20])
                
            
            print('----> Current location = ')
            print(xTraj[-1])
            print('Fraction Completed = ', xTraj[-1][0] / self.spline.end)
            
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
            disp = (count % 50 == 0) and count > 0
            # disp = True
            if disp:
                xTrajList.append(np.copy(xTraj))
                terminalPointsList.append(np.copy(terminalPoints))
                counts.append(count)
                # pdb.set_trace()
            
            xPred, uPred, xGuess, openLoopCost, slack = \
                self.runSQP(xTraj[-1], self.goal, terminalPoints, valuePoints, uGuess, disp)
            
            # Also record where the solver thought it was going
            # expected.append(xPred[1])
            
            slacks.append(slack)
            
            # Record the open loop (regularized) cost
            openLoopCosts.append(openLoopCost)
            
            # 5. Execute the first control action:
            # compute the updated state using the *nonlinear* dynamics and this
            # control action.
            xNext = self.dynamics(xTraj[-1], uPred[0])
            
            xDemo = [xPred[0]]
            for i, ut in enumerate(uPred):
                xDemo.append(self.dynamics(xDemo[-1], ut))
            
            mismatchFirst.append(np.abs(xPred[1] - xNext))
            mismatchLast.append(np.abs(xPred[-1] - xDemo[-1]))
            
            # 6. Record the executed control action and updated state  
            xTraj.append(xNext)
            uTraj.append(uPred[0])
            
            # 7. Compute the new distance to goal and update count
            deltaX = xNext - self.goal
            distLeft = np.sqrt(deltaX.T @ self.Q @ deltaX)
            count += 1
            
        xTrajList.append(np.copy(xTraj))
        terminalPointsList.append(np.copy(terminalPoints))
        counts.append(count)
        
        # fig1, axes1 = self.visualizeElapsedTrajectory(xTrajList, terminalPointsList, counts)
                
        # fig2, ax2 = self.visualizeHeatmapTrajectory(xTraj)
        
        # pdb.set_trace()

        # return the list of executed control actions and corresponding states
        return xTraj, uTraj, xTrajList, terminalPointsList, counts, mismatchFirst, mismatchLast, openLoopCosts, slacks
    
    # Takes in a state, determines K-nearest neighbors in SS, K = numNearest
    # and returns the relevant indices into SS
    # Uses inner product defined by self.Q
    def computeKnearest(self, numNearest, target):
        # if len(self.trajLengths) > self.memory:
        #     forget = sum(self.trajLengths[:-self.memory]) - len(self.trajLengths) + self.memory
        #    safeset = np.copy(self.SS[forget:])
        # else:
        #     safeset = np.copy(self.SS)
        # Each state in self.SS will become a row
        # Should be numStates x n 
        safeset = self.SS
        deltaX = target - np.array(safeset) 
        numStates = deltaX.shape[0]
        # listQ = [self.Q] * numStates
        # barQ = linalg.block_diag(linalg.block_diag(*listQ))
        
        # Diagonal elements of this matrix give the squared Q-norms
        distance = np.diag(deltaX @ self.Q @ deltaX.T)
                
        # Gives generalized distance
        # distance = deltaX.T @ barQ @ deltaX
        return np.argsort(distance)[:numNearest]
    
    # Consider uniformly drawing terminal points from a ball around the target
    
    def getTerminalPoints(self, numPoints, target):
        # Each state in self.SS will become a row
        # Should be numStates x n 
        deltaX = target - np.array(self.SS) 
        numStates = deltaX.shape[0]
        # listQ = [self.Q] * numStates
        # barQ = linalg.block_diag(linalg.block_diag(*listQ))
        
        # Diagonal elements of this matrix give the squared Q-norms
        distance = np.sqrt(np.diag(deltaX @ self.Q @ deltaX.T))
        
        neighborRad = np.sort(distance)[numPoints]
        
        r = max(neighborRad, self.radius)
        
        closeIndices = np.where(distance < r)[0]
        numNear = len(closeIndices)
        return np.random.choice(closeIndices, min(numNear, numPoints))
        
    # Given a new, full trajectory defined by xTraj, uTraj update the SS
    # and value function
    # xTraj, uTraj should be lists where each element is an array
    def updateSSandValueFunction(self, xTraj, uTraj):
        # Start at trajectory end and work backwards adding costs
        M = len(xTraj)
        listQ = [self.Q] * (M-1)
        listR = [self.R] * (M-1)
        
        pointValues = []
        for i in range(M-1,-1,-1):
            deltaX = xTraj[i] - self.goal
            if i == M-1:
                pointValues.append(deltaX.T @ self.Qf @ deltaX)
            else:
                stageCost = deltaX.T @ listQ[i] @ deltaX + uTraj[i].T @ listR[i] @ uTraj[i]
                costToCome = pointValues[-1]
                pointValues.append(stageCost + costToCome)
        
        
        pointValues = pointValues[::-1]
        self.SS.extend(xTraj)
        self.values.extend(pointValues)
        self.trajLengths.append(len(xTraj))
        
        # Return the overall cost of the trajectory
        return pointValues[0]

    # Removed buggy code for above
    # pointValues = []
    # for i in range(M-1,-1,-1):
    #     deltaX = xTraj[i] - self.goal
    #     try:
    #         stageCost = deltaX.T @ listQ[i] @ deltaX + uTraj[i].T @ listR[i] @ uTraj[i]
    #     except:
    #         pdb.set_trace()
    #     if len(pointValues):
    #         costToCome = pointValues[-1]
    #     else:
    #         costToCome = 0
    #     pointValues.append(stageCost + costToCome)
            

    # flag=1 to do dynamics via x
    def dynamics(self, x, u, flag=1):
        if flag:
            curr_state = np.copy(x)
            # Convert to (x,y)
            curr_state[0], curr_state[1] = self.spline.calcXY(x[0],x[1])
            acc = u[0]
            theta_dot = u[1]
            new_ang = theta_dot * self.dt + curr_state[3]
            new_vel = acc*self.dt + curr_state[2]
            new_x = curr_state[2] * self.dt * math.cos(curr_state[3]) + curr_state[0]
            new_y = curr_state[2] * self.dt * math.sin(curr_state[3]) + curr_state[1]
            # Convert back to (s,ey)
            new_s, new_ey = self.spline.calcSY(new_x, new_y)
            return [new_s, new_ey, new_vel, new_ang]
                
        else:
            # state = [s, y, v, theta]
            # input = [acc, theta_dot]
            # use Euler discretization
            gamma = self.spline.calc_yaw(x[0])
            curvature = self.spline.calc_curvature(x[0])
            deltaS = x[2] * math.cos(x[3] - gamma) / (1 - x[1] * curvature)
            deltaY = x[2] * math.sin(x[3] - gamma)
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
    
    # target velocity will interpolate between cruise_velocity and start
    # and end goal velocity while s is within rampLength fraction of start/end
    # Kvals controls proportionality constants of the controller
    def createDemo(self, Kvals, cruise_velocity=1, rampLength = 0.25, ds=0.1, eps = 0.01, maxIter=1e3):
        # nlp = NLP(overallN, self.Q, self.R, self.Qf, self.goal, self.dt, self.bx, self.bu, self.spline, self.printLevel)
        # nlp.solve(self.start)
        # return nlp.xPred, nlp.uPred, nlp.feasible
        
        # sVals = list(np.arange(0, self.spline.end, ds))
        
        # Placeholders
        distLeft = 1
        xPred = []
        uPred = []
        
        count = 0
        xTraj = [self.start]
        uTraj = []
        
        # plt.ion()
        # fig, ax = plt.subplots()
        # fig2, ax2 = plt.subplots()
        # fig3, ax3 = plt.subplots()
        # fig4, ax4 = plt.subplots()
        # fig5, ax5 = plt.subplots()
        
        splineLine = []
        for s in np.linspace(0, self.spline.end, 1000, endpoint=False):
            splineLine.append(self.spline.calc_position(s))
        splineLine = np.array(splineLine)
    
        # ax.plot(splineLine[:,0], splineLine[:,1], '--oy', label='Spline')
        
        count = 0
        # while not sufficiently close to goal state and < max iterations:
        while distLeft > eps and count < maxIter:        
            currState = xTraj[-1]
            currS = currState[0]
            currY = currState[1]
            currV = currState[2]
            currTheta = currState[3]
            try:
                thetaTarget = self.spline.calc_yaw(currS)
            except:
                print('Error in yaw')
                pdb.set_trace()
            # fractionDone = currS / self.spline.end

            # if fractionDone <= rampLength:
            #     alpha = max(0.1, fractionDone / rampLength)
            #     vTarget = self.start[2] * (1-alpha) + alpha * cruise_velocity
            # elif fractionDone >= 1 - rampLength:
            #     alpha = (1-fractionDone) / rampLength
            #     vTarget = self.goal[2] * (1-alpha) + alpha * cruise_velocity
            # else:
            #     vTarget = cruise_velocity
            
            # Add a slight vertex offset left
            vTarget = - cruise_velocity / (self.spline.end / 2)**2 * (currS - self.spline.end/2 + 1e-3 * self.spline.end)**2 + cruise_velocity
            
            a = Kvals[0] * (vTarget - currV) + Kvals[3] * np.abs(currY)
            thetaDot = Kvals[1] * (thetaTarget - currTheta) + Kvals[2] * (-currY)
            
            thetaDot = np.clip(thetaDot, self.theta_dotMin, self.theta_dotMax)
            a = np.clip(a, self.amin, self.amax)
            
            u = np.array([a, thetaDot])
            xNext = self.dynamics(currState, u)
            
            xTraj.append(xNext)
            uTraj.append(u)
            
            distLeft = 1 - currS / self.spline.end
            
            count += 1
            x_1, x_2 = self.spline.calcXY(currS, currY)
            # ax.scatter(x_1, x_2, c='r', s=100)
            # ax2.scatter(count, thetaDot)
            # ax3.scatter(count, a)
            # ax4.scatter(count, currY)
            # ax5.scatter(count, self.spline.calc_curvature(currS))
            # if count % 500 == 0:                            
            #     ax.set_title('Count = ' + str(count))
            #     ax2.set_title('ThetaDot ' + str(count))
            #     ax3.set_title('a ' + str(count))
            #     ax4.set_title('currY ' + str(count))
            #     fig.canvas.draw()
            #     fig2.canvas.draw()
            #     fig3.canvas.draw()
            #     fig4.canvas.draw()
            #     pdb.set_trace()
                
        pdb.set_trace()
        
        return xTraj, uTraj
    
    def visualizeElapsedTrajectory(self, xTrajList, terminalPointsList, counts):
        n = len(xTrajList)
        fig, axes = plt.subplots(n) 
        
        splineLine = []
        for s in np.linspace(0, self.spline.end, 1000, endpoint=False):
            splineLine.append(self.spline.calc_position(s))
        splineLine = np.array(splineLine)
        
        for i in range(n):
            ax = axes[i]
            xTraj = np.copy(xTrajList[i])
            terminalPoints = np.copy(terminalPointsList[i])
            
            # Plot the spline
            ax.plot(splineLine[:,0], splineLine[:,1], '--r', label='Spline')
        
            # Plot the terminal points
            terminalXY = self.convertToXY(terminalPoints.T)
            ax.scatter(terminalXY[:,0], terminalXY[:,1], color='k', s=50, label='Terminal Points')
                
            # Plot the elapsed trajectory
            xyCoords = self.convertToXY(xTraj)
            ax.plot(xyCoords[:,0], xyCoords[:,1], '--ob', label='Elapsed')
            
            if i == 0:
                ax.legend()
            # ax.set_xlabel('x')
            # ax.set_ylabel('y')
            # ax.set_title('Count = ' + str(counts[i]))
        
        fig.suptitle('Closed-Loop Trajectory')
        # fig.tight_layout()
        
        return fig, axes
    
    def visualizeHeatmapTrajectory(self, xTraj, vMax):
        fig, ax = plt.subplots() 

        splineLine = []
        for s in np.linspace(0, self.spline.end, 1000, endpoint=False):
            splineLine.append(self.spline.calc_position(s))
        splineLine = np.array(splineLine)
    
        # Plot the spline
        ax.plot(splineLine[:,0], splineLine[:,1], '--r', label='Spline')
        
        velocities = np.array(xTraj)[:,2]
        
        # Plot the elapsed trajectory
        xyCoords = self.convertToXY(xTraj)
        cm = plt.cm.get_cmap('viridis')
        sc = ax.scatter(xyCoords[:,0], xyCoords[:,1], c=velocities, cmap=cm, label='Traj')
        plt.colorbar(sc)
        
        ax.legend()
            
        return fig, ax