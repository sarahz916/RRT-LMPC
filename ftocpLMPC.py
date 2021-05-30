import pdb
import numpy as np
from cvxopt import spmatrix, matrix, solvers
from numpy import linalg as la
from scipy import linalg
from scipy import sparse
from cvxopt.solvers import qp
import datetime
from numpy import hstack, inf, ones
from scipy.sparse import vstack
from osqp import OSQP
from dataclasses import dataclass, field
# from casadi import sin, cos, SX, vertcat, Function, jacobian
from math import sin, cos

_epsilon = np.sqrt(np.finfo(float).eps)

def approx_jacobian(x, func, epsilon, *args):
    """Approximate the Jacobian matrix of callable function func

       * Parameters
         x       - The state vector at which the Jacobian matrix is
desired
         func    - A vector-valued function of the form f(x,*args)
         epsilon - The peturbation used to determine the partial derivatives
         *args   - Additional arguments passed to func

       * Returns
         An array of dimensions (lenf, lenx) where lenf is the length
         of the outputs of func, and lenx is the number of

       * Notes
         The approximation is done using forward differences

    """
    x0 = np.asfarray(x)
    f0 = func(*((x0,)+args))
    jac = np.zeros([len(x0),len(f0)])
    dx = np.zeros(len(x0))
    for i in range(len(x0)):
        dx[i] = epsilon
        allArgs = (x0+dx,)+args
        jac[i] = (np.array(func(*allArgs)) - np.array(f0))/epsilon
        dx[i] = 0.0
    return jac.transpose()

class FTOCP(object):
    """ Finite Time Optimal Control Problem (FTOCP)
    Methods:
        - solve: solves the FTOCP given the initial condition x0 and terminal contraints
        - buildNonlinearProgram: builds the ftocp program solved by the above solve method
        - model: given x_t and u_t computes x_{t+1} = f( x_t, u_t )
    """

    def __init__(self, N, Q, R, Fx, bx, Fu, bu, terminalPoints, valuePoints, spline, dt, uGuess, goal, printLevel):
        # Define variables
        self.printLevel = printLevel

        # Add spline so that can use for system dynamics
        self.spline = spline
    
        self.N  = N
        self.n  = Q.shape[1]
        self.d  = R.shape[1]
        # terminalPoints is an array of shape self.n x self.k
        self.k = terminalPoints.shape[1]
        self.Fx = Fx
        self.bx = bx
        self.Fu = Fu
        self.bu = bu
        self.Ff = Fx
        self.bf = bx
        # Replace with terminal component list
        self.terminalPoints = terminalPoints
        self.Q  = Q
        # self.Qf = Qf
        # Replace with value function approximation at the terminal points
        # Should be a vector of shape self.k
        self.valuePoints = valuePoints
        self.R  = R
        self.dt = dt
        self.uGuess = uGuess
        self.goal = goal
        
        self.buildIneqConstr()
        # self.buildAutomaticDifferentiationTree()
        self.buildCost()

        self.time = 0

    def simForward(self, x0, uGuess):
        self.xGuess = [x0]
        for i in range(0, self.N):
            xt = self.xGuess[i]
            ut = self.uGuess[i]
            self.xGuess.append(np.array(self.dynamics( xt, ut )))

    def solve(self, x0):
        """Computes control action
        Arguments:
            x0: current state
        """
        startTimer = datetime.datetime.now()
        self.simForward(x0, self.uGuess)
        self.buildEqConstr()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.linearizationTime = deltaTimer

        # Solve QP
        startTimer = datetime.datetime.now()
        
        G = self.G_in
        A = self.G_eq
        
        self.osqp_solve_qp(self.H, self.q, self.G_in, np.add(self.w_in, np.dot(self.E_in,x0)), self.G_eq, np.add(np.dot(self.E_eq,x0), self.C_eq) )
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
		
        # Unpack Solution
        self.unpackSolution(x0)
        self.time += 1

        return self.uPred[0,:]

    # Aaron: modify the uGuess using the past trajectory so that can repeat and
    # solve via SQP
    def uGuessUpdate(self):
        uPred = self.uPred
        for i in range(0, self.N):
            self.uGuess[i] = uPred[i]
        
    # Potentially round s-values that are negative but close to 0 to 0 and 
    # similarly for v-values
    def unpackSolution(self, x0, roundIt = True):
        # Extract predicted state and predicted input trajectories
        self.xPred = np.vstack((x0, np.reshape((self.Solution[np.arange(self.n*(self.N))]),(self.N,self.n))))
        self.uPred = np.reshape((self.Solution[self.n*(self.N)+np.arange(self.d*self.N)]),(self.N, self.d))
        
        if roundIt:
            print('Called roundIt')
            self.xPred[np.where((self.xPred[:,0] < 0) * (np.abs(self.xPred[:,0]) < 1e-5)),0] = 0 
            self.xPred[np.where((self.xPred[:,2] < 0) * (np.abs(self.xPred[:,2]) < 1e-5)),2] = 0
        if self.printLevel >= 2:
            print("Predicted State Trajectory: ")
            print(self.xPred)

            print("Predicted Input Trajectory: ")
            print(self.uPred)

        if self.printLevel >= 1: 
            print("Linearization + buildEqConstr() Time: ", self.linearizationTime.total_seconds(), " seconds.")
            print("Solver Time: ", self.solverTime.total_seconds(), " seconds.")

    def buildIneqConstr(self):
        # The inequality constraint is Gin z<= win + Ein x0
        rep_a = [self.Fx] * (self.N-1)
        Mat   = linalg.block_diag(linalg.block_diag(*rep_a), self.Ff)
        Fxtot = np.vstack((np.zeros((self.Fx.shape[0], self.n*self.N)), Mat))
        bxtot = np.append(np.tile(np.squeeze(self.bx), self.N), self.bf)
        
        rep_b = [self.Fu] * (self.N)
        Futot = linalg.block_diag(*rep_b)
        butot = np.tile(np.squeeze(self.bu), self.N)

        G_in = linalg.block_diag(Fxtot, Futot)
        E_in = np.zeros((G_in.shape[0], self.n))
        E_in[0:self.Fx.shape[0], 0:self.n] = -self.Fx
        w_in = np.hstack((bxtot, butot))
        
        # Horizontally stack G_in with [0 -I] to account for lambda
        largeG = np.zeros((G_in.shape[0]+self.k, G_in.shape[1]+self.k))
        largeG[:G_in.shape[0],:G_in.shape[1]] = G_in
        largeG[-self.k:, -self.k:] = -np.eye(self.k)
        G_in = largeG
        
        # Add 0 block below for lambda
        E_in = np.vstack([E_in, np.zeros((self.k, E_in.shape[1]))])
        
        # Add 0 block for lambda
        w_in = np.hstack([w_in, np.zeros(self.k)])
                
        if self.printLevel >= 2:
            print("G_in: ")
            print(G_in.shape)
            print("E_in: ")
            print(E_in)
            print("w_in: ", w_in)			

        self.G_in = sparse.csc_matrix(G_in)
        self.E_in = E_in
        self.w_in = w_in.T

    def buildCost(self):
        listQ = [self.Q] * (self.N-1)
        # Like Qf = 0
        barQ = linalg.block_diag(linalg.block_diag(*listQ), 0 * np.eye(len(self.Q)))

        listTotR = [self.R] * (self.N)
        barR = linalg.block_diag(*listTotR)

        H = linalg.block_diag(barQ, barR)
        		
        # Hint: First construct a vector z_{goal} using the goal state and then leverage the matrix H
		
        # Aaron added ###
        zGoal = np.hstack([self.goal]*self.N + [np.zeros(self.d * self.N)])
        q = -2 * H @ zGoal

        # Now, embed H into larger 0 matrix to account for lambdas
        # Overall size of H should be self.N * self.n + self.N * self.d + self.k 
        totSize = self.N * self.n + self.N * self.d + self.k
        largeH = np.zeros((totSize, totSize))
        largeH[:H.shape[0],:H.shape[1]] = H
        H = largeH
        
        # Now, extend q, should have length totSize and account for J
        largeq = np.zeros(totSize)
        largeq[:q.shape[0]] = q
        largeq[q.shape[0]:] = self.valuePoints
        q = largeq
        
        ######

        if self.printLevel >= 2:
            print("H: ")
            print(H)
            print("q: ", q)
		
        self.q = q
        self.H = sparse.csc_matrix(2 * H)  #  Need to multiply by two because CVX considers 1/2 in front of quadratic cost

    def buildEqConstr(self):
        # Hint 1: The equality constraint is: [Gx, Gu]*z = E * x(t) + C
        # Hint 2: Write on paper the matrices Gx and Gu to see how these matrices are constructed
        Gx = np.eye(self.n * self.N )
        Gu = np.zeros((self.n * self.N, self.d*self.N) )

        self.C = []
        E_eq = np.zeros((Gx.shape[0], self.n))
        for k in range(0, self.N):
            A, B, C = self.buildLinearizedMatrices(self.xGuess[k], self.uGuess[k])
            if k == 0:
                E_eq[0:self.n, :] = A
            else:
                # Aaron added, should be on the subdiagonal
                Gx[k*self.n:(k+1)*self.n, (k-1)*self.n:k*self.n] = -A
            # Aaron added, should be on the diagonal
            Gu[k*self.n:(k+1)*self.n, k*self.d:(k+1)*self.d] = -B
            self.C = np.append(self.C, C)

        G_eq = np.hstack((Gx, Gu))
        C_eq = self.C
        
        # Modify G_eq to account for lambda
        largeG = np.zeros((G_eq.shape[0]+self.n+1, G_eq.shape[1]+self.k))
        largeG[:G_eq.shape[0], :G_eq.shape[1]] = G_eq
        largeG[G_eq.shape[0] + 1:, self.n * (self.N-1): self.n * self.N] = -np.eye(self.n)
        largeG[G_eq.shape[0] + 1:, -self.k:] = self.terminalPoints
        largeG[-1, -self.k:] = 1 # Adds constraint that sum of lambdas = 1
        G_eq = largeG
        
        # Modify E_eq, C_eq to account for lambda
        E_eq = np.vstack([E_eq, np.zeros((self.n + 1, E_eq.shape[1]))])
        #C_eq = np.vstack([C_eq.T, np.zeros((self.k,))]) #C_eq.shape is (4, ) 
        C_eq = np.concatenate((C_eq, np.zeros((self.n + 1,))), axis=0)
                
        if self.printLevel >= 2:
            print("G_eq: ")
            print(G_eq.shape)
            print("E_eq: ")
            print(E_eq)
            print("C_eq: ", C_eq)

        self.C_eq = C_eq
        self.G_eq = sparse.csc_matrix(G_eq)
        self.E_eq = E_eq
        
    # def buildAutomaticDifferentiationTree(self):
    #     # Define variables
    #     n  = self.n
    #     d  = self.d
    #     X      = SX.sym('X', n);
    #     U      = SX.sym('U', d);

    #     X_next = self.dynamics(X, U)
    #     self.constraint = []
    #     for i in range(0, n):
    #         self.constraint = vertcat(self.constraint, X_next[i] )

    #     self.A_Eval = Function('A',[X,U],[jacobian(self.constraint,X)])
    #     self.B_Eval = Function('B',[X,U],[jacobian(self.constraint,U)])
    #     self.f_Eval = Function('f',[X,U],[self.constraint])
	
    # def buildLinearizedMatrices(self, x, u):
    #     # Give a linearization point (x, u) this function return an affine approximation of the nonlinear system dynamics
    #     A_linearized = np.array(self.A_Eval(x, u))
    #     B_linearized = np.array(self.B_Eval(x, u))
    #     C_linearized = np.squeeze(np.array(self.f_Eval(x, u))) - np.dot(A_linearized, x) - np.dot(B_linearized, u)
		
    #     if self.printLevel >= 3:
    #         print("Linearization x: ", x)
    #         print("Linearization u: ", u)
    #         print("Linearized A")
    #         print(A_linearized)
    #         print("Linearized B")
    #         print(B_linearized)
    #         print("Linearized C")
    #         print(C_linearized)

    #     return A_linearized, B_linearized, C_linearized

    def buildLinearizedMatrices(self, x, u):
        # s = x[0]
        # y = x[1]
        # v = x[2]
        # theta = x[3]
        # gamma = self.spline.calc_yaw(s)
        # k = self.spline.calc_curvature(s)
        # kPrime = self.spline.calc_curvaturePrime(s)
        # gammaPrime = self.spline.calc_yawPrime(s)
        # # d/ds [ cos(theta - gamm) / (1- gamma K)]
        # num = sin(theta - gamma) * gammaPrime * (1 - gamma * k) + \
        #     cos(theta - gamma) * (gammaPrime * k + kPrime * gamma)
        # den = (1 - gamma * k)**2
        
        # A = np.zeros((4,4))
        
        # # s derivatives
        # A[0,0] = 1 + v * self.dt * num/den
        # A[1,0] = - v * self.dt * cos(theta - gamma) * gammaPrime 
        # A[2,0] = 0
        # A[3,0] = 0
        
        # # y derivatives
        # A[0,1] = 0
        # A[1,1] = 1
        # A[2,1] = 0
        # A[3,1] = 0
        
        # # v derivatives
        # A[0,2] = self.dt * cos(theta - gamma) / (1-gamma * k)
        # A[1,2] = self.dt * sin(theta - gamma)
        # A[2,2] = 1
        # A[3,2] = 0
        
        # # theta derivatives
        # A[0,3] = - self.dt * v * sin(theta - gamma) / (1 - gamma * k)
        # A[1,3] = self.dt * v * cos(theta - gamma)
        # A[2,3] = 0
        # A[3,3] = 1
        
        # B = np.zeros((4,2))
        
        # B[2,0] = self.dt
        # B[3,1] = self.dt
            
        # C = self.dynamics(x, u) - A @ x - B @ u
            
        # return A, B, C
                                
        merged = np.array(list(x) + list(u))
        numJ = approx_jacobian(merged, lambda x: self.dynamics(x[:4], x[4:]), _epsilon)
        
        numA = numJ[:,:4]
        numB = numJ[:, 4:]
        numC = self.dynamics(x, u) - numA @ x - numB @ u
            
        return numA, numB, numC
    
    def osqp_solve_qp(self, P, q, G= None, h=None, A=None, b=None, initvals=None):
        """ 
        Solve a Quadratic Program defined as:
        minimize
        (1/2) * x.T * P * x + q.T * x
        subject to
			G * x <= h
			A * x == b
        using OSQP <https://github.com/oxfordcontrol/osqp>.
        """  
		
        qp_A = vstack([G, A]).tocsc()
        l = -inf * ones(len(h))
        qp_l = hstack([l, b])
        qp_u = hstack([h, b])
        
        self.osqp = OSQP() 
        self.osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)

        if initvals is not None:
            self.osqp.warm_start(x=initvals)
        res = self.osqp.solve()
        if res.info.status_val == 1:
            self.feasible = 1
        else:
            self.feasible = 0
            print("The FTOCP is not feasible at time t = ", self.time)

        self.Solution = res.x

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
            new_x = curr_state[2] * self.dt * cos(curr_state[3]) + curr_state[0]
            new_y = curr_state[2] * self.dt * sin(curr_state[3]) + curr_state[1]
            # Convert back to (s,ey)
            new_s, new_ey = self.spline.calcSY(new_x, new_y)
            return [new_s, new_ey, new_vel, new_ang]
                
        else:
            # state = [s, y, v, theta]
            # input = [acc, theta_dot]
            # use Euler discretization
            gamma = self.spline.calc_yaw(x[0])
            curvature = self.spline.calc_curvature(x[0])
            deltaS = x[2] * cos(x[3] - gamma) / (1 - x[1] * curvature)
            deltaY = x[2] * sin(x[3] - gamma)
            s_next      = x[0] + self.dt * deltaS
            y_next      = x[1] + self.dt * deltaY
            v_next      = x[2] + self.dt * u[0]
            theta_next  = x[3] + self.dt * u[1]
            state_next = [s_next, y_next, v_next, theta_next]
            return state_next


    # def dynamics(self, x, u):
    #     # state = [s, y, v, theta]
    #     # input = [acc, theta_dot]
    #     # use Euler discretization
    #     try:
    #         gamma = self.spline.calc_yaw(x[0])
    #         curvature = self.spline.calc_curvature(x[0])
    #     except:
    #         pdb.set_trace()
    #     deltaS = x[2] * cos(x[3] - gamma) / (1 - x[1] * curvature)
    #     deltaY = x[2] * sin(x[3] - gamma)
    #     s_next      = x[0] + self.dt * deltaS
    #     y_next      = x[1] + self.dt * deltaY
    #     v_next      = x[2] + self.dt * u[0]
    #     theta_next  = x[3] + self.dt * u[1]

    #     state_next = [s_next, y_next, v_next, theta_next]

    #     return state_next