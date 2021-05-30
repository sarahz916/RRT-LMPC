import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb
from casadi import sin, cos, SX, vertcat, Function, jacobian
import numdifftools as nd

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
        jac[i] = (func(*allArgs) - f0)/epsilon
        dx[i] = 0.0
    return jac.transpose()

# Revised dynamics
def combinedDynamics(x, dt, spline):
    # state = [s, y, v, theta]
    # input = [acc, theta_dot]
    # use Euler discretization
    try:
        gamma = spline.calc_yaw(x[0])
        curvature = spline.calc_curvature(x[0])
    except:
        pdb.set_trace()
    deltaS = x[2] * cos(x[3] - gamma) / (1 - x[1] * curvature)
    deltaY = x[2] * sin(x[3] - gamma)
    s_next      = x[0] + dt * deltaS
    y_next      = x[1] + dt * deltaY
    v_next      = x[2] + dt * x[4]
    theta_next  = x[3] + dt * x[5]

    state_next = np.array([s_next, y_next, v_next, theta_next])

    return state_next

# Nonlinear dynamics using Euler discretization
def dynamics(x, u, dt, spline):
    # state = [s, y, v, theta]
    # input = [acc, theta_dot]
    # use Euler discretization
    try:
        gamma = spline.calc_yaw(x[0])
        curvature = spline.calc_curvature(x[0])
    except:
        pdb.set_trace()
    deltaS = x[2] * cos(x[3] - gamma) / (1 - gamma * curvature)
    deltaY = x[2] * sin(x[3] - gamma)
    s_next      = x[0] + dt * deltaS
    y_next      = x[1] + dt * deltaY
    v_next      = x[2] + dt * u[0]
    theta_next  = x[3] + dt * u[1]

    state_next = [s_next, y_next, v_next, theta_next]

    return state_next

# x, u are the current state and action to apply while xBar, uBar
# are the state, action pair used for linearization
def linearizedDynamics(x, u, xBar, uBar, dt, spline):
    A, B, C = buildLinearizedMatrices(xBar, uBar, dt, spline)
    return A @ x + B @ u + C

# Matrices for linearized dynamics
def buildLinearizedMatrices(x, u, dt, spline):
    s = x[0]
    y = x[1]
    v = x[2]
    theta = x[3]
    gamma = spline.calc_yaw(s)
    k = spline.calc_curvature(s)
    kPrime = spline.calc_curvaturePrime(s)
    gammaPrime = spline.calc_yawPrime(s)
    # d/ds [ cos(theta - gamma) / (1- gamma K)]
    num = sin(theta - gamma) * gammaPrime * (1 - gamma * k) + \
            cos(theta - gamma) * (gammaPrime * k + kPrime * gamma)
    den = (1 - gamma * k)**2
    
    A = np.zeros((4,4))
    
    # s derivatives
    A[0,0] = 1 + v * dt * num/den
    A[1,0] = - v * dt * cos(theta - gamma) * gammaPrime 
    A[2,0] = 0
    A[3,0] = 0
    
    # y derivatives
    A[0,1] = 0
    A[1,1] = 1
    A[2,1] = 0
    A[3,1] = 0
    
    # v derivatives
    A[0,2] = dt * cos(theta - gamma) / (1-gamma * k)
    A[1,2] = dt * sin(theta - gamma)
    A[2,2] = 1
    A[3,2] = 0
    
    # theta derivatives
    A[0,3] = - dt * v * sin(theta - gamma) / (1 - gamma * k)
    A[1,3] = dt * v * cos(theta - gamma)
    A[2,3] = 0
    A[3,3] = 1
    
    B = np.zeros((4,2))
    
    B[2,0] = dt
    B[3,1] = dt
        
    C = dynamics(x, u, dt, spline) - A @ x - B @ u
            
    merged = np.array(list(x) + list(u))
    numJ = approx_jacobian(merged, combinedDynamics, _epsilon, dt, spline)
    
    numA = numJ[:,:4]
    numB = numJ[:, 4:]
    numC = dynamics(x, u, dt, spline) - numA @ x - numB @ u
    
    pdb.set_trace()
    
    return numA, numB, numC

if __name__ == '__main__': 
    xDemo = np.load('xDemo.npy')
    uDemo = np.load('uDemo.npy')
    
    with open('spline.pkl', 'rb') as input:
        spline = pickle.load(input)
          
    # alphaVals = [0.1 * i for i in range(10)]
    alphaVals = [0, 0.5, 1]
    dtVals = [10**i for i in range(-2,2)]
    
    meanErrors = np.zeros((len(alphaVals), len(dtVals), 4))
    
    for i, alpha in enumerate(alphaVals):
        for j, dt in enumerate(dtVals):
            errors = np.zeros((len(xDemo)-1, 4))
            for k in range(1, len(xDemo)):
                xBar = xDemo[k]
                uBar = uDemo[k]
                x = xDemo[k-1] * alpha + (1-alpha) * xDemo[k]
                u = uDemo[k-1] * alpha + (1-alpha) * uDemo[k]
                trueNext = dynamics(x, u, dt, spline)
                linNext = linearizedDynamics(x, u, xBar, uBar, dt, spline)
                errors[k-1,:] = trueNext - linNext
            meanErrors[i,j,:] = np.mean(np.abs(errors), axis=0)

    names = ['s','y','v','theta']
    for l in range(4):
        plt.figure()
        plt.title(names[l] + ' absolute error')
        for m, alpha in enumerate(alphaVals):
            plt.scatter(dtVals, meanErrors[m, :, l], label=str(alpha)[:4])
        plt.legend()
        plt.xlabel('dt')
        plt.ylabel('Absolute error')
        
    
    