import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import pdb

def xDynamics(curr_state, acc, theta_dot, dt):
    new_ang = theta_dot * dt + curr_state[3]
    new_vel = acc*dt + curr_state[2]
    new_x = curr_state[2] * dt * math.cos(curr_state[3]) + curr_state[0]
    new_y = curr_state[2] * dt * math.sin(curr_state[3]) + curr_state[1]
    
    #pdb.set_trace()
    return [new_x, new_y, new_vel, new_ang]

def sDynamics(x, u, dt, spline):
    # state = [s, y, v, theta]
    # input = [acc, theta_dot]
    # use Euler discretization
    try:
        gamma = spline.calc_yaw(x[0])
        curvature = spline.calc_curvature(x[0])
    except:
        pdb.set_trace()
    deltaS = x[2] * math.cos(x[3] - gamma) / (1 - gamma * curvature)
    deltaY = x[2] * math.sin(x[3] - gamma)
    s_next      = x[0] + dt * deltaS
    y_next      = x[1] + dt * deltaY
    v_next      = x[2] + dt * u[0]
    theta_next  = x[3] + dt * u[1]

    state_next = [s_next, y_next, v_next, theta_next]

    return state_next

if __name__ == '__main__': 
    xDemo = np.load('xDemo.npy')
    uDemo = np.load('uDemo.npy')
    
    dt = 0.1
    
    with open('spline.pkl', 'rb') as input:
        spline = pickle.load(input)
     
    # xDemo, uDemo start in the s-space
    errors = np.zeros((len(xDemo),2))
    for ind in range(len(xDemo)):
        sBasedX = xDemo[ind]
        u = uDemo[ind]
        xBasedX = np.copy(sBasedX)
        x, y = spline.calcXY(sBasedX[0], sBasedX[1])
        xBasedX[0] = x
        xBasedX[1] = y
        nextViaX = xDynamics(xBasedX, u[0], u[1], dt)
        nextViaS = sDynamics(sBasedX, u, dt, spline)
        nextViaS[0], nextViaS[1] = spline.calcXY(nextViaS[0], nextViaS[1])
        error = np.array([nextViaS[0] - nextViaX[0], nextViaS[1] - nextViaX[1]])
        errors[ind,:] = error
        