# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:37:15 2021

@author: zousa
"""
from body import Body
import math
import pdb
import random 

# TODO: Check that pid does not run into obstacles
# TODO: Make sure pid ranges is within workspace

def next_state(curr_state, acc, theta_dot, dt):
    new_ang = theta_dot * dt + curr_state[2]
    new_vel = acc*dt + curr_state[1]
    avg_vel = (new_vel + curr_state[1])/2
    avg_theta = (new_ang + curr_state[2])/2
    new_x = avg_vel * dt * math.sin(avg_theta) + curr_state[0][0]
    new_y = avg_vel * dt * math.cos(avg_theta) + curr_state[0][1]
    return [(new_x, new_y), new_vel, new_ang]

def dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx**2 + dy**2)

def pid(body: Body, org_path, dt):
    # state:
    #   0 - position (x, y)
    #   1 - velocity
    #   2 - heading angle (w.r.t to y axis)
    # ang want a right turn to be pi/2 rad and left turn pi/2
    # state evolves with the dynamics of the system, fully characterize the history
    # also =need inputs
    # state + input don't need to know the past to predict the future
    # how to determine time step? as small as you can with limit of computation time
    path = org_path.copy()
    curr_pt = path.pop(0)
    next_pt = path.pop(0)
    states = []
    inputs = []
    states.append([curr_pt, 1, 0])
    while dist(curr_pt, body.end) > dt:
        #add in state for every dt
        if dist(curr_pt, next_pt) < dt: #is within a time step
            next_pt = path.pop(0)
        else:
            curr_state = states[-1]
            dx = next_pt[0] - curr_pt[0]
            dy = next_pt[1] - curr_pt[1]
            acc = 0
            if (dy == 0):
                if dx > 0:
                    ang = math.pi/2
                elif dx < 0:
                    ang = - math.pi/2
            else:
                ang = math.atan(dx/dy)
            if (dy < 0):
                ang = ang + math.pi
            d_ang = (ang - curr_state[2]) / dt + random.gauss(0, math.pi)
            if abs(d_ang) > body.heading_ang/dt:
                d_ang = math.copysign(body.heading_ang/dt, d_ang)
            inputs.append([acc, d_ang])
            states.append(next_state(curr_state, acc, d_ang, dt))
            curr_pt = states[-1][0]
           
    return inputs, states

# how to add variation of PID
# look at Ugo's Github to add noise
# need to change the goal state 


            
