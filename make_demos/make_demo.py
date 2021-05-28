# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:37:15 2021

@author: zousa
"""
from body import Body
import math
import pdb
import random 
from utils import system
import numpy as np
from ftocp import FTOCP
from nlp import NLP
from numpy import linalg as la

# TODO: Check that pid does not run into obstacles
# TODO: Make sure robot stays within workspace

def next_state(curr_state, acc, theta_dot, dt):
    new_ang = theta_dot * dt + curr_state[3]
    new_vel = acc*dt + curr_state[2]
    avg_vel = (new_vel + curr_state[2])/2
    avg_theta = (new_ang + curr_state[3])/2
    new_x = avg_vel * dt * math.cos(avg_theta) + curr_state[0]
    new_y = avg_vel * dt * math.sin(avg_theta) + curr_state[1]
    #pdb.set_trace()
    return [new_x, new_y, new_vel, new_ang]

def dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx**2 + dy**2)

def calc_input(body, curr_state, tar_state, dt):
    d_ang = (tar_state[3] - curr_state[3]) / dt 
    # if abs(d_ang) > body.max_theta_dot:
    #     d_ang = math.copysign(body.max_theta_dot, d_ang)
    d_v = (tar_state[2] - curr_state[2])/dt
    if abs(d_v) > body.max_acc:
        d_v = math.copysign(body.max_acc, d_v)
    return [d_v, d_ang]
    
def calc_angle(curr_pt, next_pt):
    dx = next_pt[0] - curr_pt[0]
    dy = next_pt[1] - curr_pt[1]
    if (dx == 0):
        if dy > 0:
            ang = math.pi/2
        elif dy < 0:
            ang = -math.pi/2
    else:
        ang = math.atan(dy/dx)
    if (dx < 0):
        ang = ang + math.pi
    return ang

def nlp_to_end(body, curr_state,dt):
    x0 = np.array(curr_state)
    goal = np.array(body.end)
    N  = 40; n = 4; d = 2;
    Q  = 1*np.eye(n)
    R  = 1*np.eye(d)
    Qf = 1000*np.eye(n)
    printLevel = 0
    xub = np.array([body.max_x, body.max_x, body.max_y, body.max_y])
    uub = np.array([body.max_acc, body.max_theta_dot])
    nlp = NLP(N, Q, R, Qf, goal, dt, xub, uub, printLevel)
    nlp.solve(x0)
    f = nlp.feasible
    return nlp.xPred, nlp.uPred, f

def make_demo(body: Body, org_path: list, dt, target_velocity = 1, tol = .0001):
    # state:
    #   0 - position x
    #   1 - position y
    #   2 - velocity
    #   3 - heading angle (w.r.t to y axis)
    # ang want a right turn to be pi/2 rad and left turn pi/2
    # state evolves with the dynamics of the system, fully characterize the history
    path = org_path.copy()
    curr_pt = path.pop(0)
    next_pt = path.pop(0)
    states = []
    inputs = []
    states.append(body.start)
    while len(path) >= 1:
        curr_pt = (states[-1][0], states[-1][1])
        curr_state = states[-1]
        if dist(curr_pt, next_pt) < dt * curr_state[2]: #is within a time step
            next_pt = path.pop(0)
        else:
            ang = calc_angle(curr_pt, next_pt)
            ang = ang + random.gauss(0, math.pi/20) #add noise 
            tar_state = [next_pt[0], next_pt[1], target_velocity, ang]
            next_input = calc_input(body, curr_state, tar_state, dt)
            inputs.append(next_input)
            states.append(next_state(curr_state, next_input[0], next_input[1], dt))
            #pdb.set_trace()
            
    # go to end state
    # USE FTOCOP to get to end state
    curr_state = states[-1]
    inputs = np.array(inputs)
    states = np.array(states)
    x_pred, u_pred, f = nlp_to_end(body, curr_state, dt)
    states = np.append(states, x_pred[1:], axis=0)
    inputs = np.append(inputs, u_pred, axis=0)
    return inputs, states, f



            
