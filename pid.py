# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:37:15 2021

@author: zousa
"""
from body import Body
from rrt import RRT

def pid(body: Body, rrt: RRT, vel, dt):
    # state:
    #   position (x, y)
    #   velocity
    #   heading angle
    #   DOES THE ROBOT NEED TO STOP BEFORE CHANGING DIRECTIONS
    # NOTE: Ask Ugo what it should return 
    # ang want a right turn to be pi/2 rad and left turn pi/2
    path = rrt.planning()
    curr_pt = path.pop(0)
    next_pt = path.pop(0)
    states = []
    states.append([curr_pt, 0, 0])
    while curr_pt != body.end:
        #add in state for every dt
        if curr_pt == next_pt:
            next_pt = path.pop(0)
        else:
            #NOTE: need to keep track of relative ang
            old_vel = states[-1][1]
            old_ang = states[-1][2]
            curr_pt = (old_vel * dt * cos(old_ang), old_vel * dt * sin(old_ang))
            #need to get ang relative to previous ang
            dx = next_pt[0] - curr_pt[0]
            dy = next_pt[1] - curr_pt[1]
            ang = math.atan(dx/dy)
            d_ang = ang - old_ang
            if math.abs(d_ang) > body.heading_ang:
                d_ang = math.copysign(body.heading_ang, d_ang)
            states.append([curr_pt, vel, d_ang])
    return states
        
