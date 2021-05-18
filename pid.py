# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:37:15 2021

@author: zousa
"""
from body import Body

def pid(body: Body, rrt: RRT, vel, dt):
    # state:
    #   position (x, y)
    #   velocity
    #   heading angle
    #   DOES THE ROBOT NEED TO STOP BEFORE CHANGING DIRECTIONS
    # NOTE: Ask Ugo what it should return 
    path = rrt.planning()
    start = body.start
    states = []
    for pt in path:
        pass
    return states
        
