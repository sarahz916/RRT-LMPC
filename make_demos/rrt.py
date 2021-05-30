# -*- coding: utf-8 -*-
"""
Created on Sat May 15 21:25:08 2021

@author: zousa
"""
# RRT Code for CS 159 Final Project
# Code based off of author: AtsushiSakai(@Atsushi_twi) from Python Robotics
# Modified by Sarah Zou

from body import Body
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pdb

show_animation = False

# TODO: take max_acc, and max_theta_dot into consideration

#NOTE: expand_distance might be dependent on max_acc and max_theta_dot of body
class RRT():
    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None
            
    # def __init__(self, body: Body, max_iter, goal_sample_rate, expand_dis, 
    #              path_resolution, bubbleDist, dt):
    def __init__(self, body: Body, max_iter, goal_sample_rate, expand_dis, 
             path_resolution, bubbleDist):
        self.start = self.Node(body.start[0], body.start[1])
        self.end = self.Node(body.end[0], body.end[1])
        self.min_rand = 0
        self.max_rand = body.max_x
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = body.obs_list
        self.node_list = []
        self.bubbleDist = bubbleDist
        #self.max_theta = body.max_theta_dot * dt
    
    def planning(self, animation=False):
        """
        rrt path planning
        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            # Note that this will check for collision along the entire path
            if self.check_collision(new_node, self.obstacle_list, self.bubbleDist):
                _, ang = self.calc_distance_and_angle(nearest_node, new_node)
                #if abs(ang) < self.max_theta:
                self.node_list.append(new_node)

            if animation:
                self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].x,
                                      self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end,
                                        self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list):
                    semi_final_path = self.generate_final_course(len(self.node_list) - 1)
                    # going to refine the final path
                    # return self.clean_final_path(semi_final_path)
                    return semi_final_path
                
        return None  # cannot find path
    
    def clean_final_path(self, semi_final_path: list):
        # Written by Sarah
        # want to iterate through the paths and cut out nodes that aren't needed
        # NOTE: since final node is not on path, cannot make last part of path 
        # more efficient
        final_path = []
        i = 2
        while i < len(semi_final_path):
            prev = semi_final_path[i]
            final_path.append(prev)
            if i + 2 >= len(semi_final_path):
                i += 1
                continue
            curr = semi_final_path[i + 2] # need to make sure 
            # see if prev and curr can be connected
            # if yes then connect
            new_node = self.steer(self.Node(prev[0], prev[1]), self.Node(curr[0], curr[1]), self.expand_dis)
            if self.check_collision(new_node, self.obstacle_list, self.bubbleDist):
                final_path.append(curr)
                i  = i+3
            else:
                i += 1
            
        return final_path

    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node

        return new_node

    # Aaron updated to avoide duplicating points 
    # and don't put in goal anymore
    # add_inter = False
    def generate_final_course(self, goal_ind):
        # Aaron changed so don't add the goal index
        # path = []
        path = [(self.end.x, self.end.y)]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            # if add_inter:
            #     # Exclude the start node so no repeats then flip
            #     xPart = node.path_x[1:][::-1]
            #     yPart = node.path_y[1:][::-1]
            #     path.extend(list(zip(xPart, yPart)))
            # else:
            path.append((node.x, node.y))
            node = node.parent

        # Now, add the start node
        path.append((node.x, node.y))
        # Aaron added reversing the path
        return path[::-1]

    # def generate_final_course(self, goal_ind, add_inter = True):
    #     path = [(self.end.x, self.end.y)]
    #     node = self.node_list[goal_ind]
    #     while node.parent is not None:
    #         if add_inter:
    #             path.extend(list(zip(node.path_x[::-1], node.path_y[::-1])))
    #         else:
    #             path.append((node.x, node.y))
    #         node = node.parent

    #     # Aaron added reversing the path
    #     return path[::-1]

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([0, self.max_rand, 0, self.max_rand])
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_collision(node, obstacleList, bubbleDist=0):

        if node is None:
            return False

        for (ox, oy, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]
            
            if min(d_list) <= (size + bubbleDist) **2:
                return False  # collision

        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta


        
    