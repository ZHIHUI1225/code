#!/usr/bin/env python
# coding=utf-8

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import env
import matplotlib.patches as patches
import matplotlib as mpl 
mpl.rcParams['animation.ffmpeg_path'] = r"D:\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"

from matplotlib.patches import Rectangle
from IPython import display

class Draw_MPC_point_stabilization_v1(object):
    def __init__(self, robot_states: list, init_state: np.array, target_state: np.array, rob_diam,
                 export_fig=True):
        self.robot_states = robot_states
        self.init_state = init_state
        self.target_state = target_state
        self.rob_radius = rob_diam / 2.0
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(0, 200), ylim=(0, 200))
        self.obs_rectangle = [
             [125,80,40,30],
             [40,55,30,30]
        ]
        # self.fig.set_dpi(400)
        self.fig.set_size_inches(7, 6.5)
        # init for plot
        self.animation_init()

        self.ani = animation.FuncAnimation(self.fig, self.animation_loop, range(len(self.robot_states)),
                                           init_func=self.animation_init, interval=20, repeat=False)
        
        plt.grid('--')
        for (ox, oy, w, h) in self.obs_rectangle:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )
        plt.show()
        if export_fig:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=180)
            self.ani.save('v_single.mp4', writer=writer)
            
        

    def animation_init(self):
        # plot target state
        self.target_circle = plt.Circle(self.target_state[:2], self.rob_radius, color='b', fill=False)
        self.ax.add_artist(self.target_circle)
        self.target_arr = mpatches.Arrow(self.target_state[0], self.target_state[1],
                                         self.rob_radius * np.cos(self.target_state[2]),
                                         self.rob_radius * np.sin(self.target_state[2]), width=0.2)
        self.ax.add_patch(self.target_arr)
        self.robot_body = plt.Circle(self.init_state[:2], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.robot_body)
        self.robot_arr = mpatches.Arrow(self.init_state[0], self.init_state[1],
                                        self.rob_radius * np.cos(self.init_state[2]),
                                        self.rob_radius * np.sin(self.init_state[2]), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        return self.target_circle, self.target_arr, self.robot_body, self.robot_arr

    def animation_loop(self, indx):
        position = self.robot_states[indx][:2]
        orientation = self.robot_states[indx][2]
        self.robot_body.center = position
        # self.ax.add_artist(self.robot_body)
        self.robot_arr.remove()
        self.robot_arr = mpatches.Arrow(position[0], position[1], self.rob_radius * np.cos(orientation),
                                        self.rob_radius * np.sin(orientation), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        return self.robot_arr, self.robot_body


class Draw_MPC_Obstacle(object):
    def __init__(self, robot_states: list, init_state: np.array, target_state: np.array, obstacle: np.array,
                 rob_diam, export_fig=False):
        self.robot_states = robot_states
        self.init_state = init_state
        self.target_state = target_state
        self.rob_radius = rob_diam / 2.0
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(0,200), ylim=(0, 200))
        if obstacle is not None:
            self.obstacle = obstacle
        else:
            print('no obstacle given, break')
        self.fig.set_size_inches(7, 6.5)
        # init for plot
        self.animation_init()

        self.ani = animation.FuncAnimation(self.fig, self.animation_loop, range(len(self.robot_states)),
                                           init_func=self.animation_init, interval=100, repeat=False)

        plt.grid('--')
        if export_fig:
            self.ani.save('obstacle.gif', writer='imagemagick', fps=100)
        plt.show()

    def animation_init(self):
        # plot target state
        self.target_circle = plt.Circle(self.target_state[:2], self.rob_radius, color='b', fill=False)
        self.ax.add_artist(self.target_circle)
        self.target_arr = mpatches.Arrow(self.target_state[0], self.target_state[1],
                                         self.rob_radius * np.cos(self.target_state[2]),
                                         self.rob_radius * np.sin(self.target_state[2]), width=0.2)
        self.ax.add_patch(self.target_arr)
        self.robot_body = plt.Circle(self.init_state[:2], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.robot_body)
        self.robot_arr = mpatches.Arrow(self.init_state[0], self.init_state[1],
                                        self.rob_radius * np.cos(self.init_state[2]),
                                        self.rob_radius * np.sin(self.init_state[2]), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        self.obstacle_circle = plt.Circle(self.obstacle[:2], self.obstacle[2], color='g', fill=True)
        self.ax.add_artist(self.obstacle_circle)
        return self.target_circle, self.target_arr, self.robot_body, self.robot_arr, self.obstacle_circle

    def animation_loop(self, indx):
        position = self.robot_states[indx][:2]
        orientation = self.robot_states[indx][2]
        self.robot_body.center = position
        self.robot_arr.remove()
        self.robot_arr = mpatches.Arrow(position[0], position[1], self.rob_radius * np.cos(orientation),
                                        self.rob_radius * np.sin(orientation), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        return self.robot_arr, self.robot_body


class Draw_MPC_tracking(object):
    def __init__(self, robot_states: list, init_state: np.array, rob_diam, export_fig=False):
        self.init_state = init_state
        self.robot_states = robot_states
        self.rob_radius = rob_diam
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-1.0, 16), ylim=(-0.5, 1.5))
        # self.fig.set_size_inches(7, 6.5)
        # init for plot
        self.animation_init()

        self.ani = animation.FuncAnimation(self.fig, self.animation_loop, range(len(self.robot_states)),
                                           init_func=self.animation_init, interval=100, repeat=False)

        plt.grid('--')
        if export_fig:
            self.ani.save('tracking.gif', writer='imagemagick', fps=100)
        plt.show()

    def animation_init(self, ):
        # draw target line
        self.target_line = plt.plot([0, 12], [1, 1], '-r')
        # draw the initial position of the robot
        self.init_robot_position = plt.Circle(self.init_state[:2], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.init_robot_position)
        self.robot_body = plt.Circle(self.init_state[:2], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.robot_body)
        self.robot_arr = mpatches.Arrow(self.init_state[0], self.init_state[1],
                                        self.rob_radius * np.cos(self.init_state[2]),
                                        self.rob_radius * np.sin(self.init_state[2]), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        return self.target_line, self.init_robot_position, self.robot_body, self.robot_arr

    def animation_loop(self, indx):
        position = self.robot_states[indx][:2]
        orientation = self.robot_states[indx][2]
        self.robot_body.center = position
        self.robot_arr.remove()
        self.robot_arr = mpatches.Arrow(position[0], position[1], self.rob_radius * np.cos(orientation),
                                        self.rob_radius * np.sin(orientation), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        return self.robot_arr, self.robot_body

class Draw_MPC_two_agents_withtube(object):
    def __init__(self, robot_states: list, init_state: np.array, target_state: np.array, rob_diam,obs_rectangle,
                 export_fig=True,):
        self.robot_states = robot_states
        self.init_state = init_state
        self.target_state = target_state
        self.rob_radius = rob_diam / 2.0
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(0, 200), ylim=(0, 200))
        self.obs_rectangle = obs_rectangle
        # self.fig.set_dpi(400)
        self.fig.set_size_inches(7, 6.5)
        # init for plot
        self.animation_init()

        self.ani = animation.FuncAnimation(self.fig, self.animation_loop, range(len(self.robot_states)),
                                           init_func=self.animation_init, interval=20, repeat=False)
        
        plt.grid('--')
        for (ox, oy, w, h) in self.obs_rectangle:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )
        plt.show()
        if export_fig:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=180)
            self.ani.save('v_twoagent.mp4', writer=writer)
            

    def animation_init(self):
        # plot target state  
        self.target_circle = plt.Circle(self.target_state[-2:], 1, color='b', fill=True)
        self.ax.add_artist(self.target_circle)
        self.circle = plt.Circle(self.init_state[-2:], 1, color='g', fill=True)
        self.ax.add_artist(self.circle)
        self.robot1_body = plt.Circle(self.init_state[:2], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.robot1_body)
        self.robot1_arr = mpatches.Arrow(self.init_state[0], self.init_state[1],
                                        self.rob_radius * np.cos(self.init_state[2]),
                                        self.rob_radius * np.sin(self.init_state[2]), width=0.2, color='r')
        self.ax.add_patch(self.robot1_arr)
        self.robot2_body = plt.Circle(self.init_state[3:5], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.robot2_body)
        self.robot2_arr = mpatches.Arrow(self.init_state[3], self.init_state[4],
                                        self.rob_radius * np.cos(self.init_state[5]),
                                        self.rob_radius * np.sin(self.init_state[5]), width=0.2, color='r')
        self.ax.add_patch(self.robot2_arr)
        return self.target_circle, self.robot1_body, self.robot1_arr,self.robot2_body, self.robot2_arr

    def animation_loop(self, indx):
        # robot1
        position1 = self.robot_states[indx][:2]
        orientation1 = self.robot_states[indx][2]
        self.robot1_body.center = position1
        self.robot1_arr.remove()
        self.robot1_arr = mpatches.Arrow(position1[0], position1[1], self.rob_radius * np.cos(orientation1),
                                        self.rob_radius * np.sin(orientation1), width=0.2, color='r')
        self.ax.add_patch(self.robot1_arr)
        # robot2
        position2 = self.robot_states[indx][3:5]
        orientation2 = self.robot_states[indx][5]
        self.robot2_body.center = position2
        self.robot2_arr.remove()
        self.robot2_arr = mpatches.Arrow(position2[0], position2[1], self.rob_radius * np.cos(orientation2),
                                        self.rob_radius * np.sin(orientation2), width=0.2, color='r')
        self.ax.add_patch(self.robot2_arr)
        # the feature point of tube
        position = self.robot_states[indx][-2:]
        self.circle.center = position
        return self.robot1_arr, self.robot1_body,self.robot2_arr, self.robot2_body, self.circle
    
class Draw_MPC_two_agents_obstacles_Jerror(object):
    def __init__(self, robot_states: list, init_state: np.array, target_state: np.array, rob_diam,curve_state,obs_rectangle,obs_circle,J_error,Time,Target_circle,
                 export_fig=True,):
        self.robot_states = robot_states
        self.init_state = init_state
        self.target_state = target_state
        self.rob_radius = rob_diam / 2.0
        self.circle=[]
        self.obs_circle=obs_circle
        self.fig, (self.ax, self.ax2) =  plt.subplots(1, 2, figsize=(10, 4))
        self.ax.set_xlim(0, 200)
        self.ax.set_ylim(0, 200)
        self.ax.set_xlabel('x', fontsize=14)
        self.ax.set_ylabel('y', fontsize=14)
        self.ax2.set_xlim(0, max(Time))
        self.ax2.set_ylim(0, max(J_error))
        self.ax2.set_xlabel('t', fontsize=14)
        self.ax2.set_ylabel('error', fontsize=14)
        self.Time=Time
        self.Jerror=J_error
        self.Target_circle=Target_circle
        # self.fig = plt.figure()
        # self.ax = plt.axes(xlim=(0, 200), ylim=(0, 200))
        self.obs_rectangle = obs_rectangle
        self.curve_state=curve_state
        # self.fig.set_dpi(400)
        #self.fig.set_size_inches(7, 6.5)
        # init for plot
        self.animation_init()

        self.ani = animation.FuncAnimation(self.fig, self.animation_loop, range(len(self.robot_states)),
                                           init_func=self.animation_init, interval=20, repeat=False)
        
        plt.grid('--')
        for (ox, oy, w, h) in self.obs_rectangle:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )
        for (ox, oy, r) in self.obs_circle:
            self.ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )
        plt.show()
        if export_fig:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=180)
            self.ani.save('v_twoagent_obstacles.mp4', writer=writer)
            

    def animation_init(self):
        self.pulling=plt.Circle(self.Target_circle[:2],self.Target_circle[2],color='gray', fill=True)
        self.ax.add_artist(self.pulling)
        for i in range(int(len(self.target_state)/2)):
        # plot target state  
            self.target_circle = plt.Circle(self.target_state[i*2:2+i*2], 0.5, color='b', fill=True)
            self.ax.add_artist(self.target_circle)
            self.circle.append(plt.Circle(self.init_state[6+i*2:8+i*2], 0.5, color='g', fill=True))
            self.ax.add_artist(self.circle[i])
        self.robot1_body = plt.Circle(self.init_state[:2], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.robot1_body)
        self.robot1_arr = mpatches.Arrow(self.init_state[0], self.init_state[1],
                                        self.rob_radius * np.cos(self.init_state[2]),
                                        self.rob_radius * np.sin(self.init_state[2]), width=0.2, color='r')
        self.ax.add_patch(self.robot1_arr)
        self.robot2_body = plt.Circle(self.init_state[3:5], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.robot2_body)
        self.robot2_arr = mpatches.Arrow(self.init_state[3], self.init_state[4],
                                        self.rob_radius * np.cos(self.init_state[5]),
                                        self.rob_radius * np.sin(self.init_state[5]), width=0.2, color='r')
        self.ax.add_patch(self.robot2_arr)
        self.curve,=self.ax.plot(self.curve_state[0][0,:],self.curve_state[0][1,:])
        # J_error
        self.error,=self.ax2.plot(self.Time[0],self.Jerror[0])
        return self.target_circle, self.robot1_body, self.robot1_arr,self.robot2_body, self.robot2_arr,self.curve,self.error,self.pulling

    def animation_loop(self, indx):
        # robot1
        position1 = self.robot_states[indx][:2]
        orientation1 = self.robot_states[indx][2]
        self.robot1_body.center = position1
        self.robot1_arr.remove()
        self.robot1_arr = mpatches.Arrow(position1[0], position1[1], self.rob_radius * np.cos(orientation1),
                                        self.rob_radius * np.sin(orientation1), width=0.2, color='r')
        self.ax.add_patch(self.robot1_arr)
        # robot2
        position2 = self.robot_states[indx][3:5]
        orientation2 = self.robot_states[indx][5]
        self.robot2_body.center = position2
        self.robot2_arr.remove()
        self.robot2_arr = mpatches.Arrow(position2[0], position2[1], self.rob_radius * np.cos(orientation2),
                                        self.rob_radius * np.sin(orientation2), width=0.2, color='r')
        self.ax.add_patch(self.robot2_arr)
        #curve
        self.curve.set_ydata(self.curve_state[indx][1,:])
        self.curve.set_xdata(self.curve_state[indx][0,:])
        # the feature point of tube
        for i in range(int(len(self.target_state)/2)):
            position = self.robot_states[indx][6+i*2:8+i*2]
            self.circle[i].center = position
        #J_error_curve
        #self.error=self.ax2.plot(self.Time[:indx],self.Jerror[:indx])
        self.error.set_ydata(self.Jerror[:indx])
        self.error.set_xdata(self.Time[:indx])
        return self.robot1_arr, self.robot1_body,self.robot2_arr, self.robot2_body, self.circle,self.curve,self.ax2,self.error
    
class Draw_MPC_two_agents_obstacles(object):
    def __init__(self, robot_states: list, init_state: np.array, target_state: np.array, rob_diam,curve_state,obs_rectangle,obs_circle,
                    export_fig=True,):
        self.robot_states = robot_states
        self.init_state = init_state
        self.target_state = target_state
        self.rob_radius = rob_diam / 2.0
        self.circle=[]
        self.obs_circle=obs_circle
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(0, 200), ylim=(0, 200))
        self.obs_rectangle = obs_rectangle
        self.curve_state=curve_state
        # self.fig.set_dpi(400)
        self.fig.set_size_inches(7, 6.5)
        # init for plot
        self.animation_init()

        self.ani = animation.FuncAnimation(self.fig, self.animation_loop, range(len(self.robot_states)),
                                            init_func=self.animation_init, interval=20, repeat=False)
        
        plt.grid('--')
        for (ox, oy, w, h) in self.obs_rectangle:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )
        for (ox, oy, r) in self.obs_circle:
            self.ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )
        plt.show()
        if export_fig:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=180)
            self.ani.save('v_twoagent_obstacles.mp4', writer=writer)
        

    def animation_init(self):
        for i in range(int(len(self.target_state)/2)):
        # plot target state  
            self.target_circle = plt.Circle(self.target_state[i*2:2+i*2], 0.5, color='b', fill=True)
            self.ax.add_artist(self.target_circle)
            self.circle.append(plt.Circle(self.init_state[6+i*2:8+i*2], 0.5, color='g', fill=True))
            self.ax.add_artist(self.circle[i])
        self.robot1_body = plt.Circle(self.init_state[:2], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.robot1_body)
        self.robot1_arr = mpatches.Arrow(self.init_state[0], self.init_state[1],
                                        self.rob_radius * np.cos(self.init_state[2]),
                                        self.rob_radius * np.sin(self.init_state[2]), width=0.2, color='r')
        self.ax.add_patch(self.robot1_arr)
        self.robot2_body = plt.Circle(self.init_state[3:5], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.robot2_body)
        self.robot2_arr = mpatches.Arrow(self.init_state[3], self.init_state[4],
                                        self.rob_radius * np.cos(self.init_state[5]),
                                        self.rob_radius * np.sin(self.init_state[5]), width=0.2, color='r')
        self.ax.add_patch(self.robot2_arr)
        self.curve,=self.ax.plot(self.curve_state[0][0,:],self.curve_state[0][1,:])
        return self.target_circle, self.robot1_body, self.robot1_arr,self.robot2_body, self.robot2_arr,self.curve

    def animation_loop(self, indx):
        # robot1
        position1 = self.robot_states[indx][:2]
        orientation1 = self.robot_states[indx][2]
        self.robot1_body.center = position1
        self.robot1_arr.remove()
        self.robot1_arr = mpatches.Arrow(position1[0], position1[1], self.rob_radius * np.cos(orientation1),
                                        self.rob_radius * np.sin(orientation1), width=0.2, color='r')
        self.ax.add_patch(self.robot1_arr)
        # robot2
        position2 = self.robot_states[indx][3:5]
        orientation2 = self.robot_states[indx][5]
        self.robot2_body.center = position2
        self.robot2_arr.remove()
        self.robot2_arr = mpatches.Arrow(position2[0], position2[1], self.rob_radius * np.cos(orientation2),
                                        self.rob_radius * np.sin(orientation2), width=0.2, color='r')
        self.ax.add_patch(self.robot2_arr)
        #curve
        self.curve.set_ydata(self.curve_state[indx][1,:])
        self.curve.set_xdata(self.curve_state[indx][0,:])
        # the feature point of tube
        for i in range(int(len(self.target_state)/2)):
            position = self.robot_states[indx][6+i*2:8+i*2]
            self.circle[i].center = position
        return self.robot1_arr, self.robot1_body,self.robot2_arr, self.robot2_body, self.circle,self.curve
    
# pulling
class Draw_MPC_two_agents_obstacles_pulling(object):
    def __init__(self, robot_states: list, init_state: np.array, target_state: np.array, rob_diam,curve_state,obs_rectangle,obs_circle,Target_circle,
                    export_fig=True,):
        self.robot_states = robot_states
        self.init_state = init_state
        self.target_state = target_state
        self.rob_radius = rob_diam / 2.0
        self.circle=[]
        self.obs_circle=obs_circle
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(0, 200), ylim=(0, 200))
        self.obs_rectangle = obs_rectangle
        self.curve_state=curve_state
        self.pulling_circle=Target_circle
        # self.fig.set_dpi(400)
        self.fig.set_size_inches(7, 6.5)
        # init for plot
        self.animation_init()

        self.ani = animation.FuncAnimation(self.fig, self.animation_loop, range(len(self.robot_states)),
                                            init_func=self.animation_init, interval=20, repeat=False)
        
        plt.grid('--')
        for (ox, oy, w, h) in self.obs_rectangle:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )
        for (ox, oy, r) in self.obs_circle:
            self.ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )
        plt.show()
        if export_fig:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=180)
            self.ani.save('v_twoagent_obstacles.mp4', writer=writer)
        

    def animation_init(self):
        for i in range(int(len(self.target_state)/2)):
        # plot target state  
            self.target_circle = plt.Circle(self.target_state[i*2:2+i*2], 0.5, color='b', fill=True)
            self.ax.add_artist(self.target_circle)
            self.circle.append(plt.Circle(self.init_state[6+i*2:8+i*2], 0.5, color='g', fill=True))
            self.ax.add_artist(self.circle[i])
        self.pulling=plt.Circle(self.pulling_circle[:2],self.pulling_circle[2],color='gray', fill=True)
        self.ax.add_artist(self.pulling)
        self.robot1_body = plt.Circle(self.init_state[:2], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.robot1_body)
        self.robot1_arr = mpatches.Arrow(self.init_state[0], self.init_state[1],
                                        self.rob_radius * np.cos(self.init_state[2]),
                                        self.rob_radius * np.sin(self.init_state[2]), width=0.2, color='r')
        self.ax.add_patch(self.robot1_arr)
        self.robot2_body = plt.Circle(self.init_state[3:5], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.robot2_body)
        self.robot2_arr = mpatches.Arrow(self.init_state[3], self.init_state[4],
                                        self.rob_radius * np.cos(self.init_state[5]),
                                        self.rob_radius * np.sin(self.init_state[5]), width=0.2, color='r')
        self.ax.add_patch(self.robot2_arr)
        self.curve,=self.ax.plot(self.curve_state[0][0,:],self.curve_state[0][1,:])
        return self.target_circle, self.robot1_body, self.robot1_arr,self.robot2_body, self.robot2_arr,self.curve,self.pulling

    def animation_loop(self, indx):
        # robot1
        position1 = self.robot_states[indx][:2]
        orientation1 = self.robot_states[indx][2]
        self.robot1_body.center = position1
        self.robot1_arr.remove()
        self.robot1_arr = mpatches.Arrow(position1[0], position1[1], self.rob_radius * np.cos(orientation1),
                                        self.rob_radius * np.sin(orientation1), width=0.2, color='r')
        self.ax.add_patch(self.robot1_arr)
        # robot2
        position2 = self.robot_states[indx][3:5]
        orientation2 = self.robot_states[indx][5]
        self.robot2_body.center = position2
        self.robot2_arr.remove()
        self.robot2_arr = mpatches.Arrow(position2[0], position2[1], self.rob_radius * np.cos(orientation2),
                                        self.rob_radius * np.sin(orientation2), width=0.2, color='r')
        self.ax.add_patch(self.robot2_arr)
        #curve
        self.curve.set_ydata(self.curve_state[indx][1,:])
        self.curve.set_xdata(self.curve_state[indx][0,:])
        # the feature point of tube
        for i in range(int(len(self.target_state)/2)):
            position = self.robot_states[indx][6+i*2:8+i*2]
            self.circle[i].center = position
        # update the pulling 
        position_pulling=((self.robot_states[indx][:2]+self.robot_states[indx][3:5])/2+self.robot_states[indx][6+i:8+i])/2
        self.pulling.center=position_pulling
        return self.robot1_arr, self.robot1_body,self.robot2_arr, self.robot2_body, self.circle,self.curve,self.pulling


class Draw_MPC_two_agents_obstacles_Jerror_pulling(object):
    def __init__(self, robot_states: list, init_state: np.array, target_state: np.array, rob_diam,curve_state,obs_rectangle,obs_circle,J_error,Time,Target_circle,n_tube,
                 export_fig=True,):
        self.robot_states = robot_states
        self.init_state = init_state
        self.target_state = target_state
        self.rob_radius = rob_diam / 2.0
        self.circle=[]
        self.obs_circle=obs_circle
        self.fig, (self.ax, self.ax2) =  plt.subplots(1, 2, figsize=(10, 4))
        self.ax.set_xlim(0, 200)
        self.ax.set_ylim(0, 200)
        self.ax.set_xlabel('x', fontsize=14)
        self.ax.set_ylabel('y', fontsize=14)
        self.ax2.set_xlim(0, max(Time))
        self.ax2.set_ylim(0, max(J_error))
        self.ax2.set_xlabel('t', fontsize=14)
        self.ax2.set_ylabel('error', fontsize=14)
        self.Time=Time
        self.Jerror=J_error
        self.pulling_circle=Target_circle
        self.n_tube=n_tube
        # self.fig = plt.figure()
        # self.ax = plt.axes(xlim=(0, 200), ylim=(0, 200))
        self.obs_rectangle = obs_rectangle
        self.curve_state=curve_state
        # self.fig.set_dpi(400)
        #self.fig.set_size_inches(7, 6.5)
        # init for plot
        self.animation_init()

        self.ani = animation.FuncAnimation(self.fig, self.animation_loop, range(len(self.robot_states)),
                                           init_func=self.animation_init, interval=20, repeat=False)
        
        plt.grid('--')
        for (ox, oy, w, h) in self.obs_rectangle:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )
        for (ox, oy, r) in self.obs_circle:
            self.ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )
        plt.show()
        if export_fig:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=180)
            self.ani.save('v_twoagent_obstacles.mp4', writer=writer)
            

    def animation_init(self):
        self.target_circle = plt.Circle(self.target_state, 0.5, color='b', fill=True)
        self.ax.add_artist(self.target_circle)
        for i in range(self.n_tube):
        # plot target state  
            self.circle.append(plt.Circle(self.init_state[6+i*2:8+i*2], 0.5, color='g', fill=True))
            self.ax.add_artist(self.circle[i])
        self.pulling=plt.Circle(self.pulling_circle[:2],self.pulling_circle[2],color='gray', fill=True)
        self.ax.add_artist(self.pulling)
        self.robot1_body = plt.Circle(self.init_state[:2], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.robot1_body)
        self.robot1_arr = mpatches.Arrow(self.init_state[0], self.init_state[1],
                                        self.rob_radius * np.cos(self.init_state[2]),
                                        self.rob_radius * np.sin(self.init_state[2]), width=0.2, color='r')
        self.ax.add_patch(self.robot1_arr)
        self.robot2_body = plt.Circle(self.init_state[3:5], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.robot2_body)
        self.robot2_arr = mpatches.Arrow(self.init_state[3], self.init_state[4],
                                        self.rob_radius * np.cos(self.init_state[5]),
                                        self.rob_radius * np.sin(self.init_state[5]), width=0.2, color='r')
        self.ax.add_patch(self.robot2_arr)
        self.curve,=self.ax.plot(self.curve_state[0][0,:],self.curve_state[0][1,:])
        # J_error
        self.error,=self.ax2.plot(self.Time[0],self.Jerror[0])
        return self.target_circle, self.robot1_body, self.robot1_arr,self.robot2_body, self.robot2_arr,self.curve,self.error

    def animation_loop(self, indx):
        # robot1
        position1 = self.robot_states[indx][:2]
        orientation1 = self.robot_states[indx][2]
        self.robot1_body.center = position1
        self.robot1_arr.remove()
        self.robot1_arr = mpatches.Arrow(position1[0], position1[1], self.rob_radius * np.cos(orientation1),
                                        self.rob_radius * np.sin(orientation1), width=0.2, color='r')
        self.ax.add_patch(self.robot1_arr)
        # robot2
        position2 = self.robot_states[indx][3:5]
        orientation2 = self.robot_states[indx][5]
        self.robot2_body.center = position2
        self.robot2_arr.remove()
        self.robot2_arr = mpatches.Arrow(position2[0], position2[1], self.rob_radius * np.cos(orientation2),
                                        self.rob_radius * np.sin(orientation2), width=0.2, color='r')
        self.ax.add_patch(self.robot2_arr)
        #curve
        self.curve.set_ydata(self.curve_state[indx][1,:])
        self.curve.set_xdata(self.curve_state[indx][0,:])
        # the feature point of tube
        for i in range(self.n_tube):
            position = self.robot_states[indx][6+i*2:8+i*2]
            self.circle[i].center = position
        #J_error_curve
        #self.error=self.ax2.plot(self.Time[:indx],self.Jerror[:indx])
        self.error.set_ydata(self.Jerror[:indx])
        self.error.set_xdata(self.Time[:indx])
        # update the pulling 
        position_pulling=((self.robot_states[indx][:2]+self.robot_states[indx][3:5])/2+self.robot_states[indx][5+self.n_tube:7+self.n_tube])/2
        self.pulling.center=position_pulling
        return self.robot1_arr, self.robot1_body,self.robot2_arr, self.robot2_body, self.circle,self.curve,self.ax2,self.error,self.pulling