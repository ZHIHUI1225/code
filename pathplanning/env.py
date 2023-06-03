"""
Environment for rrt_2D
@author: huiming zhou
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
import casadi as ca
import numpy as np
import math
from matplotlib.patches import Rectangle
import copy
def polynonial(t,d):
    
    B=np.zeros((d,1))
    for i in range(d):
        B[i]=t**i
    return B
def dpolynonial(t,d):
 
    B=np.zeros((d,1))
    for i in range(d):
        if i==0:
            B[i]=0
        else:
            B[i]=i*t**(i-1)
    return B

class Env:
    def __init__(self):
        self.x_range = (0, 200)
        self.y_range = (0, 200)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()
        self.boun_point=self.boun_surface_points(self)
        self.obs_point=self.obs_surface_points(self)
        self.obs_diagonal_point=self.obs_diagonal_points(self)
    @staticmethod
    def obs_boundary():
        # ox, oy, w, h
        obs_boundary = [
            [0, 0, 1, 200],
            [0, 200, 200, 1],
            [1, 0, 200, 1],
            [200, 1, 1, 200]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle(): #ox, oy, w, h
        obs_rectangle = [
            #[105,80,30,30],
            #[50,10,40,40]
            [60,0,30,30],
            [35,75,30,30],
            #[158,170,7,7],
        ]
        return obs_rectangle

    @staticmethod
    def obs_circle(): #x, y, r
        obs_cir = [
            #[170, 120, 10],
            [120, 90, 10],
            #[168,160,5],
        ]
        return obs_cir
    def expendobs(self):
        obs_cir=copy.deepcopy(self.obs_circle)
        obs_rectangle=copy.deepcopy(self.obs_rectangle)
        expend=17
        for i in range(len(self.obs_circle)):
            obs_cir[i][2]=obs_cir[i][2]+expend
        for i in range(len(self.obs_rectangle)):
            obs_rectangle[i][0]=obs_rectangle[i][0]-expend
            obs_rectangle[i][1]=obs_rectangle[i][1]-expend
            obs_rectangle[i][2]=obs_rectangle[i][2]+expend
            obs_rectangle[i][3]=obs_rectangle[i][3]+expend
        points=[]
        for i in range(len(obs_rectangle)):
            obs=obs_rectangle[i]
            points.append([obs[0],obs[1]])
            points.append([obs[0]+obs[2],obs[1]+obs[3]])
            points.append([obs[0]+obs[2],obs[1]])
            points.append([obs[0],obs[1]+obs[3]])
        for i in range(len(obs_cir)):
            obs=obs_cir[i]
            points.append([obs[0],obs[1]+obs[2]])
            points.append([obs[0],obs[1]-obs[2]])
            points.append([obs[0]+obs[2],obs[1]])
            points.append([obs[0]-obs[2],obs[1]])
        return points
    @staticmethod
    def boun_surface_points(self):
        points=[]
        for i in range(4):
            obs=self.obs_boundary[i]
            for w in range(obs[2]):
                points.append([obs[0]+w,obs[1]])
                points.append([obs[0]+w,obs[1]+obs[3]])
            for h in range(1,obs[3]):
                points.append([obs[0],obs[1]+h])
                points.append([obs[0]+obs[2],obs[1]+h])
        return points
    
    @staticmethod
    def obs_surface_points(self):
        points=[]
        for i in range(len(self.obs_rectangle)):
            obs=self.obs_rectangle[i]
            for w in range(obs[2]):
                points.append([obs[0]+w,obs[1]])
                points.append([obs[0]+w,obs[1]+obs[3]])
            for h in range(1,obs[3]):
                points.append([obs[0],obs[1]+h])
                points.append([obs[0]+obs[2],obs[1]+h])
        return points
    
    @staticmethod
    def obs_diagonal_points(self):
        points=[]
        for i in range(len(self.obs_rectangle)):
            obs=self.obs_rectangle[i]
            points.append([obs[0],obs[1]])
            points.append([obs[0]+obs[2],obs[1]+obs[3]])
            points.append([obs[0]+obs[2],obs[1]])
            points.append([obs[0],obs[1]+obs[3]])
        for i in range(len(self.obs_circle)):
            obs=self.obs_circle[i]
            points.append([obs[0],obs[1]+obs[2]])
            points.append([obs[0],obs[1]-obs[2]])
            points.append([obs[0]+obs[2],obs[1]])
            points.append([obs[0]-obs[2],obs[1]])
        return points

class Plotting:
    def __init__(self):
        self.env = Env()
        self.obs_bound = self.env.obs_boundary
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle

    def plot_grid(self, name):
        fig, ax = plt.subplots()

        for (ox, oy, w, h) in self.obs_bound:
            ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )

        for (ox, oy, w, h) in self.obs_rectangle:
            ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        for (ox, oy, r) in self.obs_circle:
            ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        # plt.plot(self.xI[0], self.xI[1], "bs", linewidth=3)
        # plt.plot(self.xG[0], self.xG[1], "gs", linewidth=3)

        plt.title(name)
        plt.axis("equal")
    @staticmethod
    def plotpath(x1_plot,x2_plot,d_plot,l_plot,h,CM,d):
        plt.plot(x1_plot, x2_plot)
        N=len(x1_plot)
        for k in range(0,N+1,2):
            t=k*h
            dXk=ca.mtimes(CM[0:2,:],dpolynonial(t,d))
            ang=math.atan2(dXk[1],dXk[0])
            point=np.array([x1_plot[k]-d_plot[k]*(math.cos(math.pi/2+ang))/2,x2_plot[k]-d_plot[k]*(math.sin(math.pi/2+ang))/2])

            plt.gca().add_patch(Rectangle((point[0][0],point[1][0]),d_plot[k],l_plot[k],angle=90+math.degrees(ang),
                            edgecolor='red',
                            facecolor='none',
                            lw=1))
        plt.show()
    @staticmethod
    def plotpathsingle(x1_plot,x2_plot,l1_plot,l2_plot):
        plt.plot(x1_plot, x2_plot)
        plt.plot(l1_plot, l2_plot)
        plt.show()