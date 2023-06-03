# generate the targets at random
# pick the target to home

import casadi as ca
import env
import numpy as np
from draw import Draw_MPC_two_agents_obstacles_Jerror
import time
import matplotlib.pyplot as plt
from numpy import linalg as LA
from tube_sineshape import tubeshape
from updateJmatrix import Jmatrix
import datetime
#enviroment
#  env
Plot=env.Plotting()
bounry_points=Plot.env.boun_point
obs_points=Plot.env.obs_point
obs_diagonal_point=Plot.env.obs_diagonal_point
T = 1# sampling time [s]
N = 10 # prediction horizon
rob_diam = 2 # [m]
L=30 #the length of tube
v_max = 0.8
omega_max = np.pi/6
sf=0.3
n_tube=5 # the points of the tube
Tube=tubeshape(length=L,p1=[10,167],p2=[10,185])
home=np.array(Tube.get_points(n_tube)).ravel()
