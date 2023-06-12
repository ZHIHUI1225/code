#!/usr/bin/env python
# -*- coding: utf-8 -*-
# MPC 
# single agent 
# use angle information to avoid obstacles

import casadi as ca
import casadi.tools as ca_tools
import env
import numpy as np
from numpy import linalg as LA
from draw import Draw_MPC_point_stabilization_v1
import time
#enviroment
#  env
Plot=env.Plotting()
bounry_points=Plot.env.boun_point
obs_points=Plot.env.obs_point
obs_diagonal_point=Plot.env.obs_diagonal_point
l=10
def sigmoid(x):
    return 1 / (1 + ca.exp(-x))

def tanh(x):
    return 2 / (1 + ca.exp(-2*x))-1

def shift_movement(T, t0, x0, u, f,un):
    for i in range(un):
        f_value = f(x0, u[i, :])
        x0 = x0 + T*f_value.T
        t0 = t0 + T
    state_next_=x0
    t_ = t0
    u_next_ = ca.vertcat(u[un:, :], u[-un:, :])
    return t_, state_next_, u_next_

def getd(Q):
    J=0
    for i in range(0,len(obs_diagonal_point),2):
        J0_norm=ca.dot(obs_diagonal_point[i]-Q.T,obs_diagonal_point[i+1]-Q.T)/ca.norm_2(obs_diagonal_point[i]-Q.T)/ca.norm_2(obs_diagonal_point[i+1]-Q.T)
        #J0_norm=sigmoid(5/3*J0_norm+10/3)
        #J0_norm=tanh(2/3.2*J0_norm+2-2/3.2*1.2)
        J0_norm=tanh(3*J0_norm+3)
        J=J+J0_norm
    return J
#get the coefficient of the derivative of  cos\theta 
def getdOk(Q):
    Q=Q.reshape((2,1))
    K=0
    for i in range(0,len(obs_diagonal_point),2):
        P=np.array(obs_diagonal_point[i]).reshape((2,1))
        l1=LA.norm(P-Q,2)
        z1=(P-Q)/l1
        P=np.array(obs_diagonal_point[i+1]).reshape((2,1))
        l2=LA.norm(P-Q)
        z2=(P-Q)/l2
        O=np.dot(np.transpose(z1),z2)
        K0=np.dot(np.concatenate((1/l1-O/l1,1/l1-O/l2),axis=1),np.concatenate((np.transpose(z1),np.transpose(z2)),axis=0))
        K=K+LA.norm(K0,2)

    return K

if __name__ == '__main__':
    T = 1# sampling time [s]
    N = 10 # prediction horizon
    rob_diam = 5 # [m]
    v_max = 1
    omega_max = np.pi/6

    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    theta = ca.SX.sym('theta')
    d = ca.SX.sym('d')
    statesq = ca.vertcat(x, y)
    states = ca.vertcat(statesq, theta)
    n_states = states.size()[0]

    v = ca.SX.sym('v')
    omega = ca.SX.sym('omega')
    controls = ca.vertcat(v, omega)
    n_controls = controls.size()[0]

    ## rhs
    # rhs = ca.horzcat(v*ca.cos(theta), v*ca.sin(theta))
    # rhs = ca.horzcat(rhs, omega)
    rhs = ca.horzcat(v*ca.cos(theta)-l*ca.sin(theta)*omega, v*ca.sin(theta)+l*ca.cos(theta)*omega)
    rhs = ca.horzcat(rhs, omega)
    # ##calculate the Derivative of d
    # dd=0
    # for i in range(0,len(obs_diagonal_point),2):
    #     l1=ca.norm_2(obs_diagonal_point[i]-statesq)
    #     z1=(obs_diagonal_point[i]-statesq)/l1
    #     M1=np.identity(2)-ca.mtimes(z1,z1.T)
    #     l2=ca.norm_2(obs_diagonal_point[i+1]-statesq)
    #     z2=(obs_diagonal_point[i+1]-statesq)/l2
    #     M2=np.identity(2)-ca.mtimes(z2,z2.T)
    #     dd0=-ca.mtimes((ca.mtimes(z1.T,M2)/l2+ca.mtimes(z2.T,M1)/l1),ca.vertcat(v*ca.cos(theta), v*ca.sin(theta)))
    #     #J0_norm=ca.dot(obs_diagonal_point[i]-x,obs_diagonal_point[i+1]-x)/ca.norm_2(obs_diagonal_point[i]-statesq)/ca.norm_2(obs_diagonal_point[i+1]-statesq)
    #     #dd=ca.if_else(J0_norm>=1.2,dd,dd+dd0)
    #     dd=dd+dd0
    # rhs = ca.horzcat(rhs, omega,dd)
    ## function
    f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    ## for MPC
    U = ca.SX.sym('U', N, n_controls)

    X = ca.SX.sym('X', N+1, n_states)
    #cos\theta presents the distant with obstacles
    D= ca.SX.sym('D', N+1, 1)

    P = ca.SX.sym('P', n_states+n_states)


    ### define
    X[0, :] = P[:n_states] # initial condiction

    #### define the relationship within the horizon
    for i in range(N):
        f_value = f(X[i, :], U[i, :])
        X[i+1, :] = X[i, :] + f_value*T
    # claculte D
    for i in range(N+1):
        D[i]=getd(X[i,:2])

    ff = ca.Function('ff', [U, P], [X], ['input_U', 'target_state'], ['horizon_states'])

    #Q = np.array([[1.0, 0.0],[0.0, 5.0]])
    Q = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 0.0]])
    R = np.array([[0.05, 0.0], [0.0, 0.005]])
    g = [] # equal constrains
    for i in range(N+1):
        g.append(X[i, 0])
        g.append(X[i, 1])

   

    lbg = 0
    ubg = 200
    lbx = []
    ubx = []
    for _ in range(N):
        lbx.append(0)
        ubx.append(v_max)
    for _ in range(N):
        lbx.append(-omega_max)
        ubx.append(omega_max)


    # Simulation
    t0 = 0.0
    x0 = np.array([10.0, 10.0,-np.pi/2]).reshape(-1, 1)# initial state
    xs = np.array([170, 160,-np.pi/2]).reshape(-1, 1) # final state
    u0 = np.array([1,0]*N).reshape(-1, 2)# np.ones((N, 2)) # controls
    x_c = [] # contains for the history of the state
    u_c = []
    t_c = [t0] # for the time
    xx = []
    sim_time = 100
    un=4 # control step
    ## start MPC
    mpciter = 0
    start_time = time.time()
    index_t = []
    while(np.linalg.norm(x0-xs)>1 and mpciter-sim_time/T<0.0 ):
        ## set parameter
        c_p = np.concatenate((x0, xs))
        init_control = ca.reshape(u0, -1, 1)
        t_ = time.time()
        #### cost function
        obj = 0 #### cost
        for i in range(N):
        #without angle error
        #obj = obj + ca.mtimes([X[i, :2]-P[n_states:-1].T, Q, (X[i, :2]-P[n_states:-1].T).T]) + ca.mtimes([U[i, :], R, U[i, :].T])
            obj = obj + ca.mtimes([X[i, :]-P[n_states:].T, Q, (X[i, :]-P[n_states:].T).T]) -2.5*LA.norm(x0-xs,2)/3/(1-tanh(3)**2)/getdOk(x0[:2])*getd(X[i,:2])+ ca.mtimes([U[i, :], R, U[i, :].T])
        nlp_prob = {'f': obj, 'x': ca.reshape(U, -1, 1), 'p':P, 'g':ca.vertcat(*g)}
        opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
        
        res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
        index_t.append(time.time()- t_)
        u_sol = ca.reshape(res['x'],  N, n_controls) # one can only have this shape of the output
        ff_value = ff(u_sol, c_p) # [n_states, N]
        x_c.append(ff_value)
        u_c.append(u_sol[:, 0])
        t_c.append(t0)
        t0, x0, u0 = shift_movement(T, t0, x0, u_sol, f,un)
        x0 = ca.reshape(x0, -1, 1)
        xx.append(x0.full())
        mpciter = mpciter + 1
    t_v = np.array(index_t)

    print(t_v.mean())
    print((time.time() - start_time)/(mpciter))

    draw_result = Draw_MPC_point_stabilization_v1(rob_diam=rob_diam, init_state=x0.full(), target_state=xs, robot_states=xx )
    # print(xx)
