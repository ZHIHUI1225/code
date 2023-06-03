#!/usr/bin/env python
# -*- coding: utf-8 -*-
# two agents without obstacles
# constant Jacobian matrix of deformable tube

import casadi as ca
import casadi.tools as ca_tools
import env
import numpy as np
from draw import Draw_MPC_two_agents_withtube
import time
import matplotlib.pyplot as plt
import csv
#enviroment
#  env
Plot=env.Plotting()
bounry_points=Plot.env.boun_point
obs_points=Plot.env.obs_point
obs_diagonal_point=Plot.env.obs_diagonal_point
def sigmoid(x):
    return 1 / (1 + ca.exp(-x))

def tanh(x):
    return 2 / (1 + ca.exp(-2*x))-1

def shift_movement(T, t0, x0, u, f,un,J):
    for i in range(un):
        f_value1 = f(x0[:3], u[i, :2])
        x0[:3]= x0[:3] + T*f_value1.T
        f_value2= f(x0[3:-2], u[i, 2:])
        x0[3:-2]= x0[3:-2] + T*f_value2.T
        x0[-2:]=x0[-2:]+ca.mtimes(J,ca.vertcat(T*f_value1[:2].T,T*f_value2[:2].T))
        t0 = t0 + T
    state_next_=x0
    t_ = t0
    u_next_ = ca.vertcat(u[un:, :], u[-un:, :])
    return t_, state_next_, u_next_
def deformbletube(x,un,J):
    for i in range(un):
        x[-2:]=x[-2:]+ca.mtimes(J,ca.vertcat(x[:2]-x[:2],x[3:5]-x[3:5]))
    return x
# def getd(Q):
#     J=0
#     for i in range(0,len(obs_diagonal_point),2):
#         J0_norm=ca.dot(obs_diagonal_point[i]-Q.T,obs_diagonal_point[i+1]-Q.T)/ca.norm_2(obs_diagonal_point[i]-Q.T)/ca.norm_2(obs_diagonal_point[i+1]-Q.T)
#         #J0_norm=sigmoid(5/3*J0_norm+10/3)
#         J0_norm=tanh(2/3.2*J0_norm+2-2/3.2*1.2)
#         #J0_norm=ca.if_else(J0_norm>=1.1,1.1,J0_norm)
#         J=J+J0_norm
#     return J

if __name__ == '__main__':
    T = 1# sampling time [s]
    N = 10 # prediction horizon
    rob_diam = 5 # [m]
    L=30 #the length of tube
    v_max = 1
    omega_max = np.pi/6

    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    theta = ca.SX.sym('theta')
    statesq = ca.vertcat(x, y)
    states = ca.vertcat(statesq, theta)
    n_states = states.size()[0]

    v = ca.SX.sym('v')
    omega = ca.SX.sym('omega')
    controls = ca.vertcat(v, omega)
    n_controls = controls.size()[0]

    ## rhs
    rhs = ca.horzcat(v*ca.cos(theta), v*ca.sin(theta))
    rhs = ca.horzcat(rhs, omega)
    ## function
    f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    ## for MPC
    U = ca.SX.sym('U', N, n_controls*2)

    X = ca.SX.sym('X', (N+1), n_states*3-1) # x1,y1,theta1,x2,y1,thate2,xs,ys

    P = ca.SX.sym('P', n_states*3-1+2)#initial states +target states 2 of s


    ### define
    X[0,:] = P[:n_states*3-1] # initial condiction
    J=np.array([[0.5,0,0.5,0],[-0,0.5,-0,0.5]])
    #### define the relationship within the horizon
    for i in range(N):
        f_value = f(X[i, :n_states], U[i, :2])
        X[i+1, :n_states] = X[i, :n_states] + f_value*T
        f_value = f(X[i, n_states:-2], U[i, 2:])
        X[i+1, n_states:-2] = X[i, n_states:-2] + f_value*T
        # J matrix
        X[i+1,-2:]=X[i,-2:]+ca.mtimes(J,ca.vertcat(X[i+1,:2].T-X[i,:2].T,X[i+1,3:5].T-X[i,3:5].T)).T
    # claculte D
    # for i in range(N+1):
    #     D[i,0]=getd(X[i,:2])
    #     D[i,1]=getd(X[i,n_states:-1])

    ff = ca.Function('ff', [U, P], [X], ['input_U', 'target_state'], ['horizon_states'])

    Q = np.eye(2)
    R=0.01*np.eye(4)
    #Q = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 0.0]])
    #R = np.array([[0.05, 0.0], [0.0, 0.005]])
    #### cost function
    obj = 0 #### cost
    for i in range(N):
        #without angle error
        #obj = obj + ca.mtimes([X[i, -2:]-P[-2:].T, Q, (X[i, -2:]-P[-2:].T).T])+ ca.mtimes([U[i, :], R, U[i, :].T])-0.03*ca.norm_2(X[i,:2]-X[i,3:-3])*ca.norm_2((X[i,:2]+X[i,3:-3])/2-X[i, -2:])
        obj = obj + ca.mtimes([X[i, -2:]-P[-2:].T, Q, (X[i, -2:]-P[-2:].T).T])+ ca.mtimes([U[i, :], R, U[i, :].T])
    g = [] # equal constrains
    lbg = []
    ubg = []
    for i in range(N+1):
        for j in range(4):
            g.append(X[i, j])
            lbg.append(0)
            ubg.append(200)
    for i in range(N):
        g.append(ca.norm_2(X[i,:2]-X[i,3:-3]))
        lbg.append(L/5)
        ubg.append(L)

    nlp_prob = {'f': obj, 'x': ca.reshape(U, -1, 1), 'p':P, 'g':ca.vertcat(*g)}
    opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    # lbg = 0
    # ubg = 200
    lbx = []
    ubx = []
    for _ in range(N):
        lbx.append(0)
        ubx.append(v_max)
    for _ in range(N):
        lbx.append(-omega_max)
        ubx.append(omega_max)
    for _ in range(N):
        lbx.append(0)
        ubx.append(v_max)
    for _ in range(N):
        lbx.append(-omega_max)
        ubx.append(omega_max)



    # Simulation
    t0 = 0.0
    x0 = np.array([10.0, 5.0,-np.pi/2,10.0, 26.0,-np.pi/2,2,15]).reshape(-1, 1)# initial state
    xs = np.array([150, 160]).reshape(-1, 1) # final state
    u0 = np.array([1,0,1,0]*N).reshape(-1, 4)# np.ones((N, 2)) # controls
    x_c = [] # contains for the history of the state
    u_c = []
    t_c = [t0] # for the time
    xx = []
    sim_time = 300
    un=3 # control step
    ## start MPC
    mpciter = 0
    start_time = time.time()
    index_t = []
    while(np.linalg.norm(x0[-2:]-xs)>0.1 and mpciter-sim_time/T<0.0 ):
        ## set parameter
        c_p = np.concatenate((x0, xs))
        init_control = ca.reshape(u0, -1, 1)
        t_ = time.time()
        res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
        index_t.append(time.time()- t_)
        u_sol = ca.reshape(res['x'],  N, n_controls*2) # one can only have this shape of the output
        ff_value = ff(u_sol, c_p) # [n_states, N]
        x_c.append(ff_value)
        u_c.append(u_sol[:, 0])
        t_c.append(t0)
        t0, x0, u0 = shift_movement(T, t0, x0, u_sol, f,un,J)
        x0 = ca.reshape(x0, -1, 1)
        xx.append(x0.full())
        mpciter = mpciter + 1
    t_v = np.array(index_t)

    print(t_v.mean())
    print((time.time() - start_time))
    xx=np.concatenate(xx)
    xnew=xx.reshape(int(xx.size/x0.size()[0]),x0.size()[0])
    plt.plot(xnew[:,-2], xnew[:,-1])
    plt.plot(xnew[:,0], xnew[:,1])
    plt.plot(xnew[:,3], xnew[:,-4])
    plt.show()

    data = [
        [xs],
        [x0],
        [rob_diam]
    ]
    for i in range(len(xnew)):
        data.append(xnew[i,:])
    with open('data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


    draw_result =Draw_MPC_two_agents_withtube(rob_diam=rob_diam, init_state=x0.full(), target_state=np.array(xs), robot_states=xnew,obs_rectangle=[] )
    # print(xx)
