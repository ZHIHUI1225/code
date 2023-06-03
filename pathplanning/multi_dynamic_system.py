# based on MPCtwo.py , MPC_single.py
# add the obstacle avoidence
# 2 agents system ,dynamic obstacles
import casadi as ca
import env
import numpy as np
from draw import Draw_MPC_two_agents_obstacles
import time
import matplotlib.pyplot as plt
from numpy import linalg as LA
from tube_sineshape import tubeshape
from updateJmatrix import Jmatrix
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

def shift_movement(T, t0, x0, u, f,un,):
    for i in range(un):
        #two agent 
        f_value1 = f(x0[:3], u[i, :2])
        x0[:3]= x0[:3] + T*f_value1.T
        f_value2= f(x0[3:6], u[i, 2:])
        x0[3:6]= x0[3:6] + T*f_value2.T
        t0 = t0 + T
    state_next_=x0
    t_ = t0
    u_next_ = ca.vertcat(u[un:, :], u[-un:, :])
    return t_, state_next_, u_next_
def getd(Q):
    J=0
    sf=0.2
    for i in range(0,len(obs_diagonal_point),2):
        J0_norm=ca.dot(obs_diagonal_point[i]-Q.T,obs_diagonal_point[i+1]-Q.T)/ca.norm_2(obs_diagonal_point[i]-Q.T)/ca.norm_2(obs_diagonal_point[i+1]-Q.T)
        J0_norm=tanh(3/(sf+1)*(J0_norm+1))
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

# generate FF
def gene_ff(X,U,P,f,J):
    ### define
    X[0,:] = P[:n_states*2+n_tube*2] # initial condiction
    #### define the relationship within the horizon
    for i in range(N):
        f_value = f(X[i, :n_states], U[i, :2])
        X[i+1, :n_states] = X[i, :n_states] + f_value*T
        f_value = f(X[i, n_states:n_states+3], U[i, 2:])
        X[i+1, n_states:n_states+3] = X[i, n_states:n_states+3] + f_value*T
        # J matrix
        X[i+1,-n_tube*2:]=X[i,-n_tube*2:]+ca.mtimes(J,ca.vertcat(X[i+1,:2].T-X[i,:2].T,X[i+1,3:5].T-X[i,3:5].T)).T

    ff = ca.Function('ff', [U, P], [X], ['input_U', 'target_state'], ['horizon_states'])

    return [X,ff]

if __name__ == '__main__':
    T = 1# sampling time [s]
    N = 10 # prediction horizon
    rob_diam = 2 # [m]
    L=30 #the length of tube
    v_max = 0.8
    omega_max = np.pi/5
    n_tube=5 # the points of the tube
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    theta = ca.SX.sym('theta')
    statesq = ca.vertcat(x, y)
    states = ca.vertcat(statesq, theta)
    n_states = states.size()[0]
    x0 = np.array([5.0, 5.0,-np.pi/2,26.0, 5.0,-np.pi/2]).reshape(-1, 1)
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

    X = ca.SX.sym('X', (N+1), n_states*2+n_tube*2) # x1,y1,theta1,x2,y1,thate2,(xs,ys)*n_tube

    P = ca.SX.sym('P', n_states*2+n_tube*2+2*n_tube)#initial states +target states 2 of points of tube
    
    Matrix=Jmatrix(T=8,L=L,n_tube=n_tube,p1=x0[:2],p2=x0[3:5])

    [X,ff]=gene_ff(X,U,P,f,Matrix.J)

    Q = np.eye(2*n_tube)
    # R=0.05*np.eye(4)
    R=np.array([[0.01,0,0,0],[0,0.001,0,0],[0,0,0.01,0],[0,0,0,0.001]])
    g = [] # equal constrains
    lbg = []
    ubg = []
    for i in range(N+1):
        for j in range(4):
            g.append(X[i, j])
            lbg.append(0)
            ubg.append(200)
    for i in range(N):
        g.append(ca.norm_2(X[i,:2]-X[i,3:5]))
        lbg.append(L/3)
        ubg.append(L*5/6)

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
    x0 = np.array([5.0, 5.0,-np.pi/2,26.0, 5.0,-np.pi/2]).reshape(-1, 1)# initial state
    Tube=tubeshape(length=L,p1=x0[:2],p2=x0[3:5])
    xt=np.array(Tube.get_points(n_tube)).ravel()
    x0=np.vstack((x0,np.reshape(xt,(len(xt),1))))

    Tube.update(np.array([170,150]),np.array([170,170]))
    xt=np.array(Tube.get_points(n_tube)).ravel()
    xs=np.reshape(xt,(len(xt),1))
    u0 = np.array([0.5,0,0.5,0]*N).reshape(-1, 4)# np.ones((N, 2)) # controls
       
    x_c = [] # contains for the history of the state
    u_c = []
    t_c = [t0] # for the time
    xx = []
    error=[] # the error of deltas- J*delata r
    xp=[] # save the information to plot the curve
    sim_time = 300
    un=3# control step
    ## start MPC
    mpciter = 0
    Time=[]
    start_time = time.time()
    index_t = []
    while(np.linalg.norm(x0[-2*n_tube:]-xs,2)>0.1 and mpciter-sim_time/T<0.0 ):
        ## set parameter
        c_p = np.concatenate((x0, xs))
        init_control = ca.reshape(u0, -1, 1)
        t_ = time.time()
        # update the optimization objective function
        obj = 0 #### cost
        for i in range(N):
        #without angle error
            obj = obj +ca.mtimes([X[i, -2*n_tube:]-P[-2*n_tube:].T, Q, (X[i, -2*n_tube:]-P[-2*n_tube:].T).T])\
            + ca.mtimes([U[i, :], R, U[i, :].T])\
            -3*LA.norm(x0[-2*n_tube:]-xs,2)/3/(1-tanh(3)**2)/getdOk(x0[:2])*getd(X[i,:2])\
            -3*LA.norm(x0[-2*n_tube:]-xs,2)/3/(1-tanh(3)**2)/getdOk(x0[3:5])*getd(X[i,3:5])\
            -5*LA.norm(x0[-2*n_tube:]-xs,2)/3/(1-tanh(3)**2)/getdOk(x0[-2:])*getd(X[i,-2:])
            for j in range(1,n_tube):
                obj=obj-5*LA.norm(x0[-2*n_tube:]-xs,2)/3/(1-tanh(3)**2)/getdOk(x0[-2*(j+1):-2*j])*getd(X[i,-2*(j+1):-2*j])
        nlp_prob = {'f': obj, 'x': ca.reshape(U, -1, 1), 'p':P, 'g':ca.vertcat(*g)}
        opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
        res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
        index_t.append(time.time()- t_)
        u_sol = ca.reshape(res['x'],  N, n_controls*2) # one can only have this shape of the output
        ff_value = ff(u_sol, c_p) # [n_states, N]
        x_c.append(ff_value)
        u_c.append(u_sol)
        t_c.append(t0)
        x0old=x0
        #real system model
        t0, x0, u0 = shift_movement(T, t0, x0, u_sol.toarray(), f,un)
        xold=np.array(Tube.get_points(n_tube)).ravel()
        Tube.update(x0[:2],x0[3:5])
        xt=np.array(Tube.get_points(n_tube)).ravel()
        deltap=np.vstack((x0[:2]-x0old[:2],x0[3:5]-x0old[3:5]))
        deltas=xt-xold
        error.append(deltas.reshape(2*n_tube,1)-np.dot(Matrix.J,deltap))
        xp.append(Tube.getshape())
        Matrix.unpdateJ(x0[:2].reshape(2,1),x0[3:5].reshape(2,1),xtn=xt.reshape(2*n_tube,1))
        [X,ff]=gene_ff(X,U,P,f,Matrix.J)
        x0[6:]=np.reshape(xt,(len(xt),1))
        xx.append(x0.copy())
        #x0 = np.reshape(x0,(-1, 1))
        mpciter = mpciter + 1
        Time.append(mpciter )

       
    t_v = np.array(index_t)
    print(mpciter)
    print(x0[-2*n_tube:]-xs)

    draw_result =Draw_MPC_two_agents_obstacles(rob_diam=rob_diam, init_state=x0, target_state=np.array(xs), robot_states=xx,curve_state=xp,obs_rectangle=Plot.obs_rectangle,obs_circle=Plot.obs_circle)