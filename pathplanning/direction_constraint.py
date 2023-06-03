# based on MPCtwo.py , MPC_single.py
# add the obstacle avoidence
import casadi as ca
import env
import numpy as np
from draw import Draw_MPC_two_agents_obstacles_Jerror_pulling
from draw import Draw_MPC_two_agents_obstacles_Jerror
import time
import matplotlib.pyplot as plt
from numpy import linalg as LA
from tube_sineshape import tubeshape
from updateJmatrix import Jmatrix
import datetime
import math
import random
#enviroment
#  env
Plot=env.Plotting()
bounry_points=Plot.env.boun_point
obs_points=Plot.env.obs_point
# obs_diagonal_point=Plot.env.obs_diagonal_point
obs_diagonal_point=Plot.env.expendobs()
sf=0.1
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
    state_next_=x0.copy()
    t_ = t0
    u_next_ = ca.vertcat(u[un:, :], u[-un:, :])
    return t_, state_next_, u_next_
def getd(Q):
    J=0
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
#get the coefficient of the derivative of  cos\theta multipy the Jacobi Matrix
def getdM(Q,M,k,j):
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
        J0_norm=ca.dot(obs_diagonal_point[i]-Q.T,obs_diagonal_point[i+1]-Q.T)/ca.norm_2(obs_diagonal_point[i]-Q.T)/ca.norm_2(obs_diagonal_point[i+1]-Q.T)
        J0_norm=1-(tanh(3/(sf+1)*(J0_norm+1)))**2
        K=K+J0_norm*LA.norm(np.dot(K0,M[2*k:2*k+2,2*j:2*j+2]))
    return K
# generate FF
def gene_ff(X,U,P,f,J):
    ### define
    X[0,:] = P# initial condiction
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
# from world to local
def rotationR(x0,n_tube):
    theta=math.atan2(x0[4]-x0[1],x0[3]-x0[0])#(y,x)
    R=np.array([[math.cos(theta),math.sin(theta)],[-math.sin(theta),math.cos(theta)]])
    O=x0[5+n_tube:n_tube+7]
    return [R,O]

def carotationR(x0,n_tube):  
    theta=ca.atan2(x0[4]-x0[1],x0[3]-x0[0])
    R=ca.horzcat(ca.vertcat(ca.cos(theta),-ca.sin(theta)), ca.vertcat(ca.sin(theta),ca.cos(theta)))
    O=x0[5+n_tube:n_tube+7].T
    return [R,O]

def getcenter(X,n_tube):
    Xcenter=((X[:2]+X[3:5])/2+X[5+n_tube:n_tube+7])/2
    return Xcenter

# def gettargetrandomly(r,obs_boundary,obs_rectangle,obs_cir):
#     #  obs_boundary 4* ox, oy, w, h
#     # obs_rectangle ox, oy, w, h
#     # obs_circle x, y, r
#     wide=obs_boundary[0][3]
#     height=obs_boundary[3][3]
#     flag=1
#     while(flag==1):
#         flag=0
#         h=obs_boundary[0][2]
#         x=random.uniform(obs_boundary[0][0]+r+h,obs_boundary[0][0]+wide/2-r-h)
#         y=random.uniform(obs_boundary[0][1]+r+h,obs_boundary[0][1]+height/2-r-h)
#         for i in range(len(obs_rectangle)):
#             if x>obs_rectangle[i][0]-r*2 and x<obs_rectangle[i][0]+obs_rectangle[i][2]+r*2 and y>obs_rectangle[i][1]-r*2 and y<obs_rectangle[i][1]+obs_rectangle[i][3]+r*2:
#                 flag=1
#         for i in range(len(obs_cir)):
#             if (x-obs_cir[i][0])**2+(y-obs_cir[i][1])**2<(r*2+obs_cir[i][2])**2:
#                 flag=1
#     return [x,y,r]

def gettargetrandomly(r,obs_boundary,obs_rectangle,obs_cir):
    #  obs_boundary 4* ox, oy, w, h
    # obs_rectangle ox, oy, w, h
    # obs_circle x, y, r
    wide=obs_boundary[0][3]
    height=obs_boundary[3][3]
    flag=1
    while(flag==1):
        flag=0
        h=obs_boundary[0][2]
        x=random.uniform(obs_boundary[0][0]+wide*2/3+r+h,obs_boundary[0][0]+wide*4/5-r-h)
        y=random.uniform(obs_boundary[0][1]+wide*2/3+r+h,obs_boundary[0][1]+height*4/5-r-h)
        for i in range(len(obs_rectangle)):
            if x>obs_rectangle[i][0]-r*2 and x<obs_rectangle[i][0]+obs_rectangle[i][2]+r*2 and y>obs_rectangle[i][1]-r*2 and y<obs_rectangle[i][1]+obs_rectangle[i][3]+r*2:
                flag=1
        for i in range(len(obs_cir)):
            if (x-obs_cir[i][0])**2+(y-obs_cir[i][1])**2<(r*2+obs_cir[i][2])**2:
                flag=1
    return [x,y,r]

if __name__ == '__main__':
    T = 1# sampling time [s]
    N =15# prediction horizon
    rob_diam = 2 # [m]
    L=30 #the length of tube
    v_max = 1.2
    omega_max = np.pi/4
    n_tube=3 # the points of the tube
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    theta = ca.SX.sym('theta')
    statesq = ca.vertcat(x, y)
    states = ca.vertcat(statesq, theta)
    n_states = states.size()[0]
    Initial_circle=gettargetrandomly(5,Plot.env.obs_boundary,Plot.env.obs_rectangle,Plot.env.obs_circle)
    x0 = np.array([Initial_circle[0]+Initial_circle[2]+5, Initial_circle[1]-2,-np.pi/2,Initial_circle[0]-Initial_circle[2]-5, Initial_circle[1]-2,-np.pi/2]).reshape(-1, 1)
    Tube=tubeshape(length=L,p1=np.array([Initial_circle[0]+Initial_circle[2]+5,Initial_circle[1]-2]),p2=np.array([Initial_circle[0]-Initial_circle[2]-5,Initial_circle[1]-2]))
    xt=np.array(Tube.get_points(n_tube)).ravel()
    x0=np.vstack((x0,np.reshape(xt,(len(xt),1))))   
    v = ca.SX.sym('v')
    omega = ca.SX.sym('omega')
    controls = ca.vertcat(v, omega)
    n_controls = controls.size()[0]
    # generate target
    Target_circle=[25,25,5]#[x,y,r]
   
    ## rhs
    rhs = ca.horzcat(v*ca.cos(theta), v*ca.sin(theta))
    rhs = ca.horzcat(rhs, omega)
    ## function
    f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    ## for MPC
    U = ca.SX.sym('U', N, n_controls*2)

    X = ca.SX.sym('X', (N+1), n_states*2+n_tube*2) # x1,y1,theta1,x2,y1,thate2,(xs,ys)*n_tube
    Xnew = ca.SX.sym('Xnew', N, 2)
    P = ca.SX.sym('P', n_states*2+n_tube*2)#initial states of points of tube

    Matrix=Jmatrix(T=20,L=L,n_tube=n_tube,p1=x0[:2],p2=x0[3:5])

    [X,ff]=gene_ff(X,U,P,f,Matrix.J)

    Q = np.eye(2*n_tube)
    # R=0.05*np.eye(4)
    R=0.001*np.array([[0.01,0,0,0],[0,0.003,0,0],[0,0,0.01,0],[0,0,0,0.003]])
    
    g = [] # equal constrains
    lbg = []
    ubg = []
    for i in range(N+1):
        for j in range(2):
            g.append(X[i, 3*j])
            lbg.append(0)
            ubg.append(200)
            g.append(X[i, 3*j+1])
            lbg.append(0)
            ubg.append(200)
        for j in range(n_states*2,n_states*2+n_tube*2):
            g.append(X[i, j])
            lbg.append(0)
            ubg.append(200)
    for i in range(N):
        g.append(ca.norm_2(X[i,:2]-X[i,3:5]))
        lbg.append(L/2)
        ubg.append(L*3/4)

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
    for i in range(N):
            [M,O]=carotationR(X[i,:],n_tube)
            b=ca.mtimes(M,X[i,3:5].T)-ca.mtimes(M,O)
            # b=ca.mtimes(M,X[i,3+n_tube:n_tube+5].T)-ca.mtimes(M,O)
            Kb=ca.fabs(b[1]/b[0])
            Xnew[i,:]=(ca.mtimes(M,(X[i+1,5+n_tube:7+n_tube]).T)).T-ca.mtimes(M,O).T
            g.append(ca.fabs(Xnew[i,0])-(Xnew[i,1]/Kb))
            lbg.append(-100)
            ubg.append(0)
            g.append(Xnew[i,0])
            lbg.append(-1)
            ubg.append(1)
            g.append(Xnew[i,1])
            lbg.append(0)
            ubg.append(1)


    # Simulation
    t0 = 0.0
#     x0 = np.array([100.0, 85.0,-np.pi/2,100.0, 65.0,-np.pi/2]).reshape(-1, 1)# initial state
#    # x0 = np.array([20,160,np.pi/2,40,160,np.pi/2]).reshape(-1, 1)# 
#     Tube=tubeshape(length=L,p1=x0[:2],p2=x0[3:5])
#     xt=np.array(Tube.get_points(n_tube)).ravel()
#     x0=np.vstack((x0,np.reshape(xt,(len(xt),1))))

    # Tube.update(np.array([13,15]),np.array([13,35]))
    # xt=np.array(Tube.get_points(n_tube)).ravel()
    # xs=np.reshape(xt,(len(xt),1))
    xs=ca.DM(Target_circle[:2]) #target position
    u0 = np.array([0,0,0,0]*N).reshape(-1, 4)# np.ones((N, 2)) # controls
   
    x_c = [] # contains for the history of the state
    u_c = []
    t_c = [t0] # for the time
    xx = []
    error=[] # the error of deltas- J*delata r
    xp=[] # save the information to plot the curve
    sim_time = 200
    un=2# control step
    ## start MPC
    mpciter = 0
    Time=[]
    start_time = time.time()
    index_t = []
    Ks=[0 for _ in range(n_tube)]  # the coefficient of ostacle avoidance of feature points
    K=[0 for _ in range(2)]
    Tube.update(x0[:2],x0[3:5])
    begin_time = datetime.datetime.now()
    print(begin_time)
    while(np.linalg.norm(((x0[:2]+x0[3:5])/2+x0[5+n_tube:n_tube+7])/2-xs,2)>0.3 and mpciter-sim_time/T<0.0 ):
        # [M,O]=rotationR(x0,n_tube)
        # b=np.dot(M,x0[:2])-O
        # Kb=abs(b[1]/b[0])
        # ## set parameter
        # g0=g.copy()
        # lbg0=lbg.copy()
        # ubg0=ubg.copy()
        c_p = np.concatenate((x0))
        init_control = ca.reshape(u0, -1, 1)
        t_ = time.time()
        # update the optimization objective function
        obj = 0 #### cost
        for i in range(N):
        #without angle error
            # kd=ca.atan2(xs[-n_tube:-n_tube+1]-X[i,6+n_tube],xs[-n_tube-1:-n_tube]-X[i,5+n_tube])
            # ktheta=ca.atan2((X[i,1]+X[i,4])/2-X[i,6+n_tube],(X[i,0]+X[i,3])/2-X[i,5+n_tube])
            # obj = obj +ca.mtimes([X[i, -2*n_tube:]-xs.T, Q, (X[i, -2*n_tube:].T-xs)])\
            obj = obj +ca.mtimes(getcenter(X[i,:],n_tube)-xs.T,(getcenter(X[i,:],n_tube).T-xs))\
            + ca.mtimes([U[i, :], R, U[i, :].T])
            # central avoidence
            K=3*(1+sf)/3/(1-(tanh(3))**2)*np.linalg.norm(((x0[:2]+x0[3:5])/2+x0[5+n_tube:n_tube+7])/2-xs,2)/getdOk(((x0[:2]+x0[3:5])/2+x0[5+n_tube:n_tube+7])/2)
            obj=obj-K*getd(getcenter(X[i,:],n_tube))

            # for k in range(n_tube):
            #     Ks[k]=(1+sf)/3/(1-(tanh(3))**2)*np.linalg.norm(((x0[:2]+x0[3:5])/2+x0[5+n_tube:n_tube+7])/2-xs,2)/getdOk(x0[6+2*k:6+2*k+2])
            #     obj=obj-Ks[k]*getd(X[i,6+2*k:6+2*k+2])
            # # virtual middle point to avoid the obstacles
            # Kk=(1+sf)/3/(1-(tanh(3))**2)*np.linalg.norm(((x0[:2]+x0[3:5])/2+x0[5+n_tube:n_tube+7])/2-xs,2)/getdOk((x0[:2]+x0[3:5])/2)
            # obj=obj-Kk*getd((X[i,:2]+X[i,3:5])/2)
            # for k in range(2):
            #     Kt=0
            #     for ki in range(n_tube):
            #         Kt=Kt+Ks[ki]*getdM(x0[6+2*ki:6+2*ki+2],Matrix.J,ki,k)
            # #     if Kt<=2*LA.norm(np.dot((((x0[:2]+x0[3:5])/2+x0[5+n_tube:n_tube+7])/2-xs).reshape(1,2*n_tube),Matrix.J[:,2*k:2*k+2]))*(1+sf):
            # #         Kt=2*LA.norm(np.dot((((x0[:2]+x0[3:5])/2+x0[5+n_tube:n_tube+7])/2-xs).reshape(1,2*n_tube),Matrix.J[:,2*k:2*k+2]))*(1+sf)
            #     K[k]=Kt/getdOk(x0[3*k:3*k+2])/(1-(tanh(3))**2)/3
            #     obj=obj-K[k]*getd(X[i,3*k:3*k+2])


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
        x0old=x0.copy()
        #real system model
        t0, x0, u0 = shift_movement(T, t0, x0, u_sol.toarray(), f,un)
        xold=np.array(Tube.get_points(n_tube)).ravel()
        Tube.update(x0[:2],x0[3:5])
        xt=np.array(Tube.get_points(n_tube)).ravel()
        deltap=np.vstack((x0[:2]-x0old[:2],x0[3:5]-x0old[3:5]))
        deltas=xt-xold
        error.append(deltas.reshape(2*n_tube,1)-np.dot(Matrix.J,deltap))
        xp.append(Tube.getshape())
        #Matrix.unpdateJ(x0[:2].reshape(2,1),x0[3:5].reshape(2,1),xtn=xt.reshape(2*n_tube,1))
        #[X,ff]=gene_ff(X,U,P,f,Matrix.J)
        x0[6:]=np.reshape(xt,(len(xt),1))
        xx.append(x0.copy())
        #x0 = np.reshape(x0,(-1, 1))
        mpciter = mpciter + 1
        Time.append(mpciter )

    current_time = datetime.datetime.now()
    print(current_time)
    deltat=current_time-begin_time
    print('cost time',deltat)
    t_v = np.array(index_t)
    print(mpciter)
    print(((x0[:2]+x0[3:5])/2+x0[5+n_tube:n_tube+7])/2-xs)
    e=np.stack(error).reshape(len(error),2*n_tube)
    e1=[]
    for i in range(len(error)):
        e1.append(LA.norm(e[i,:]))
    draw_result =Draw_MPC_two_agents_obstacles_Jerror_pulling(rob_diam=rob_diam, init_state=x0, target_state=np.array(xs), robot_states=xx,curve_state=xp,obs_rectangle=Plot.obs_rectangle,obs_circle=Plot.obs_circle,J_error=e1,Time=Time,Target_circle=Target_circle,n_tube=n_tube)
    #draw_result =Draw_MPC_two_agents_obstacles_Jerror(rob_diam=rob_diam, init_state=x0, target_state=np.array(xs), robot_states=xx,curve_state=xp,obs_rectangle=Plot.obs_rectangle,obs_circle=Plot.obs_circle,J_error=e1,Time=Time)