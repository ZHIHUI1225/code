import LOScontrol 
import Staticcontrol
import time
import numpy as np
import math
from scipy.optimize import fsolve
from scipy.special import ellipe
from Dynamicupdateplot import DynamicUpdate

def  get_targets(Xc,Yc,thetac,curvature,length):

    def sin_fitiing(x):
        x0=float(x[0])
        a=float(x[1])
        l=length # length of the tube
        K=curvature # curvature of the peak point
        return [l-(2*math.sqrt(x0**2+a**2 *math.pi**2))/math.pi * ellipe(1/math.sqrt(1+x0**2/(a**2 *math.pi**2))),
        K-a*math.pi**2 /x0**2
    ]
    result = fsolve(sin_fitiing, [1,1])
    x0=result[0]
    a=result[1]
    # coordinate translation and rotation
    positionc=np.array([[Xc],[Yc]])
    M=np.array([[math.cos(thetac),math.sin(thetac)],[-math.sin(thetac),math.cos(thetac)]])
    X0=np.array([[x0],[0]])
    position0=positionc-np.dot(M,np.array([[x0/2],[a]]))
    positon_I=np.dot(M,X0)+position0
    # return the two end points n inertia coordinate
    return [position0,positon_I]
    

def kinematics(x,y,theta,v,w):
    R=0.015
    L=0.08
    sample_time=0.1
    M=np.array([[1/R,L/(2*R)],[1/R,-L/(2*R)]])
    W=np.dot(M,np.array([[v],[w]]))
    wr=float(W[0])
    wl=float(W[1])
    w_max=11
    w_min=2
    if wr>w_max:
        wr=w_max
    elif wr<-w_max:
        wr=-w_max
    elif wr<w_min and wr>-w_min:
        wr=0
    if wl>w_max:
        wl=w_max
    elif wl<-w_max:  
        wl=-w_max
    elif wl<w_min and wl>-w_min:
        wl=0
    x_new=x+R/2*(wr+wl)*math.cos(theta)*sample_time
    y_new=y+R/2*(wr+wl)*math.sin(theta)*sample_time
    theta_new=theta+R/L*(wr-wl)*sample_time
    if theta_new>math.pi:
        theta_new=-math.pi
    elif theta_new<-math.pi:
        theta_new=math.pi
    return [x_new,y_new,theta_new]

def test_control(Kv, Kw, target_x, target_y,initial_x,initial_y,initial_theta):
    control = LOScontrol.controllaw(Kv,Kw)
    control.Setx=target_x
    control.Sety=target_y
    control.setSampleTime(0.01)
    errorx=target_x-initial_x
    errory=target_y-initial_y
    error_x=[[0] for _ in range(10)]
    error_y=[[0] for _ in range(10)]
    x=initial_x
    y=initial_y
    theta=initial_theta
    v=0
    w=0
    d=DynamicUpdate()
    while errorx**2>=0.001 or errory**2>=0.001:
        control.update(x,y,theta)
        output_v = control.v_output
        output_w = control.w_output
        v=output_v
        w=output_w
        [x_new,y_new,theta_new]=kinematics(x,y,theta,v,w)
        x=x_new
        y=y_new
        theta=theta_new
        errorx=target_x-x
        errory=target_y-y
        error_x[0].append(errorx)
        error_y[0].append(errory)
        d.on_running(error_x, error_y)
        #print(error_x,error_y,v,w,theta)
        time.sleep(0.01)
    d.hold()
    return [x,y,theta]


Xc=0.4
Yc=0.2
thetac=0
curvature=1
length=0.3
target=get_targets(Xc,Yc,thetac,curvature,length)
print(target)
Kv=1
Kw=4
result=test_control(Kv,Kw,target[0][0],target[0][1],0.0,0.0,2)
print(np.array(result[0:2])-(np.array(target[0])).T[0])
