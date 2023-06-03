#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Pose
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from geometry_msgs.msg import Twist
import numpy as np
import cv2
import math
import Staticcontrol
import time
from scipy.optimize import fsolve
from scipy.special import ellipe
from dlodynamics import dlodynamics
import matplotlib.pyplot as plt 
from Dynamicupdateplot import DynamicUpdate

class TUBE:
    def __init__(self):
        self.tubex=[0.0]*5
        self.tubey=[0.0]*5
        self.tubez=[0.0]*5
        self.sub = rospy.Subscriber('/tube', PointCloud, self.tube_callback,queue_size=10)
    def tube_callback(self, msg): # tube msg
        i=0
        for point in msg.points:
            self.tubex[i]=point.x
            self.tubey[i]=point.y
            self.tubez[i]=point.z
            i=i+1
        # print(i,self.tubey[3])
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.points)


class QRrobot:
    def __init__(self,image=None,x=None,y=None,xmax=None,ymax=None):
        self.robotx=[0.0]*10
        self.roboty=[0.0]*10
        self.robotyaw=[0.0]*10
        self.robotID=[0]*10
        self.sub = rospy.Subscriber('/QR_1', Pose, self.pose_callback,queue_size=10)
        self.sub = rospy.Subscriber('/QR_2', Pose, self.pose_callback,queue_size=10)
        

    def pose_callback(self, msg): # feedback means actual value.
        #rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.position)
        if msg.position.z>=10:
            i=int(msg.position.z-10)
        else:
            i=int(msg.position.z) #ID
        self.robotx[i]=msg.position.x
        self.roboty[i]=msg.position.y
        self.robotID[i]=i
        self.robotyaw[i]=msg.orientation.w
        #print(self.robotID[i],"x",self.robotx[i],self.roboty[i])




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

def vtow(v,w):
    R=0.015
    L=0.08
    sample_time=0.01
    M=np.array([[1/R,L/(2*R)],[1/R,-L/(2*R)]])
    W=np.dot(M,np.array([[v],[w]]))
    wr=float(W[0])
    wl=float(W[1])
    return [wr,wl]

# calculate the error between the modal and tube
def cal_error():
    #length 30cm
    T=TUBE()
    L=0.3
    n=len(T.tubex)
    error=[0]*n
    if T.tubey[0]-T.tubey[n-1]+T.tubex[0]-T.tubex[n-1] !=0:
        [DLO,DLOangle]=dlodynamics(T.tubex[0], T.tubey[0], T.tubex[4], T.tubey[4],  T.tubez[0], T.tubez[4], L)
        l=DLOangle.size
        for i in range(0,n-1):
            error[i]=(T.tubex[i]-DLO[0,int((l+1)/n*i)])**2+(T.tubey[i]-DLO[0,int((l+1)/n*i)])**2
        error_sum=sum(error)
        print('error is',error)
        return error


if __name__ == '__main__':
    try:
        rospy.init_node('DETECT', anonymous=True)
        State = QRrobot()
        tube=TUBE()
        pub = rospy.Publisher('anglevelocity', Twist, queue_size=10)
        vel_msg=Twist()
        vel_msg.linear.x = 0
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.z=0 #ID
        rate = rospy.Rate(30)
        Kv=0.1
        Kw=0.05
        Yc=2
        Xc=2
        thetac=0
        curvature=1
        length=0.3
        N=int(10)
        target=get_targets(Xc,Yc,thetac,curvature,length)
        target_x=[3 for i in range(N)]
        target_y=[1.5 for i in range(N)]
        control= [Staticcontrol.controllaw(Kv,Kw) for _ in range(N)]
        error_x=[[0] for _ in range(N)]
        error_y=[[0] for _ in range(N)]
        wr=[[0] for _ in range(N)]
        wl=[[0] for _ in range(N)]
        errorplot=DynamicUpdate()

        while not rospy.is_shutdown():
            for i in range(0,10):
                if State.robotID[i]!=0:
                    control[i].Setx=target_x[i]
                    control[i].Sety=target_y[i]
                    control[i].setSampleTime(0.01)
                    errorx=target_x[i]-State.robotx[i]
                    errory=target_y[i]-State.roboty[i]
                    if np.sqrt(errorx**2+errory**2)>=0.05:
                        error=cal_error()
                        #print(error)
                        control[i].update(State.robotx[i],State.roboty[i],State.robotyaw[i])
                        output_v = control[i].v_output
                        output_w = control[i].w_output
                        wr=vtow(output_v,output_w)[0]
                        wl=vtow(output_v,output_w)[1]
                        vel_msg.angular.x= wr*26.362-26.451 #PWM 
                        vel_msg.angular.y= wl*26.362-26.451
                        vel_msg.angular.z=State.robotID[i]
                        rospy.loginfo(vel_msg)
                        pub.publish(vel_msg)
                        error_x[i].append(errorx)
                        error_y[i].append(errory)
                        rate.sleep()

                errorplot.on_running(error_x, error_y)
        
    except rospy.ROSInterruptException:
        pass
