# planning without obstacle
# distance between two agents is constant 
import numpy as np
import math
from casadi import *
import matplotlib.pyplot as plt

class trajectory:
    def __init__(self, order,L,step):
          self.order=order #the order of polynomial of natural basis 1,t,...,t**order
          max=1.1
          self.t=np.arange(0,max,step)
          self.x=np.arange(0,max,step)
          self.y=np.arange(0,max,step)
          self.n=self.t.size
          self.d=[[L/2] for _ in range(self.t.size)]
          self.dy=[[L/2] for _ in range(self.t.size)]
          self.L=L
          self.area=np.multiply(self.d,self.dy)
          self.c=MX.zeros(3,order+1)

    def updatexy(self):
        P=MX.zeros((3,self.n))
        for i in range(0,self.n):
            beta=np.array([1])
            for n in range(1,self.order+1):
                beta=np.append(beta,[self.t[i]**n])
            P[:,i]=mtimes(self.c,beta)
        self.x=P[0,:]
        self.y=P[1,:]
        self.d=P[2,:]

    def getarea(self):
        self.dy=np.sqrt([[self.L] for _ in range(self.n)]**2/4-self.d**2/4)
        self.area=np.multiply(self.d,self.dy)

    def plottraj(self):
        plt.figure()
        plt.plot(self.x, self.y, 'ro')
        plt.show()



P0=MX([0,0])
Pd0=0 #direction at P0
PT=MX([200,300])
PdT=1 #direction at PT
order=3
L=30 
step=0.1

#decision variables coeffient
c=MX.sym('c',3*(order+1),1)

P=trajectory(order,L,step)
P.c=reshape(c,(3,(order+1)))
P.updatexy()
# objective function length of trajectory
f1=np.sqrt((P.x[0]-P0[0])**2+(P.y[0]-P0[1])**2)
# objective function area of trajectory swept
f2=P.area[0]*f1
for i in range(1,P.n):
    df1=np.sqrt((P.x[i]-P.x[i-1])**2+(P.y[i]-P.y[i-1])**2)
    f1=f1+df1
    f2=f2+df1*P.area[i-1]

df1=sqrt((PT[0]-P.x[P.n-1])**2+(PT[1]-P.y[P.n-1])**2)
f1=f1+df1
f2=f2+df1*P.area[P.n-1]

a=MX([1,-1])
J=dot(a, vertcat(f1,f2))

f=Function('f',[c],[J])
nlp = {'x':c, 'f':f(c)}
S = nlpsol('S', 'ipopt', nlp)
print(S)
r = S(x0=MX.ones(3*(order+1),1),\
      lbg=0, ubg=0)
x_opt = r['x']
print('x_opt: ', x_opt)
#opti.minimize(J)

