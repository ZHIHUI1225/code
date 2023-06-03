# test the update J Matrix 
from tube_sineshape import tubeshape
from updateJmatrix import Jmatrix
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

v_max = 0.8
omega_max = np.pi/10
n_tube=3 # the points of the tube
L=30
p1=np.array([[0],[0]])
p2=np.array([[0],[20]])
Matrix=Jmatrix(T=8,L=L,n_tube=n_tube,p1=p1,p2=p2)
# Matrix.J=0.3*np.eye(2*n_tube,4)
Tube=tubeshape(length=L,p1=p1,p2=p2)
time=[]
error=[]
j=[]
N=4000
t=0
for i in range(N): 
    deltap=1*np.random.uniform(-1, 1, size=(1, 4)) 
    p1=p1+np.reshape(deltap[0][:2],(2,1))
    p2=p2+np.reshape(deltap[0][2:],(2,1))
    if LA.norm(p1-p2)>=L:
        p2=p2-np.reshape(deltap[0][2:],(2,1))
    if Tube.get_a()==0:
        continue
    told=np.array(Tube.get_points(n_tube))
    Tube.update(p1,p2)
    if Tube.get_a()==0:
        continue
    tn=np.array(Tube.get_points(n_tube))
    deltas=tn-told
    Matrix.unpdateJ(p1,p2,tn.reshape(6,1))
    error.append(deltas.reshape(6,1)-np.dot(Matrix.J,deltap.T))
    j.append(Matrix.j)
    t=t+1
    time.append(t)
e=np.stack(error).reshape(len(error),2*n_tube)
time=np.stack(time)
fig1, ax1 = plt.subplots()
for i in range(6):
    plt.plot(time,e[:,i])
ax1.set_title('e')

fig2, ax2 = plt.subplots()
ax2.plot(time,j)
ax2.set_title('j')
plt.show()
