# planning two path directly 
#polynomial basic function fitting
#problem: the distance constraints between two agents 
#line 84-87 cause exit : Invalid number in NLP function or derivative detected.
import casadi as ca
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import env 
from scipy.spatial.distance import cdist

def sigmoid(x):
    return 1 / (1 + ca.exp(-x))

# Degree of interpolating polynomial
d = 7
# Time horizon
T = 10.
# Control discretization
N = 20 # number of control intervals
h = T/N
#length of the tube
L=30
#safety clearance sf
sf=5
#  env
Plot=env.Plotting()
bounry_points=Plot.env.boun_point
obs_points=Plot.env.obs_point
obs_diagonal_point=Plot.env.obs_diagonal_point

#calculate collision penalty
Q=ca.SX.sym('Q',2)
J=ca.SX.sym('J',1)
J=0
for i in range(0,len(obs_diagonal_point),2):
    #J0=ca.dot(obs_diagonal_point[i]-Q,obs_diagonal_point[i+1]-Q)
    J0_norm=ca.dot(obs_diagonal_point[i]-Q,obs_diagonal_point[i+1]-Q)/ca.norm_2(obs_diagonal_point[i]-Q)/ca.norm_2(obs_diagonal_point[i+1]-Q)
    #J=ca.if_else(J0>=10,J,J+J0)
    #J=ca.if_else(J0_norm>=1.2,J+1.2,J+J0_norm)
    J0_norm=sigmoid(5/3*J0_norm+10/3)
    J=J+J0_norm
    #J=J+J0
F=ca.Function('F',[Q],[J])

# Start with an empty NLP
w=[]
w0 = []
lbw = []
ubw = []
J = 0
g=[]
lbg = []
ubg = []
# "Lift" initial conditions
X_initial = [20,20]
X_target=[170,170]
Y_initial = [20,25]
Y_target=[170,175]
# coefficient of the length and area
a1=1
a2=0.05
a3=2000
Xlast=X_initial
Ylast=Y_initial
Cx=ca.SX.sym('C',2*d)
Cy=ca.SX.sym('C',2*d)
w.append(Cx)
w.append(Cy)
w0.append(np.zeros(4*d)) #initial value
lbw.append(np.ones(4*d)*-100)
ubw.append(np.ones(4*d)*100)
Cx=ca.reshape(Cx,(2,d))
Cy=ca.reshape(Cy,(2,d))
# Formulate the NLP
for k in range(N+1):
    # coeeficient variable
    t=h*(k)
    Xk=ca.mtimes(Cx[0:2,:],env.polynonial(t,d))
    dXk=ca.mtimes(Cx[0:2,:],env.dpolynonial(t,d))
    Yk=ca.mtimes(Cy[0:2,:],env.polynonial(t,d))
    dYk=ca.mtimes(Cy[0:2,:],env.dpolynonial(t,d))
    #Dk=ca.if_else(Xk>Yk,Xk-Yk,Yk-Xk)
    Dk=ca.norm_2(Xk-Yk)
    g.append(Dk)
    lbg.append([L**2/25])
    ubg.append([L**2])
  
    g.append(Xk)
    lbg.append([2,2])
    ubg.append([190,190])
    g.append(Yk)
    lbg.append([2,2])
    ubg.append([190,190])

    J1=(dXk[0]**2+dXk[1]**2)*h+(dYk[0]**2+dYk[1]**2)*h
  
    J3=F(Xk)+F(Yk)
    J=J+a1*J1-a3*J3
    Xlast=Xk
    Ylast=Yk
    if k==0:
        g.append(X_initial-Xk)
        lbg.append([0, 0])
        ubg.append([0, 0])
    if k==0:
        g.append(Y_initial-Yk)
        lbg.append([0, 0])
        ubg.append([0, 0])

# Add equality constraint
g.append(X_target-Xk)
lbg.append([0, 0])
ubg.append([0, 0])
# Add equality constraint
g.append(Y_target-Yk)
lbg.append([0, 0])
ubg.append([0, 0])

# Concatenate vectors
w = ca.vertcat(*w)
g = ca.vertcat(*g)
w0 = np.concatenate(w0)
lbg = np.concatenate(lbg)
ubg = np.concatenate(ubg)
lbw = np.concatenate(lbw)
ubw = np.concatenate(ubw)

# Create an NLP solver
prob = {'f': J, 'x': w, 'g': g}
# Create an NLP solver, using SQP and active-set QP for accurate multipliers
#opts = dict(qpsol='qrqp', qpsol_options=dict(print_iter=False,error_on_fail=False), print_time=False)
solver = ca.nlpsol('solver', 'ipopt', prob)

# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
C=sol['x']
# the coefficient matrix of x
CM=C[0:2*d]
CMy=C[2*d:]
CM=ca.reshape(CM,(2,d))
CMy=ca.reshape(CMy,(2,d))
x_plot = []
y_plot = []
for k in range(N+1):
    t=k*h
    X=ca.mtimes(CM,env.polynonial(t,d)) 
    Y=ca.mtimes(CMy,env.polynonial(t,d)) 
    x_plot.append(np.array(X[0:2]))
    y_plot.append(np.array(Y[0:2]))   
Total=sol['f']
x_plot = np.concatenate(x_plot)
y_plot = np.concatenate(y_plot)
#d_plot = np.concatenate(d_plot)

x1_plot=x_plot[0::2]
x2_plot=x_plot[1::2]
l1_plot=y_plot[0::2]
l2_plot=y_plot[1::2]
Plot.plot_grid('test')
Plot.plotpathsingle(x1_plot,x2_plot,l1_plot,l2_plot)
 

