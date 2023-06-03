import numpy as np
import math
import casadi as ca
import matplotlib.pyplot as plt

def dlodynamics(xleft, yleft, xright, yright, L):
    opti=ca.Opti()
    #decision variables
    order=3
    n = opti.variable(order*2+2,1)
    #optimization fuction
    f=n[1]**2*L
    for i in range(1,order):
        f=f+n[2 * i ]**2 *(2 * math.pi * i / L)**2 * L / 2 + n[2 * i + 1]**2 * (2 * math.pi * i / L)**2 * L / 2
    opti.minimize(f)
    N = 100
    Lx = xright - xleft 
    Ly = yright - yleft
    lx = 0.00
    ly = 0.00
    for k in range(1,N):
        phi = n[0] + n[1] * L * k / N
        for i in range(1,order):
            phi = phi + n[2 * i ] * math.sin(2 * math.pi * i * k / N) + n[2 * i + 1] * math.cos(2 * math.pi * i * k / N)
        opti.subject_to(phi<math.pi*5/6)
        opti.subject_to(phi>-math.pi*5/6)
        lx = lx + phi.cos() * L / N
        ly = ly + phi.sin() * L / N
    theta1 = n[0]
    theta2 = n[0] + n[1] * L
    for i in range(1,order):
        theta1 = theta1 + n[2 * i+1]
        theta2 = theta2 + n[2 * i+1]
    u1=n[1]*L
    for i in range(1,order):
        u1=u1+n[2*i]*2*math.pi*i
    # Concatenate nonlinear constraints  
    opti.minimize(f)
    opti.subject_to(lx == Lx)
    opti.subject_to(ly == Ly)
    opti.subject_to(u1 == 0)
    opti.subject_to(theta1>0)
    opti.subject_to(theta2<0)
    #opti.subject_to(n[1]*L==math.atan2(Ly,Lx))

    n0=[0.10,0]*(order+1)
    opti.set_initial(n,n0)
    # ---- solve NLP              ------
    opti.solver("ipopt") # set numerical backend
    sol = opti.solve()   # actual solve
    opti.debug.value(n)
    para_n = sol.value(n)
    px = xleft
    py = yleft
    DLO = np.array([[px],[py]]) 
    DLOangle = np.array([0])
    numOfData = 100
    for k in range(1,numOfData):
        phi = para_n[0] + para_n[1] * L * k / numOfData
        for i in range(1,order):
            phi = phi + para_n[2 * i ] * math.sin(2 * math.pi * i * k / numOfData) + para_n[2 * i + 1] * math.cos(2 * math.pi * i * k / numOfData)
        px = px + math.cos(phi) * L / numOfData
        py = py + math.sin(phi) * L / numOfData
        DLO=np.concatenate((DLO,[[px],[py]]),axis=1)
        DLOangle=np.append(DLOangle,[phi],axis=0)
    return [DLO,DLOangle]

[DLO,DLOangle]=dlodynamics(0.00, 4.00,6.00, 3.00, 10)
x=DLO[0,:]
y=DLO[1,:]
plt.plot(x,y)
plt.show()