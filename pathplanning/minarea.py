import numpy as np
import math
import casadi as ca
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.special import ellipe

def minarea():
    L=30 # length of tube is 30cm
    opti=ca.Opti()
    #decision variables
    x = opti.variable(1,1) #x,y
    y=np.sqrt(L**2/4-x**2/4)
    f=x*y
    opti.minimize(-f)
    opti.subject_to(x<=L)
    opti.subject_to(x>=0)
    opti.set_initial(x,L/2)
    opti.solver("ipopt") # set numerical backend
    sol = opti.solve()   # actual solve
    opti.debug.value(x)
    para_n = sol.value(x)
    return [para_n] 

[x]=minarea()
print(x)