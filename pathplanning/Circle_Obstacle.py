from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve
import math

def calculate_circle_line_intersection(center, radius,slope_k):
    # 计算圆与直径的交点

    x, y = symbols('x y', real=True)

    # 圆的方程
    circleEquation = (x - center[0])**2 + (y - center[1])**2 - radius**2

    # 线段的方程
    lineEquation = (y - center[1]) - slope_k* (x - center[0])


    # 求解交点
    intersectionPoints = solve([circleEquation, lineEquation], (x, y))

    # 提取交点坐标
    intersectionPoints = [[float(point[0].evalf()), float(point[1].evalf())] for point in intersectionPoints]

    P = []
    for point in intersectionPoints:
        P.append(point)

    return P


x = np.linspace(0, 80, 40)
y = np.linspace(0, 80, 40)
Ctheta = np.zeros((len(x), len(y)))
Ctanh = np.zeros((len(x), len(y)))
Sf=0.5
Smin=0.2
R = 20  # 感知半径
C = np.array([40, 40])
for i in range(len(x)):
    for j in range(len(y)):
        V = np.array([x[i], y[j]])
        if np.linalg.norm(V-C) <= R:
            pass
        else:
            K=-1*(C[0]-V[0])/(C[1]-V[1])
            P=np.array(calculate_circle_line_intersection(C,R,K))
            Ctheta[i,j]= np.dot(P[0,:] - V,P[1,:]  - V) / np.linalg.norm(P[0,:] - V) / np.linalg.norm(P[1,:]  - V)
        Ctanh[i,j]=math.tanh(3/(Sf-Smin)*(Ctheta[i,j]-Smin))


# Create a meshgrid
X, Y = np.meshgrid(x, y)

# Transpose Ctheta
Ctheta_t = np.transpose(Ctheta)
Cthetah_t = np.transpose(Ctanh)
# Create the figure and plot the mesh
fig = plt.figure()
ax1=fig.add_subplot(121,projection='3d')

# ax = plt.axes(projection='3d')
ax1.plot_surface(X, Y, Ctheta_t,cmap='viridis', edgecolor='none')
ax1.set_title('cos')
ax2=fig.add_subplot(122,projection='3d')
ax2.plot_surface(X, Y, Cthetah_t,cmap='viridis', edgecolor='none')
ax2.set_title('tanhcos')
# ax.set_title('derivative of tanhcos')
plt.show()