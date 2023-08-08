import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve

def calculate_circle_line_intersection(center, radius, startPoint, endPoint):
    # 计算圆与线段的交点

    x, y = symbols('x y', real=True)

    # 圆的方程
    circleEquation = (x - center[0])**2 + (y - center[1])**2 - radius**2

    # 线段的方程
    if startPoint[0] - endPoint[0] != 0:
        lineEquation = (y - startPoint[1]) - ((startPoint[1] - endPoint[1]) / (startPoint[0] - endPoint[0])) * (x - startPoint[0])
    else:
        lineEquation = x - startPoint[0]

    # 求解交点
    intersectionPoints = solve([circleEquation, lineEquation], (x, y))

    # 提取交点坐标
    intersectionPoints = [[float(point[0].evalf()), float(point[1].evalf())] for point in intersectionPoints]

    P = []
    for point in intersectionPoints:
        if ((point[0] - startPoint[0]) * (point[0] - endPoint[0])) <= 0 and ((point[1] - startPoint[1]) * (point[1] - endPoint[1])) <= 0:
            P.append(point)

    return P
points = np.array([[30, 30], [70, 30], [70, 70], [30, 70]])
x = np.linspace(0, 100, 50)
y = np.linspace(0, 100, 50)
R = 25  # 感知半径
Ctheta = np.zeros((len(x), len(y)))

for i in range(len(x)):
    for j in range(len(y)):
        V = np.array([x[i], y[j]])

        if V[0] < points[1, 0] and V[0] > points[0, 0] and V[1] < points[3, 1] and V[1] > points[0, 1]:
            Ctheta[i, j] = -1
        else:
            # 计算交点
            P = []
            for n in range(3):
                intersectionPoints = calculate_circle_line_intersection(V, R, points[n, :], points[n + 1, :])
                if len(intersectionPoints)!= 0:
                    P.append(intersectionPoints)
            
            intersectionPoints = calculate_circle_line_intersection(V, R, points[3, :], points[0, :])
            if len(intersectionPoints) != 0:
                P.append(intersectionPoints)
            
            if len(P) > 0 and len(P[0]) == 2:
                if P[0][0][0] == P[1][0][0] or P[0][1][0] == P[1][1][0]:
                    Ctheta[i, j] = Ctheta[i, j] + np.dot(P[0][0] - V, P[0][1] - V) / np.linalg.norm(P[0][0] - V) / np.linalg.norm(P[0][1] - V)
                else:
                    dmin = 100000000
                    for m in range(4):
                        if np.linalg.norm(points[m, :] - V) < dmin:
                            dmin = np.linalg.norm(points[m, :] - V)
                            k = m
                    # 如果在角
                    if (V[0] - points[k, 0]) * (V[0] - P[0][0]) * (V[1] - points[k, 1]) * (V[1] - P[0][1]) > 0 and \
                            (V[0] - points[k, 0]) * (V[0] - P[1][0]) * (V[1] - points[k, 1]) * (V[1] - P[1][1]) > 0:
                        pass
                    else:
                        d1 = np.linalg.norm(points[k, :] - P[0][0])
                        d2 = np.linalg.norm(points[k, :] - P[0][1])
                        if d1 > d2:
                            P[1, :] = points[k, :]
                        else:
                            P[0, :] = points[k, :]

            if len(P) >= 2:
                if len(P) == 3:
                    for n in range(2):
                        Ctheta[i, j] = Ctheta[i, j] + np.dot(P[n, :] - V, P[2, :] - V) / np.linalg.norm(P[n, :] - V) / np.linalg.norm(P[2, :] - V)
                else:
                    Ctheta[i, j] = Ctheta[i, j] + np.dot(P[0][0] - V, P[1][0] - V) / np.linalg.norm(P[0][0] - V) / np.linalg.norm(P[1][0] - V)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, Ctheta.T)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('costheta')
ax.set_title('costheta')

plt.show()




