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
R = 20  # 感知半径
V = np.array([20, 50])
if V[0] < points[1, 0] and V[0] > points[0, 0] and V[1] < points[3, 1] and V[1] > points[0, 1]:
    pass
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
    P=np.reshape(P,(-1,2))
    if P[1,0] == P[0,0] or P[0,1] == P[1,1]:
        Ctheta = np.dot(P[0,:] - V, P[1,:] - V) / np.linalg.norm(P[0,:] - V) / np.linalg.norm(P[1,:] - V)
    else:
        dmin = 100000000
        for m in range(4):
            if np.linalg.norm(points[m, :] - V) < dmin:
                dmin = np.linalg.norm(points[m, :] - V)
                k = m
        # 如果在角
        if (V[0] - points[k, 0]) * (V[0] - P[0,0]) * (V[1] - points[k, 1]) * (V[1] - P[0,1]) > 0 and \
                (V[0] - points[k, 0]) * (V[0] - P[1,0]) * (V[1] - points[k, 1]) * (V[1] - P[1,1]) > 0:
            pass
        else:
            d1 = np.linalg.norm(points[k, :] - P[0,:])
            d2 = np.linalg.norm(points[k, :] - P[1,:])
            if d1 > d2:
                P[0,:] = points[k, :]
            else:
                P[1,:] = points[k, :]
print(P)
