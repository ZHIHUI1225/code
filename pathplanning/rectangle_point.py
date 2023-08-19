from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math

def point_line_distance(point, start, end, is_segment=False):
    dx = end.x - start.x
    dy = end.y - start.y
    d = dx*dx + dy*dy
    t = ((point.x - start.x) * dx + (point.y - start.y) * dy) / d

    if not is_segment:
        p = (start.x + t * dx, start.y + t * dy)
    else:
        if d:
            if t < 0:
                p = start
            elif t > 1:
                p = end
            else:
                p = (start.x + t * dx, start.y + t * dy)
        else:
            p = start
        
    dx = point.x - p.x
    dy = point.y - p.y
    return math.sqrt(dx*dx + dy*dy)

class Rectangle:
    def __init__(self):
        self.points = np.array([[50, 30], [80, 30], [80, 70], [50, 70]])
        self.senseP=np.zeros((2,2))
        self.flag_in=True
        self.vector=[]
        self.vector.append(self.points[1]-self.points[0])
        self.vector.append(self.points[2]-self.points[3])
        self.vector.append(self.points[3]-self.points[0])
        self.vector.append(self.points[2]-self.points[1])
    def calculateP(self,P):
        # weather between vector 0 and 1
        if np.cross(self.vector[0],P-self.points[0])*np.cross(self.vector[1],P-self.points[3])<=0:
            if np.cross(self.vector[2],P-self.points[0])*np.cross(self.vector[3],P-self.points[1])<=0:
                self.flag_in=True
            elif np.cross(self.vector[2],P-self.points[0])>0:
                self.flag_in=False
                self.senseP[0,:]=self.points[0]
                self.senseP[1,:]=self.points[3]
            else:
                self.flag_in=False
                self.senseP[0,:]=self.points[1]
                self.senseP[1,:]=self.points[2]
        elif np.cross(self.vector[0],P-self.points[0])<0:
            self.flag_in=False
            if np.cross(self.vector[2],P-self.points[0])*np.cross(self.vector[3],P-self.points[1])<=0:
                self.senseP[0,:]=self.points[0]
                self.senseP[1,:]=self.points[1]
            elif np.cross(self.vector[2],P-self.points[0])>0:
                self.senseP[0,:]=self.points[3]
                self.senseP[1,:]=self.points[1]
            else:
                self.senseP[0,:]=self.points[2]
                self.senseP[1,:]=self.points[0]
        else:
            self.flag_in=False
            if np.cross(self.vector[2],P-self.points[0])*np.cross(self.vector[3],P-self.points[1])<=0:
                self.senseP[0,:]=self.points[3]
                self.senseP[1,:]=self.points[2]
            elif np.cross(self.vector[2],P-self.points[0])>0:
                self.senseP[0,:]=self.points[0]
                self.senseP[1,:]=self.points[2]
            else:
                self.senseP[0,:]=self.points[3]
                self.senseP[1,:]=self.points[1]

R=Rectangle()
x = np.linspace(0, 100, 50)
y = np.linspace(0, 100, 50)
Ctheta = np.zeros((len(x), len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        V = np.array([x[i], y[j]])
        R.calculateP(V)
        if R.flag_in is False:
            Ctheta[i,j]= np.dot(R.senseP[0,:] - V, R.senseP[1,:]  - V) / np.linalg.norm(R.senseP[0,:] - V) / np.linalg.norm(R.senseP[1,:]  - V)
        else:
            Ctheta[i,j]=-1
# Create a meshgrid
X, Y = np.meshgrid(x, y)

# Transpose Ctheta
Ctheta_t = np.transpose(Ctheta)

# Create the figure and plot the mesh
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Ctheta_t,cmap='viridis', edgecolor='none')
ax.set_title('3D line plot')
plt.show()