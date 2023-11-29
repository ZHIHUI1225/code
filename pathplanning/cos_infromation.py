from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math

def point_line_distance(point, start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1] 
    d = dx*dx + dy*dy
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / d
    if d:
        if t < 0:
            p = start
        elif t > 1:
            p = end
        else:
            p = np.array([start[0]+ t * dx, start[1] + t * dy])
    else:
            p = start
        
    dx = point[0]- p[0]
    dy = point[1] - p[1]
    distence=math.sqrt(dx*dx + dy*dy)
    return [distence,p]

class Rectangle:
    def __init__(self):
        self.points = np.array([[50, 50], [80, 50], [80, 70], [50, 70]])
        self.senseP=np.zeros((2,2))
        self.flag_in=True
        self.flag_side=True
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
                self.flag_side=True
            else:
                self.flag_in=False
                self.senseP[0,:]=self.points[1]
                self.senseP[1,:]=self.points[2]
                self.flag_side=True
        elif np.cross(self.vector[0],P-self.points[0])<0:
            self.flag_in=False
            if np.cross(self.vector[2],P-self.points[0])*np.cross(self.vector[3],P-self.points[1])<=0:
                self.senseP[0,:]=self.points[0]
                self.senseP[1,:]=self.points[1]
                self.flag_side=True
            elif np.cross(self.vector[2],P-self.points[0])>0:
                self.senseP[0,:]=self.points[3]
                self.senseP[1,:]=self.points[1]
                self.flag_side=False
            else:
                self.senseP[0,:]=self.points[2]
                self.senseP[1,:]=self.points[0]
                self.flag_side=False
        else:
            self.flag_in=False
            if np.cross(self.vector[2],P-self.points[0])*np.cross(self.vector[3],P-self.points[1])<=0:
                self.senseP[0,:]=self.points[3]
                self.senseP[1,:]=self.points[2]
                self.flag_side=True
            elif np.cross(self.vector[2],P-self.points[0])>0:
                self.senseP[0,:]=self.points[0]
                self.senseP[1,:]=self.points[2]
                self.flag_side=False
            else:
                self.senseP[0,:]=self.points[3]
                self.senseP[1,:]=self.points[1]
                self.flag_side=False

    def EGO(self,P):
        if np.cross(self.vector[0],P-self.points[0])*np.cross(self.vector[1],P-self.points[3])<=0 and np.cross(self.vector[2],P-self.points[0])*np.cross(self.vector[3],P-self.points[1])<=0:
            self.flag_in=True
        else:
            self.flag_in=False
        d=np.zeros((4,1))
        p=np.zeros((4,2))
        for i in range(3):
            [d[i],p[i,:]]=point_line_distance(P, self.points[i], self.points[i+1])
        [d[3],p[3,:]]=point_line_distance(P, self.points[3], self.points[0])
        min_value = np.min(d)
        min_index = np.argmin(d)
        if self.flag_in is True:
            min_value=-min_value
        return [min_value,p[min_index,:]]

R=Rectangle()
x = np.linspace(20, 100, 80)
y = np.linspace(20, 100, 80)
Ctheta = np.zeros((len(x), len(y)))
dCtheta=np.zeros((len(x), len(y)))
Ctanh = np.zeros((len(x), len(y)))
dCtanh=np.zeros((len(x), len(y)))
dCtanh2=np.zeros((len(x), len(y)))
F=np.zeros((len(x), len(y)))
EGOJ=np.zeros((len(x), len(y)))
Sf=0.5
Smin=0.2
sf=15
for i in range(len(x)):
    for j in range(len(y)):
        V = np.array([x[i], y[j]])
        R.calculateP(V)
        if R.flag_side is True:
            K=3 # range 2/3 pi ~ pi
        else:
            K=6 # range pi/3~pi/2
        [d,points]=R.EGO(V)
        c=sf-d
        if c<=0:
            EGOJ[i,j]=0
        elif c>0 and c<=sf:
            EGOJ[i,j]=c**3
        else:
            EGOJ[i,j]=3*sf*c**2-3*sf**2*c+sf**3
        if R.flag_in is False:
            Ctheta[i,j]= np.dot(R.senseP[0,:] - V, R.senseP[1,:]  - V) / np.linalg.norm(R.senseP[0,:] - V) / np.linalg.norm(R.senseP[1,:]  - V)
            l1=np.linalg.norm(R.senseP[0,:] - V)
            l2=np.linalg.norm(R.senseP[1,:] - V)
            z1=(R.senseP[0,:] - V)/ l1
            z2=(R.senseP[1,:] - V) / l2
            dCtheta_vector=(1/l2-Ctheta[i,j]/l1)*z1+(1/l1-Ctheta[i,j]/l2)*z2
                 
            if R.flag_side is True:
                if Ctheta[i,j]>0.5:
                    F[i,j]=-1
                    dCtheta[i,j]=0
                elif Ctheta[i,j]<0:
                    F[i,j]=1
                    dCtheta[i,j]=0
                else:
                    # F[i,j]=3*Ctheta[i,j]-4*Ctheta[i,j]**3
                    dCtheta[i,j]=(3-12*Ctheta[i,j]**2)*np.linalg.norm(dCtheta_vector)  
                    F[i,j]=-(-1+2*Ctheta[i,j]**2)*(16*Ctheta[i,j]**4-16*Ctheta[i,j]**2+1)
                    dCtheta[i,j]=-(4*Ctheta[i,j]*(16*Ctheta[i,j]**4-16*Ctheta[i,j]**2+1)+(2*Ctheta[i,j]**2-1)*(16*4*Ctheta[i,j]**3-32*Ctheta[i,j]))*np.linalg.norm(dCtheta_vector)   
            else:
                if Ctheta[i,j]>math.sqrt(3)/2:
                    F[i,j]=-1
                    dCtheta[i,j]=0
                elif Ctheta[i,j]<0.5:
                    F[i,j]=1
                    dCtheta[i,j]=0
                else:
                    F[i,j]=(-1+2*Ctheta[i,j]**2)*(16*Ctheta[i,j]**4-16*Ctheta[i,j]**2+1)
                    dCtheta[i,j]=(4*Ctheta[i,j]*(16*Ctheta[i,j]**4-16*Ctheta[i,j]**2+1)+(2*Ctheta[i,j]**2-1)*(16*4*Ctheta[i,j]**3-32*Ctheta[i,j]))*np.linalg.norm(dCtheta_vector)   
            
        else:
            F[i,j]=1
        # Ctanh[i,j]=math.tanh(3/(Sf-Smin)*(Ctheta[i,j]-Smin))

        # dCtanh[i,j]=3/(Sf-Smin)*(1-np.square(Ctanh[i,j]))*dCtheta[i,j]
        # dCtanh2[i,j]=3/(Sf-Smin)*(1-np.square(math.tanh(2)))*dCtheta[i,j]
# Create a meshgrid
X, Y = np.meshgrid(x, y)

# Transpose Ctheta
# Ctheta_t = np.transpose(Ctheta)
# Ctheta_t = np.transpose(Ctanh)
Ctheta_t=np.transpose(dCtheta)
# Ctheta_t = np.transpose(dCtanh)
# Ctheta_t = np.transpose(dCtanh2)
# Ctheta_t = np.transpose(EGOJ)
F_t = np.transpose(F)

# Create the figure and plot the mesh
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, F_t,cmap='viridis', edgecolor='none')
# ax.set_title('avoidance term')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Ctheta_t,cmap='viridis', edgecolor='none')
ax.set_title('pushing force')
ax.set_title('tanhcos')
ax.set_title('derivative of tanhcos')
plt.show()