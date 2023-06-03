import numpy as np
from numpy import linalg as LA
from tube_sineshape import tubeshape
from sklearn.linear_model import LinearRegression

class Jmatrix():
    def __init__(self,L:float,T:int,n_tube:int,p1:np.array,p2=np.array):
        self.T=T #sampling date
        self.L=L # the length of tube
        self.n_tube=n_tube # the points of the tube
        self.e=0.0003 #error
        self.j=0
        self.p1=p1
        self.p2=p2
        [delta,R,C]=self.initialJ(self.T,self.n_tube,self.L,self.p1,self.p2)
        self.J=C
        # give the sine model of tube
        self.Tube=tubeshape(length=L,p1=self.p1,p2=self.p2)
         # initial matrix Q and delta R
        self.delta=delta
        self.R=R
         # the initial position of the tube
        xt=np.array(self.Tube.get_points(self.n_tube)).ravel()
        self.xt=np.reshape(xt,(len(xt),1))
    @staticmethod
    def initialJ(T,n,L,p10,p20):
    #points of tube n 
        Tube=tubeshape(length=L,p1=p10,p2=p20)
        x=[]
        y=[]
        p1=p10
        p2=p20
        tp=Tube.get_points(n)
        for i in range(T):
            #random move
            deltap=1*np.random.uniform(-1, 1, size=(1, 4))
            p1=p1+np.reshape(deltap[0][:2],(2,1))
            p2=p2+np.reshape(deltap[0][2:],(2,1))
            Tube.update(p1,p2)
            tn=Tube.get_points(n)
            y.append(np.reshape(np.array(tn)-np.array(tp),(1,n*2))[0])
            x.append(deltap[0])
            tp=tn
        # Find the coefficients (m and b) of the line y = mx + b that best fits the data
        model = LinearRegression().fit(x, y)
        delta=np.array(y)#size 8*6 T*P
        R=np.array(x)#size 8*4 T*M
        return [delta,R,model.coef_]
        

    def unpdateJ(self,p1,p2,xtn):
        gamma=0.000005
        # delat r
        deltap=np.concatenate(((p1-self.p1).reshape(1,2),(p2-self.p2).reshape(1,2)), axis=1)
        self.p1=p1
        self.p2=p2
        # self.Tube.update(p1,p2)
        # # the change of tube
        # xtn=np.array(self.Tube.get_points(self.n_tube)).ravel().reshape(2*self.n_tube,1)
        #delta s
        delta_t=xtn-self.xt
        self.xt=xtn
        for i in range(2*self.n_tube):
            self.j=LA.norm(np.dot(deltap.reshape(1,4),self.J[i,:].reshape(4,1))-delta_t[i])**2/2+LA.norm(np.dot(self.R,self.J[i,:].reshape(4,1))-self.delta[:,i].reshape(self.T,1))**2/2
            if self.j>self.e:
                qn=self.J[i,:].reshape(4,1)\
                -gamma*(np.dot(self.R.T,np.dot(self.R,self.J[i,:].reshape(4,1))-self.delta[:,i].reshape(self.T,1))+np.dot(deltap.reshape(4,1),np.dot(deltap.reshape(1,4),self.J[i,:].reshape(4,1))-delta_t[i]).reshape(4,1))
                self.J[i,:]=qn.reshape(1,4)
        # update R
        self.R=np.vstack((self.R[len(deltap):,:],deltap))
         #update delta
        self.delta=np.vstack((self.delta[len(delta_t.T):,:],delta_t.T))
