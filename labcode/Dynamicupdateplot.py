import matplotlib.pyplot as plt
import numpy as np
import time
plt.ion()
class DynamicUpdate():
    #Suppose we know the x range
    #min_x = 0
    #max_x = 300

    def __init__(self):
        #Set up plot
        #self.figure, self.ax = plt.subplots()
        self.figure, (self.ax1, self.ax2) = plt.subplots(1, 2, sharey=True)
        self.lines1=[]
        self.lines2=[]
        for i in range(10):
            line, = self.ax1.plot([],[],label=str(i)) #error_x
            self.lines1.append(line)
            line, = self.ax2.plot([],[],label=str(i)) #error_x
            self.lines2.append(line)
        #self.lines1, = self.ax1.plot([],[]) #error_x
        #self.lines2, = self.ax2.plot([],[]) #error_y
        #Autoscale on unknown axis and known lims on the other
        self.ax1.set_autoscaley_on(True)
        #self.ax1.set_xlim(self.min_x, self.max_x)
        self.ax2.set_autoscaley_on(True)
        #self.ax2.set_xlim(self.min_x, self.max_x)
        #Other stuff
        self.ax1.grid()
        self.ax2.grid()
        self.ax1.legend(loc='upper left')
        self.ax2.legend(loc='upper left')

        ...

    def on_running(self, error_x, error_y):
        #Update data (with the new _and_ the old points)
        for i in range(10):
            xdata=np.arange(0,len(error_x[i]))
            self.lines1[i].set_xdata(xdata)
            self.lines1[i].set_ydata(error_x[i])
            ydata=np.arange(0,len(error_y[i]))
            self.lines2[i].set_xdata(ydata)
            self.lines2[i].set_ydata(error_y[i])
        #Need both of these in order to rescale
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
    
    def hold(self):
        self.figure.savefig("figure.jpg")

        
    '''
    #Example
    def __call__(self):

        xdata = []
        ydata = []
        for x in np.arange(0,10,0.5):
            xdata.append(x)
            ydata.append(np.exp(-x**2)+10*np.exp(-(x-7)**2))
            self.on_running(xdata, ydata)
            time.sleep(1)
        return xdata, ydata

d = DynamicUpdate()
d()
'''