import matplotlib.pyplot as plt
import time
import csv
class Plotupdate():
    def __init__(self,name,yname,num):
        #Set up plot
        self.figure, self.ax = plt.subplots()
        self.figure.set_figwidth(4) 
        self.figure.set_figheight(2) 
        self.line_objects = []
        self.num=num
        for i in range(num):
            line, = self.ax.plot([], [], label=f'Line {i}')
            self.line_objects.append(line)
        self.starttime= time.time()
        self.xdata=[]
        self.ax.set_autoscaley_on(True)
        self.ax.grid()
        self.figure.suptitle(name)
        self.ax.legend(loc='upper left')
        self.ax.set_xlabel('time(s)',fontsize=8)
        self.ax.set_ylabel(yname,fontsize=8)
        self.figure.suptitle(name,fontsize=8)
        self.ax.legend(loc='upper left')
        self.ax.tick_params(axis='x', labelsize=8)
        self.ax.tick_params(axis='y', labelsize=8)
    def on_running(self, error):
        timec = time.time()
        self.xdata.append(timec-self.starttime)
        for i in range(self.num):
            self.line_objects[i].set_xdata(self.xdata)
            self.line_objects[i].set_ydata(error[i])
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
    def save_plot(self,figname,dataname):
        plt.savefig(figname)
        fp = open(dataname, mode='a+', newline='')
        dp = csv.writer(fp)
        for i in range(self.num):
            dp.writerow(self.line_objects[i].get_xdata())
            dp.writerow(self.line_objects[i].get_ydata())   
        fp.close() 