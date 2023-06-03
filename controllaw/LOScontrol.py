import time
import math
import numpy as np
class controllaw:
    def __init__(self, Kv, Kw):
        self.Kv = Kv
        self.Kw = Kw
        self.sample_time = 0.01
        self.v_max=5
        self.w_max=1
        self.current_time = time.time()
        self.last_time = self.current_time
        self.clear()
    def clear(self):
        self.Setx = 0.0
        self.Sety=0.0
        self.v_output = 0.0
        self.w_output=0.0

    def update(self, feedback_x, feedback_y,theta):
        p_error = np.sqrt((self.Setx - feedback_x)**2+(self.Sety - feedback_y)**2)[0]
        theta_error=math.atan2(self.Sety - feedback_y,self.Setx - feedback_x)-theta
        self.current_time = time.time()
        delta_time = self.current_time - self.last_time
        if (delta_time >= self.sample_time):
            self.v_output=self.Kv *p_error
            if self.v_output>self.v_max :
                self.v_output=self.v_max
            self.w_output=self.Kw*theta_error
            if self.w_output>self.w_max :
                self.v_output=self.v_max
            elif self.w_output<-self.w_max:
                self.w_output=-self.w_max
            self.last_time = self.current_time


    def setSampleTime(self, sample_time):
        self.sample_time = sample_time    