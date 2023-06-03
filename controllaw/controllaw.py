import time
import math
import numpy as np
class controllaw:
    def __init__(self, Kv, Kw):
        self.Kv = Kv
        self.Kw = Kw
        self.sample_time = 0.00
        self.current_time = time.time()
        self.last_time = self.current_time
        self.clear()
    def clear(self):
        self.Setx = 0.0
        self.Sety=0.0
        self.last_p_error = 0.0
        self.last_theta_error = 0.0
        self.v_output = 0.0
        self.w_output=0.0

    def update(self, feedback_x, feedback_y):
        p_error = np.sqrt((self.Setx - feedback_x)**2+(self.Sety - feedback_y)**2)
        theta_error=math.atan2(self.Sety - feedback_y,self.Setx - feedback_x)
        self.current_time = time.time()
        delta_time = self.current_time - self.last_time
        if (delta_time >= self.sample_time):
            self.v_output=self.Kv *(p_error*math.cos(theta_error)-self.last_p_error*math.cos(self.last_theta_error))
            self.w_output=self.Kw* (math.sin(theta_error)*math.cos(theta_error)-math.sin(self.last_theta_error)*math.cos(self.last_theta_error))
            +self.Kw*(theta_error-self.last_theta_error)
            self.last_time = self.current_time
            self.last_p_error = p_error
            self.last_theta_error=theta_error


    def setSampleTime(self, sample_time):
        self.sample_time = sample_time    