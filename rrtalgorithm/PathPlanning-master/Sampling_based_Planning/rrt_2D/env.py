"""
Environment for rrt_2D
@author: huiming zhou
"""


class Env:
    def __init__(self):
        self.x_range = (0, 50)
        self.y_range = (0, 30)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()

    @staticmethod
    def obs_boundary():
        obs_boundary = [
            [0, 0, 1, 30],
            [0, 30, 50, 1],
            [1, 0, 50, 1],
            [50, 1, 1, 30]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle(): #ox, oy, w, h
        obs_rectangle = [

        ]
        return obs_rectangle

    @staticmethod
    def obs_circle(): #x, y, r
        obs_cir = [
            [7, 12, 3],
            [43, 22, 5],
        ]

        return obs_cir
