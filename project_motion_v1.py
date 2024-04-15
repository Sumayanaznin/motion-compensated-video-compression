import cv2 as cv
import numpy as np
import math
import random as rn


class Trajectory():
    def __init__(self, sid):
        self.sid = sid
        self.t_ref = np.asmatrix([0, 0, 1]).T
        self.x_origin = 1000
        self.y_origin = 300
        self.x_wsize = 1200
        self.y_wsize = 1200
        #self.odd_frame = 1
        self.r_ref = np.asmatrix(([1, 0, 0], [0, 1, 0], [0, 0, 1]))
        self.trajectory = np.zeros((self.x_wsize, self.y_wsize, 3), dtype=np.uint8)
        cv.line(self.trajectory, (0, self.y_origin), (self.x_wsize,self.y_origin),(0,255,0),1)
        cv.line(self.trajectory, (self.x_origin, 0), (self.x_origin, self.y_wsize), (0, 255, 0), 1)

    def process_motion_hypothesis(self, del_x, del_y, del_theta):
        theta = del_theta
        tx = del_x
        ty = del_y
        '''
        if self.t_ref is None or self.r_ref is None:
            self.t_ref = np.asmatrix([tx, ty, 1]).T
            self.r_ref = np.asmatrix(([math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]))
        else:'''
        t = np.asmatrix([tx, ty, 1]).T
        r = np.asmatrix(([math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]))
        self.t_ref = self.t_ref + self.r_ref * t
        self.r_ref = r * self.r_ref

        #print(r)
        #if self.odd_frame % 2 != 0:
        print("Tx: ", tx," Ty: ", ty, " Theta: ", theta)
        #self.odd_frame += 1
       # print("X:", self.t_ref[0, 0], " Y:", self.t_ref[1, 0])
        x = round(self.t_ref[0, 0])
        y = round(self.t_ref[1, 0])
        cr = rn.randint(1,255)
        cg = rn.randint(1,255)
        cb = rn.randint(1,255)
        cv.circle(self.trajectory, (x+self.x_origin, y+self.y_origin), 2, (cb, cg, cr), 1)
        #cv.rectangle(self.trajectory, (10, 20), (600, 60), (0, 0, 0), -1)
        cv.imshow("Trajectory "+str(self.sid), self.trajectory)
