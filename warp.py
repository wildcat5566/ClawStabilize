import cv2
import time
import serial
import numpy as np
import numpy.linalg as la

height = 640
width = 800
distHeight = 480
distWidth = 640

K = np.matrix([[507.47,     0., 330.20],
               [    0., 507.24, 211.71],
               [    0.,     0.,     1.]])

def cos(angle):
    return np.cos(np.deg2rad(angle))

def sin(angle):
    return np.sin(np.deg2rad(angle))

def findHomography(r, p, y):
    
    Rx = np.matrix([[cos(r), -sin(r), 0.],
                    [sin(r),  cos(r), 0.],
                    [    0.,      0., 1.]])
    
    Ry = np.matrix([[1.,     0.,      0.],
                    [0., cos(p), -sin(p)], 
                    [0., sin(p),  cos(p)]])


    Rz = np.matrix([[ cos(y), 0., sin(y)],
                    [     0., 1.,     0.],
                    [-sin(y), 0., cos(y)]])

    R = Rx*Ry*Rz
    H = K*R*la.inv(K)
    return(H)

warp = np.zeros((height, width, 3), dtype=float)
crop = np.zeros((distHeight, distWidth, 3), dtype=float)

cp0 = cv2.VideoCapture(0)
cp0.set(3,800)
cp0.set(4,640)

ser = serial.Serial('COM28', 115200)
for i in range(5):
        ser.write(b'1')
        txt = ser.readline()
        print(txt)
        
while 1:
        ret0, frame0 = cp0.read()
        #cv2.imshow('Test0', frame0)
        ser.write(b'1')
        txt = ser.readline().decode('utf-8')
        values = txt.split(',')
        print(values[0] + ('roll: \t') + values[5] + ('pitch: \t') + values[6])

        roll = float(values[5])
        pitch = float(values[6])

        ### warp first ###
        H = findHomography(-roll, pitch, 0.)
        warp = cv2.warpPerspective(frame0, H, (width, height))
        cv2.imshow('Test0', warp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cp0.release()
