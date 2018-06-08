import cv2
import time
import serial
import numpy as np
import numpy.linalg as la
import cubicspline as cs

height = 960
width = 1280
distHeight = 640
distWidth = 800

K = np.matrix([[756.78768558,   0.        , 629.89805344],
               [  0.        , 756.86336981, 345.49169401],
               [  0.        ,   0.        ,   1.        ]])

def cos(angle):
    return np.cos(np.deg2rad(angle))

def sin(angle):
    return np.sin(np.deg2rad(angle))

def findHomography(r, p, y, t):
    
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

    n = [0., 0., 1.]
    d = 50
    
    H = K*(R + t*np.transpose(n)/d)*la.inv(K)
    return(H)

def findZ(angle):
    R = 13.5
    if angle <= 90:
        return R
    elif angle > 90 and angle <= 135:
        return R*cos(angle - 90)
    else:
        return R*cos(180 - angle)

warp = np.zeros((height, width, 3), dtype=float)

cp0 = cv2.VideoCapture(0)
cp0.set(3,width)
cp0.set(4,height)

ser = serial.Serial('COM28', 115200)
for i in range(5):
        ser.write(b'1')
        txt = ser.readline()
        print(txt)
counts=0
ver = 0        
while 1:
        ret0, frame0 = cp0.read()
        ser.write(b'1')
        txt = ser.readline().decode('utf-8')
        values = txt.split(',')

        ### warp first ###
        # Cover roll and pitch angles based on IMU data.
        theta = float(values[1])%180
        roll = float(values[5])
        pitch = float(values[6])
        z = findZ(theta)

        # Spline smoothing
        R = cs.Spline()

        print(str(counts) + ('roll:') + str(roll) + (', pitch:') + str(pitch)+',')
        H = findHomography(-roll, pitch, 0., np.array([[0.], [0.], [0.]])) #13.5-z
        warp = cv2.warpPerspective(frame0, H, (width, height))

        cv2.imshow('New', frame0)
        fname = './data/original/' + str(counts)+'.jpg'
        counts=counts+1
        #cv2.imwrite(fname, frame0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cp0.release()
            

