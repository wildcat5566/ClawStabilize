import cv2
import time
import serial
import numpy as np
import numpy.linalg as la

height = 960
width = 1280
distHeight = 480
distWidth = 640

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

warp = np.zeros((height, width, 3), dtype=float)
crop = np.zeros((distHeight, distWidth, 3), dtype=float)

cp0 = cv2.VideoCapture(0)
cp0.set(3,width)
cp0.set(4,height)

ser = serial.Serial('COM28', 115200)
for i in range(5):
        ser.write(b'1')
        txt = ser.readline()
        print(txt)

dz = 0        
while 1:
        ret0, frame0 = cp0.read()
        ser.write(b'1')
        txt = ser.readline().decode('utf-8')
        values = txt.split(',')
        print('[predict] ' + ('roll: \t') + values[5] + ('pitch: \t') + values[6])

        roll = float(values[5])
        pitch = float(values[6])

        ### warp first ###
        dz = dz - 0.1
        H = findHomography(-roll, pitch, 0., np.array([[0.], [dz], [0.]])) # -10
        warp = cv2.warpPerspective(frame0, H, (width, height))

        ### and then crop ###
        tb = int((height - distHeight)*0.5)
        bb = tb + distHeight
        lb = int((width - distWidth)*0.5)
        rb = lb + distWidth
        #print(tb)
    
        rect = np.array([[lb, tb],
                         [rb, tb],
                         [rb, bb],
                         [lb, bb]], dtype = "float32")

        ### Plot rectangle ###
        for j in range(4): 
            cv2.line(warp, (rect[j][0], rect[j][1]), (rect[(j+1)%4][0], rect[(j+1)%4][1]) , (255,0,0), 2)

        crop = warp[tb:bb, lb:rb, :]

        cv2.imshow('Test0', crop)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cp0.release()

