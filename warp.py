import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

height = 600
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
    #print(la.inv(K))
    return(H)

#imgseq = np.zeros((70, height, width, 3), dtype=float)
#warp = np.zeros((70, height, width, 3), dtype=float)
crop = np.zeros((70, distHeight, distWidth, 3), dtype=float)

roll = [1.57, 1.40, 1.35, 1.36, 1.07,       0.72, 0.68, 0.59, 0.55, 0.62,
        0.74, 0.79, 0.62, 0.43, 0.29,       0.31, 0.28, 0.30, 0.39, 0.40,
        0.36, 0.30, 0.33, 0.29, 0.23,       0.18, 0.36, 0.55, 0.78, 1.05,
        1.71, 2.13, 1.37, 1.41, 1.38,       0.50, -0.09 , -0.23, -0.85, -1.12,
        -0.79, -0.18, 0.22, 0.49, 1.01,     1.18, 1.28, 1.38, 1.10, 1.05,
        1.18, 1.37, 1.61, 1.73, 1.65,       1.28, 1.02, 0.89, 0.96, 0.89,
        0.87, 0.85, 0.90, 0.87, 0.79,       0.75, 0.63, 0.53, 0.44, 0.39]

pitch = [-1.44, -1.38, -1.31, -1.17, -1.21, -1.29, -1.18, -1.04, -0.85, -0.69,
         -0.44, -0.22, -0.20, -0.44, -0.49, -0.37, -0.28, -0.20, -0.14, -0.16,
         -0.22, -0.28, -0.25, -0.20, -0.21, -0.21, 0.03, 0.23, 0.44, 0.86,
         1.15, 0.67, 0.65, -0.09, -0.73,    -2.47, -3.36, -3.10, -3.00, -0.53,
         0.82, -0.42, -0.18, -0.3, -0.63,   -0.67, -0.59, -0.68, -0.79, -0.84,
         -0.82, -0.65, -0.41, -0.15, 0.16,  -0.04, -0.54, -0.54, -0.36, -0.21,
         -0.24, -0.32, -0.34, -0.41, -0.44, -0.46, -0.44, -0.43, -0.47, -0.40]

z = [13.5, 13.5, 13.5, 13.5, 13.5,      13.5, 13.5, 13.5, 13.5, 13.5,
     13.5, 13.5, 13.5, 13.5, 13.5,      13.5, 13.5, 13.5, 13.5, 13.5,
     13.5, 13.5, 13.5, 13.5, 13.5,      13.5, 13.5, 13.5, 13.44, 13.29,
     13.02, 12.37, 12.23, 11.72, 11.16, 10.39, 9.50, 10.28, 10.90, 11.55,
     12.17, 12.71, 13.09, 13.33, 13.46, 13.5, 13.5, 13.5, 13.5, 13.5,
     13.5, 13.5, 13.5, 13.5, 13.5,      13.5, 13.5, 13.5, 13.5, 13.5,
     13.5, 13.5, 13.5, 13.5, 13.5,      13.5, 13.5, 13.5, 13.5, 13.5]

for i in range(70):
    filename = './0530/orig/00' + str(i+29) + '.jpg'
    imgseq = cv2.imread(filename)

    ### warp first ###
    H = findHomography(roll[i], -pitch[i], 0.)
    warp = cv2.warpPerspective(imgseq, H, (width, height))
    
    

    ### and then crop ###

    tb = int((height - distHeight)*0.5) - int((13.5 - z[i])*7.5)
    bb = tb + distHeight
    lb = int((width - distWidth)*0.5)
    rb = lb + distWidth
    print(tb)
    
    rect = np.array([[lb, tb],
                     [rb, tb],
                     [rb, bb],
                     [lb, bb]], dtype = "float32")

    # Plot rectangle
    for j in range(4): 
        cv2.line(warp, (rect[j][0], rect[j][1]), (rect[(j+1)%4][0], rect[(j+1)%4][1]) , (255,0,0), 2)
                 
    crop[i] = warp[tb:bb, lb:rb, :]
    
    

### write files ###
for i in range(70):
    newfilename = './0530/new/_00' + str(i+29) + '.jpg'
    c = np.array(crop[i], dtype=float)
    cv2.imwrite(newfilename, c)

