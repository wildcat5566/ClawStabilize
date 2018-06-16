import cv2
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import cubicspline as cs
import goodfeature as gf

height = 600
width = 800

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

roll_imu =  [1.13,
1.16,
1.19,
1.22,
1.25,
1.28,
1.31,
1.34,
1.37,
1.4,
1.43,
1.45,
1.48,
1.51,
1.53,
1.56,
1.58,
1.6,
1.62,
1.65,
1.67,
1.69,
1.71,
1.74,
1.76,
1.78,
1.8,
1.82,
1.85,
1.86,
1.88,
1.89,
1.91,
1.93,
1.95,
1.96,
1.97,
1.98,
1.98,
1.98,
1.99,
1.99,
1.99,
1.98,
1.99,
1.99,
1.99,
1.98,
1.98,
1.99,
1.98,
1.99,
1.99,
1.99,
1.97,
1.94,
1.92,
1.9,
1.88,
1.86,
1.84,
1.81,
1.78,
1.75,
1.71,
1.66,
1.63,
1.59,
1.56,
1.53,
1.52,
1.51,
1.52,
1.57,
1.64,
1.7,
1.75,
1.78,
1.81,
1.84,
1.83,
1.78,
1.71,
1.65,
1.59,
1.58,
1.57,
1.58,
1.59,
1.59,
1.61,
1.64,
1.63,
1.62,
1.6,
1.59,
1.58,
1.57,
1.57,
1.55,
1.54,
1.58,
1.57,
1.56,
1.54,
1.53,
1.56,
1.62,
1.67,
1.69,
1.74,
1.74,
1.74,
1.75,
1.75,
1.77,
1.74,
1.75,
1.76,
1.78,
1.79,
1.8,
]

pitch_imu = [0.74,
0.72,
0.7,
0.68,
0.66,
0.64,
0.62,
0.6,
0.58,
0.55,
0.53,
0.52,
0.49,
0.47,
0.44,
0.42,
0.4,
0.38,
0.36,
0.33,
0.31,
0.29,
0.28,
0.26,
0.25,
0.23,
0.22,
0.2,
0.19,
0.18,
0.16,
0.15,
0.14,
0.14,
0.13,
0.12,
0.12,
0.11,
0.1,
0.09,
0.08,
0.07,
0.06,
0.05,
0.04,
0.02,
0.01,
0,
0,
-0.02,
-0.02,
-0.03,
-0.04,
-0.04,
-0.04,
-0.03,
-0.01,
0,
0.01,
0.02,
0.03,
0.05,
0.06,
0.07,
0.09,
0.1,
0.11,
0.11,
0.11,
0.11,
0.11,
0.12,
0.18,
0.3,
0.44,
0.57,
0.72,
0.86,
0.99,
1.08,
1.1,
1.08,
1.06,
1.06,
1.1,
1.13,
1.15,
1.19,
1.2,
1.18,
1.19,
1.24,
1.26,
1.21,
1.17,
1.13,
1.11,
1.1,
1.09,
1.08,
1.08,
1.08,
1.05,
1.02,
0.93,
0.8,
0.65,
0.55,
0.42,
0.31,
0.24,
0.15,
0.07,
-0.02,
-0.07,
-0.11,
-0.18,
-0.22,
-0.25,
-0.32,
-0.39,
-0.45
]

sampling_ratio = 5
R = cs.Spline()
P = cs.Spline()
roll_targets = R.findWarpTargets(sampling_ratio, roll_imu)
pitch_targets = P.findWarpTargets(sampling_ratio, pitch_imu)

print('targets fetched')
img1 = cv2.imread('./data/0530/original/0.jpg')
roll_bias = roll_imu[0]
pitch_bias = pitch_targets[0]-pitch_imu[0]
H = findHomography(roll_bias, pitch_bias, 0., np.array([[0.], [0.], [0.]])) #13.5-z
warp1 = cv2.warpPerspective(img1, H, (width, height))

features_count = 30
features_res = 0.01
features_dist = 30
match_minimum_dist = 1000
f = gf.init(warp1, features_count, features_res, features_dist)
window = [img1.shape[0]*0.5, img1.shape[1]*0.5]
for i in range(1, 2):
    img2 = cv2.imread('./data/0530/original/'+str(i)+'.jpg')
    roll_bias = roll_imu[i]
    pitch_bias = pitch_targets[i]-pitch_imu[i]
    H2 = findHomography(roll_bias, pitch_bias, 0., np.array([[0.], [0.], [0.]])) #13.5-z
    warp2 = cv2.warpPerspective(img2, H, (width, height))

    window, f = gf.matchFeatures(warp2, window, f, features_count, features_res, features_dist, match_minimum_dist, plot=True, img1=warp1, count=i)

    warp1 = warp2
