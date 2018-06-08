import cv2
import numpy as np
import numpy.linalg as la
import cubicspline as cs
import matplotlib.pyplot as plt

height = 960
width = 1280

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

R = cs.Spline()
P = cs.Spline()

roll_imu =  [1.25, 
1.25,
1.3,
1.35,
1.26,
1.22,
1.58,
2.05,
2.06,
1.93,
1.44,
0.82,
0.25,
0.24,
-0.03,
-0.17,
-0.71,
-0.71,
-1.99,
-2.74,
-3.84,
-4.63,
-6.01,
-7.07,
-8.06,
-9.11,
-9.5,
-9.31,
-7.98,
-6.13,
-3.01,
0.34,
4.1,
7.24,
8.08,
8.67,
8.6,
6.35,
6.53,
7.3,
7.6,
7.2,
6.19,
5.05,
3.48,
1.26,
-0.28,
-1.38,
-3.75,
-7.76,
-6.84,
-4.77,
-2.53,
-3.27,
-4.28,
-4.1,
-3.14,
-2.81,
-3.04,
-3.69,
-4.65,
-5.81,
-6.22,
-5.45,
-3.62,
-1.69,
-1.65,
-1.66,
-1.12,
-0.02,
1.12,
2.37,
3.22,
3.4,
2.85,
3.09,
3.26,
-0.36,
-3.01,
-2.39,
-1.44,
-3.25,
-2.17,
-0.88,
-1.45,
-4.34,
-3.62,
-2.43,
-2.28,
-2.36,
-1.22,
0.71,
2.85,
4.74,
5.89,
4.56,
3.32,
1.77,
-0.45,
-3.01
]

pitch_imu = [0.33, 
0.33,
0.25,
0.14,
-0.06,
-0.41,
-0.91,
-0.97,
-0.64,
-0.23,
0.32,
0.85,
1.42,
2.41,
3.39,
4.37,
5.88,
7.19,
8.01,
8.96,
9.1,
8.54,
8.28,
8.47,
8.94,
8.74,
9.12,
9.1,
8.42,
7.49,
6.91,
5.73,
4.54,
3.87,
1.65,
2.93,
4.32,
4.84,
4.6,
3.97,
3.03,
3.14,
3.52,
3.82,
3.8,
4.51,
6.39,
6.68,
7.13,
8.54,
10.16,
10.84,
9.72,
7.09,
6.92,
8.77,
10.27,
11.67,
12.4,
12.61,
13.04,
13.83,
14.74,
13.83,
12.51,
11.65,
11.05,
11.21,
12.31,
12.54,
12.77,
12.47,
11.86,
13.27,
14.34,
13.9,
13.03,
11.19,
10.73,
9.66,
10.22,
8.53,
8.71,
8.59,
8.47,
8.47,
9.05,
9.53,
9.3,
9.2,
9.31,
9.68,
9.52,
9.45,
10.63,
11.58,
12.61,
13.25,
14.1,
14.73
]

# Sampling: anti-outlier
roll_pivots = [roll_imu[0]]
pitch_pivots = [pitch_imu[0]]

sampling_ratio = 20
frames = len(roll_imu)
pivots_count = int(frames / sampling_ratio)
x_timestamps = np.arange(0, pivots_count, 1 / sampling_ratio)

for i in range(1, pivots_count):
    roll_pivots.append((roll_imu[i * sampling_ratio - 1] + roll_imu[i * sampling_ratio] + roll_imu[i * sampling_ratio + 1]) / 3)
    pitch_pivots.append((pitch_imu[i * sampling_ratio - 1] + pitch_imu[i * sampling_ratio] + pitch_imu[i * sampling_ratio + 1]) / 3)

# Targets
roll_targets =  roll_imu[0:sampling_ratio]
pitch_targets = pitch_imu[0:sampling_ratio]

# Start
roll_df = 0
pitch_df = 0
for i in range(pivots_count - 3):

    roll_df = R.findSpline(roll_pivots[i], roll_pivots[i+1], roll_pivots[i+2], roll_pivots[i+3], roll_df, i)
    roll_targets.append(roll_pivots[i+1])
    for j in range(sampling_ratio - 1):
        roll_targets.append(R.mapSpline(x_timestamps[i * sampling_ratio + j + 1] - i))

    pitch_df = P.findSpline(pitch_pivots[i], pitch_pivots[i+1], pitch_pivots[i+2], pitch_pivots[i+3], pitch_df, i)
    pitch_targets.append(pitch_pivots[i + 1])
    for j in range(sampling_ratio - 1):
        pitch_targets.append(P.mapSpline(x_timestamps[i * sampling_ratio + j + 1] - i))

warp = np.zeros((height, width, 3), dtype=float)

for i in range(frames - 2*sampling_ratio):
    img = cv2.imread('./data/original/'+str(i)+'.jpg')
    roll_bias = roll_imu[i]
    pitch_bias = pitch_targets[i]-pitch_imu[i]
    print(str(i) + ' target: ' + str(roll_targets[i]) + ', imu: ' + str(roll_imu[i]) + ', b: ' + str(roll_bias))
    #H = findHomography((-roll_imu[i]), pitch_imu[i], 0., np.array([[0.], [0.], [0.]]))
    H = findHomography(roll_bias, pitch_bias, 0., np.array([[0.], [0.], [0.]])) #13.5-z
    warp = cv2.warpPerspective(img, H, (width, height))
    cv2.imwrite('./data/20/'+str(i)+'.jpg', warp)

