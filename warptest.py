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

x_timestamps = np.arange(0, 40, 0.2)
roll_predict =  [1.81,
1.81,
1.75,
1.68,
1.57,
1.63,
2.97,
3.44,
3.28,
0.25,
-2.41,
-3.92,
-3.51,
-3.39,
-1.94,
-1.08,
-0.44,
1.71,
1.62,
1.65,
0.87,
0.82,
0.09,
-2.48,
-2.41,
-1.89,
0.87,
2.6,
3.18,
5.31,
5.5,
5.47,
3.68,
1.92,
-1.28,
-2.96,
-5.53,
-8.38,
-10.89,
-11.46,
-12.57,
-12.29,
-11.22,
-8.36,
-6.34,
-5.09,
-2.99,
-2.05,
-1.09,
-1.18,
-1.02,
-1.05,
-1.62,
-2.66,
-4.63,
-6.28,
-8.44,
-11.54,
-13.73,
-16.87,
-17.65,
-17.66,
-16.91,
-17.62,
-17.44,
-17.38,
-16.59,
-17.16,
-18.16,
-17.24,
-18.91,
-17.77,
-19.7,
-18.61,
-17.16,
-20.89,
-21.31,
-18.98,
-17.16,
-11.75,
-11.28,
-9.48,
-8.68,
-6.29,
-4.78,
-1.1,
1.46,
5.96,
8.67,
12.1,
14.59,
17.56,
19.32,
20.99,
25.69,
29.05,
29.1,
26.83,
27.18,
25.57,
21.58,
16.1,
12.93,
7.76,
1.7,
0.99,
1.28,
-0.25,
-1.51,
-1.41,
1.6,
1.35,
0.26,
1.06,
4.78,
4.24,
2.3,
2.44,
4.34,
3.18,
1.6,
3.21,
2.4,
2.75,
2.71,
2.99,
2.55,
4.75,
3.46,
3.76,
4.94,
4.8,
5.8,
5.52,
4.16,
5.84,
6.86,
5.18,
4.1,
4.5,
4.05,
3.26,
5.33,
4.3,
2.72,
3.06,
-0.15,
-2.7,
-5.54,
-9.36,
-11.79,
-14.74,
-18.57,
-20.15,
-24.55,
-22.7,
-20.42,
-16.21,
-11.76,
-8.31,
-7.68,
-5.59,
-3.66,
1.32,
5.24,
5.27,
10.64,
12.21,
13.2,
15.68,
14.99,
16.83,
15.56,
13.51,
16.53,
13.57,
11.07,
4.3,
2.27,
-1.99,
-3.91,
-6.56,
-8.97,
-10.74,
-6.53,
-6.12,
-5.21,
-10.58,
-5.36,
-3.93,
-11.09,
-17.3,
-20.97,
-21.35,
-22.4,
-20.99,
-20.13,
-18.42,
-15.52,
-8.4
]

pitch_predict = [ -0.45,
-0.45,
-0.42,
-0.44,
-0.5,
-0.82,
-2.02,
-1.87,
-1.7,
-0.29,
0.78,
2.19,
2.98,
3.0,
1.87,
1.1,
0.27,
-0.98,
-0.39,
0.2,
-0.23,
-0.9,
-0.8,
-0.43,
-2.44,
-3.43,
-4.62,
-5.98,
-4.47,
-3.61,
-2.65,
-0.89,
1.74,
4.42,
7.59,
5.19,
7.04,
8.02,
8.77,
6.29,
5.01,
2.23,
-0.76,
-3.64,
-4.65,
-5.71,
-5.0,
-4.93,
-4.02,
-2.53,
-2.72,
-2.11,
-0.6,
1.36,
3.26,
4.65,
6.22,
6.72,
7.11,
6.63,
3.2,
0.61,
-0.81,
-0.85,
-1.07,
-0.2,
0.15,
3.6,
7.56,
4.71,
8.63,
6.14,
12.96,
8.65,
6.43,
9.74,
8.46,
8.66,
7.28,
-2.01,
-0.28,
-1.59,
-3.84,
-6.03,
-4.41,
-3.94,
-5.53,
-5.6,
-7.51,
-6.55,
-3.8,
-1.13,
-2.6,
-3.0,
-5.98,
-6.01,
-2.65,
1.99,
6.22,
10.35,
10.58,
9.86,
12.51,
11.47,
8.61,
8.6,
10.07,
6.05,
1.4,
-0.4,
0.75,
-1.5,
-3.96,
-5.45,
-1.92,
-4.15,
-5.82,
-4.06,
-3.36,
-5.13,
-4.74,
-3.25,
-5.49,
-4.23,
-3.47,
-6.74,
-8.37,
-9.61,
-13.14,
-14.68,
-19.66,
-21.32,
-23.36,
-28.54,
-29.4,
-26.34,
-22.27,
-20.13,
-13.23,
-7.94,
-8.05,
-5.76,
-4.43,
-2.01,
2.98,
5.34,
6.56,
9.02,
9.02,
9.96,
9.87,
8.94,
7.2,
11.32,
12.4,
9.28,
6.11,
5.74,
2.26,
-1.36,
-1.02,
-2.16,
-5.43,
-5.36,
-7.57,
-7.77,
-11.94,
-14.65,
-15.42,
-17.5,
-17.06,
-17.32,
-19.31,
-15.75,
-14.26,
-13.76,
-12.06,
-13.08,
-8.06,
-10.36,
-9.05,
-13.23,
-14.74,
-11.73,
-4.92,
0.25,
6.66,
3.81,
-0.03,
-2.2,
-0.27,
-1.6,
-2.89,
-0.85,
-3.91,
-1.54,
-2.26,
-5.08,
-4.51,
-6.61
]

# Sampling: anti-outlier
roll_pivots = [roll_predict[0]]
pitch_pivots = [pitch_predict[0]]

for i in range(1, 40):
    roll_pivots.append((roll_predict[i*5-1] + roll_predict[i*5] + roll_predict[i*5+1])/3)
    pitch_pivots.append((pitch_predict[i*5-1] + pitch_predict[i*5] + pitch_predict[i*5+1])/3)

# Targets
roll_targets = [1.84,1.885, 2.134, 2.292, 2.314]#,            2.056, 1.461, 0.613, -0.346, -1.274,
                #-2.027, -2.334, -2.315, -2.055, -1.639,     -1.153, -0.573, -0.002, 0.506, 0.896]
pitch_targets = [-0.432, -0.524, -0.725, -0.956, -1.079]#,    -1.113, -0.897, -0.511, -0.034, 0.460,
                 #0.893, 1.161, 1.289, 1.302, 1.225,         1.08, 0.905, 0.688, 0.419, 0.090]

# Start
roll_df = 0
pitch_df = 0
for i in range(37):

    #plt.subplot(2,3,i+1)
    #plt.title('Roll t = ' + str(i+1) + '-' + str(i+2))
    roll_df = R.findSpline(roll_pivots[i], roll_pivots[i+1], roll_pivots[i+2], roll_pivots[i+3], roll_df, i)
    roll_targets.append(roll_pivots[i+1])
    #roll_mapped = []
    for j in range(4):
        #roll_mapped.append(R.mapSpline(x_timestamps[i*5+j+1] - i))
        roll_targets.append(R.mapSpline(x_timestamps[i*5+j+1] - i))

    #plt.plot([i,i+1,i+2,i+3],
    #         [roll_pivots[i], roll_pivots[i+1], roll_pivots[i+2], roll_pivots[i+3]], 'ro')
    #plt.plot(x_timestamps[(i*5):(i*5+15)], roll_predict[(i*5):(i*5+15)], 'x')
    #plt.plot(x_timestamps[(i*5+6):(i*5+10)], roll_mapped, 'co')

    ######
    #plt.subplot(2,3,i+4)
    #plt.title('Pitch t = ' + str(i+1) + '-' + str(i+2))
    pitch_df = P.findSpline(pitch_pivots[i], pitch_pivots[i+1], pitch_pivots[i+2], pitch_pivots[i+3], pitch_df, i)
    pitch_targets.append(pitch_pivots[i+1])
    #pitch_mapped = []
    for j in range(4):
        #pitch_mapped.append(P.mapSpline(x_timestamps[i*5+j+1] - i))
        pitch_targets.append(P.mapSpline(x_timestamps[i*5+j+1] - i))
    
    #plt.plot([i,i+1,i+2,i+3],
    #         [pitch_pivots[i], pitch_pivots[i+1], pitch_pivots[i+2], pitch_pivots[i+3]], 'ro')
    #plt.plot(x_timestamps[(i*5):(i*5+15)], pitch_predict[(i*5):(i*5+15)], 'x')
    #plt.plot(x_timestamps[(i*5+6):(i*5+10)], pitch_mapped, 'mo')
    
#plt.show()

warp = np.zeros((height, width, 3), dtype=float)

for i in range(190):
    img = cv2.imread('./data/original/'+str(i)+'.jpg')
    #H = findHomography((-roll_predict[i]), pitch_predict[i], 0., np.array([[0.], [0.], [0.]]))
    H = findHomography((roll_targets[i]-roll_predict[i]), -pitch_targets[i]+pitch_predict[i], 0., np.array([[0.], [0.], [0.]])) #13.5-z
    warp = cv2.warpPerspective(img, H, (width, height))
    cv2.imwrite('./data/_'+str(i)+'.jpg', warp)
