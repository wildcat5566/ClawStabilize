import cv2
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import csv
import cubicspline as cs
import goodfeature as gf

K = np.matrix([[756.78768558,   0.        , 629.89805344],
               [  0.        , 756.86336981, 345.49169401],
               [  0.        ,   0.        ,   1.        ]])

def cos(angle):
    return np.cos(np.deg2rad(angle))

def sin(angle):
    return np.sin(np.deg2rad(angle))

def findZ(angle):
    R = 13.5
    if angle <= 90:
        return R
    elif angle > 90 and angle <= 135:
        return R*cos(angle - 90)
    else:
        return R*cos(180 - angle)

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

dataset = 'ramp2'
date_mmdd = '0616'
data_dir = '../data/' + date_mmdd + '/' + dataset + '/original/'
data_csv = '../data/' + date_mmdd + '/' + dataset + '/data.csv'
data_warpdir = '../data/' + date_mmdd + '/' + dataset + '/warp/'
data_matchdir = '../data/' + date_mmdd + '/' + dataset + '/match/'
data_cropdir = '../data/' + date_mmdd + '/' + dataset + '/crop/'

roll_imu = []
pitch_imu = []
dz = []
with open (data_csv) as target:
    reader = csv.DictReader(target)
    for row in reader:
        roll_imu.append(float(row['roll']))
        pitch_imu.append(float(row['pitch']))
        pos_angle = float(row['pos_angle'])
        dz.append(13.5 - findZ(pos_angle))
        #print(str(pos_angle) + "," + str(findZ(pos_angle)))

img1 = cv2.imread(data_dir+'0.jpg')
height = img1.shape[0]
width = img1.shape[1]

sampling_ratio = 10
R = cs.Spline()
P = cs.Spline()
roll_targets = R.findWarpTargets(sampling_ratio, roll_imu)
pitch_targets = P.findWarpTargets(sampling_ratio, pitch_imu)

#for i in range(len(dz)):
#    print(dz[i])

roll_bias = -roll_imu[0]
pitch_bias = -pitch_imu[0]#+pitch_targets[0]
H = findHomography(roll_bias, pitch_bias, 0., np.array([[0.], [dz[0]*1.5], [0.]])) #13.5-z
warp1 = cv2.warpPerspective(img1, H, (width, height))

features_count = 30
features_res = 0.01
features_dist = 20
std_range = 1.5
match_minimum_dist = 400
f = gf.init(warp1, features_count, features_res, features_dist)
window_center = [height*0.5, width*0.5]

#for i in range(1, 50):
for i in range(1, len(pitch_targets)):
    print(i)
    img2 = cv2.imread(data_dir + str(i) + '.jpg')
    roll_bias = -roll_imu[i]
    pitch_bias = -pitch_imu[i]#+pitch_targets[i]
    H = findHomography(roll_bias, pitch_bias, 0., np.array([[0.], [dz[i]*1.5], [0.]])) #13.5-z
    warp2 = cv2.warpPerspective(img2, H, (width, height))
    cv2.imwrite((data_warpdir + str(i) + '.jpg'), warp2)

    #window_center, f = gf.matchFeatures(warp2, window_center, f, features_count, features_res, features_dist, match_minimum_dist, std_range,
    #                                    plot=True, img1=warp1, count=i, savedir=data_matchdir)

    warp1 = warp2
