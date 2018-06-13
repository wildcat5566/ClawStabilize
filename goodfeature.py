import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
from fractions import Fraction  
edge = 20
dist_width = 640
dist_height = 480
width = 800
height = 600

col = ['b', 'g', 'r', 'c', 'm', 'y',
       'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
       'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']

def init(img1,features_count, features_res, features_dist):
    gray1 = np.float32(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))[edge:img1.shape[0]-edge, edge:img1.shape[1]-edge]
    f = cv2.goodFeaturesToTrack(gray1, features_count, features_res, features_dist)
    return f
    
def matchFeatures(img2, window, f1, features_count, features_res, features_dist, match_minimum_dist, plot=True, img1=None, count=0):
    t = time.time()
    gray2 = np.float32(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))[edge:img2.shape[0]-edge, edge:img2.shape[1]-edge]
    f2 = cv2.goodFeaturesToTrack(gray2, features_count, features_res, features_dist)

    #avoid boundaries
    for i in range(f1.shape[0]):
        f1[i][0][0] = f1[i][0][0] + edge
        f1[i][0][1] = f1[i][0][1] + edge

    for i in range(f2.shape[0]):
        f2[i][0][0] = f2[i][0][0] + edge
        f2[i][0][1] = f2[i][0][1] + edge
       
    matches = []
    for i in range(f1.shape[0]):
        for j in range(i, f2.shape[0]):
            x1 = f1[i][0][0]
            y1 = f1[i][0][1]
            x2 = f2[j][0][0]
            y2 = f2[j][0][1]
            dist2 = (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2)
            if(dist2 < match_minimum_dist):
                matches.append([x1, y1, x2, y2])

    true_matches = findDisplacement(matches)
    displacementField = findDisplacementField(true_matches)
    window[0] = window[0] + displacementField[0]
    window[1] = window[1] + displacementField[1]

    elapsed = time.time() - t
    #print(elapsed)

    if(plot==True):
        fig = plt.figure(figsize=(12,6), dpi=160)
        
        ax1 = plt.subplot(121)
        ax1.imshow(img1)
        for i in range(f1.shape[0]):
            ax1.plot(f1[i][0][0], f1[i][0][1], 'o', color=col[i%15], markerfacecolor='w', markersize=5)
        
        ax2 = plt.subplot(122)
        ax2.imshow(img2)
        for i in range(f2.shape[0]):
            ax2.plot(f2[i][0][0], f2[i][0][1], 'o', color=col[i%15], markerfacecolor='None', markersize=5)

        for i in range(len(matches)):
            ax2.plot(matches[i][0], matches[i][1], 'o', color='k', markerfacecolor='None', markersize=5)

        for i in range(len(true_matches)):
            ax2.arrow(true_matches[i][0], true_matches[i][1],  10*(true_matches[i][2]-true_matches[i][0]), 10*(true_matches[i][3]-true_matches[i][1]),
                      fc="b", ec="b", head_width=10, head_length=10)

        ax2.arrow(img2.shape[1]/2, img2.shape[0]/2, 10*displacementField[0], 10*displacementField[1], fc="r", ec="r", head_width=20, head_length=20, width=5)
        ax2.plot([(window[0]+0.5*800)-0.5*dist_width, (window[0]+0.5*800)+0.5*dist_width], [(window[1]+0.5*600)-0.5*dist_height, (window[1]+0.5*600)-0.5*dist_height], 'r-')
        ax2.plot([(window[0]+0.5*800)-0.5*dist_width, (window[0]+0.5*800)+0.5*dist_width], [(window[1]+0.5*600)+0.5*dist_height, (window[1]+0.5*600)+0.5*dist_height], 'r-')
        ax2.plot([(window[0]+0.5*800)-0.5*dist_width, (window[0]+0.5*800)-0.5*dist_width], [(window[1]+0.5*600)-0.5*dist_height, (window[1]+0.5*600)+0.5*dist_height], 'r-')
        ax2.plot([(window[0]+0.5*800)+0.5*dist_width, (window[0]+0.5*800)+0.5*dist_width], [(window[1]+0.5*600)-0.5*dist_height, (window[1]+0.5*600)+0.5*dist_height], 'r-')

        fig.savefig('./data/0530/marked/'+str(count)+'.jpg')
        plt.close()

    for i in range(f2.shape[0]):
        f2[i][0][0] = f2[i][0][0] - edge
        f2[i][0][1] = f2[i][0][1] - edge

    return window, f2

def findDisplacement(matches):
    dx = []
    dy = []
    for i in range(len(matches)):
        #arrow length
        dx.append(matches[i][2] - matches[i][0])
        dy.append(matches[i][3] - matches[i][1])
    
    xdisp_std = max(1,np.std(dx))
    ydisp_std = max(1,np.std(dy))
    xdisp_mean = np.mean(dx)
    ydisp_mean = np.mean(dy)
    
    #print('xstd: ' + str(xdisp_std) + '  , ystd: ' + str(ydisp_std))
    #print(dx)
    #print(dy)
    true_matches = []
    for i in range(len(matches)):
        if(abs(dx[i] - xdisp_mean) < 1.0*xdisp_std) and (abs(dy[i] - ydisp_mean) < 1.0*ydisp_std):
            true_matches.append(matches[i])

    return true_matches

def findDisplacementField(true_matches):
    #print(true_matches)

    dx = []
    dy = []
    for i in range(len(true_matches)):
    #arrow length
        x = int(true_matches[i][2] - true_matches[i][0])
        y = int(true_matches[i][3] - true_matches[i][1])
        dx.append(np.cbrt(x))
        dy.append(np.cbrt(y))

    print(dx)
    print(dy)

    dx = np.mean(dx)
    dy = np.mean(dy)
        
    field = []

    field.append(dx**3)
    field.append(dy**3)
    #print(field)
    return field
    
