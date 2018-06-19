import numpy as np
import matplotlib.pyplot as plt

col = ['k', 'g', 'm', 'r']
tol = 1.2

def ransac_init(fullset, plot=False):
    plt.plot(fullset[:,0], fullset[:,1], 'ko', markersize=5, markerfacecolor='None')
    plt.plot(fullset[:,2], fullset[:,3], 'ko', markersize=5)
    hypo_inliers = fullset[0:int(len(fullset) / 3)]

    ### Model fitting
    con_x = []
    con_y = []
    for [x0, y0, x1, y1] in hypo_inliers:
        con_x.append(x1-x0)
        con_y.append(y1-y0)
        
    model = [np.mean(con_x), np.mean(con_y), tol*np.std(con_x), tol*np.std(con_y)]

    if plot==True:
        print("#Model#")
        print(model)
        for [x0, y0, x1, y1] in fullset:
            plt.arrow(x0, y0, (x1 - x0), (y1 - y0), fc="k", ec="k", head_width=5, head_length=5, width=1,length_includes_head=True)
        for [x0, y0, x1, y1] in hypo_inliers:
            plt.arrow(x0, y0, (x1 - x0), (y1 - y0), fc="g", ec="g", head_width=5, head_length=5, width=1,length_includes_head=True)
        plt.arrow(np.mean(fullset[:,0]), np.mean(fullset[:,1]), np.mean(con_x), np.mean(con_y),
                  fc=col[1], ec=col[1], head_width=10, head_length=10, width=2,length_includes_head=True)
    return model

def ransac(fullset, model, iteration, plot=False):
    con_x = []
    con_y = []
    for [x0, y0, x1, y1] in fullset:
        if (x1-x0) <= model[0] + model[2] and (x1-x0) >= model[0] - model[2] and (y1-y0) <= model[1] + model[3] and (y1-y0) >= model[1] - model[3]:
            con_x.append(x1 - x0)
            con_y.append(y1 - y0)
            #print([x1 - x0, y1 - y0])
            if plot==True:
                plt.arrow(x0, y0, x1-x0, y1-y0, fc=col[iteration], ec=col[iteration], head_width=5, head_length=5, width=1,length_includes_head=True)
                
    model = [np.mean(con_x), np.mean(con_y), tol*np.std(con_x), tol*np.std(con_y)]
    
    if plot==True:
        print("#Model#")
        print(model)
        plt.arrow(np.mean(fullset[:,0]), np.mean(fullset[:,1]), model[0], model[1],
                  fc=col[iteration], ec=col[iteration], head_width=10, head_length=10, width=2,length_includes_head=True)
    return model

###Test code###
"""
matches = np.array([[547.0, 227.0, 547.0, 233.0], [211.0, 262.0, 234.0, 284.0], [184.0, 352.0, 185.0, 357.0], [429.0, 501.0, 444.0, 
508.0], [670.0, 144.0, 672.0, 144.0], [492.0, 476.0, 493.0, 483.0], [651.0, 382.0, 645.0, 391.0], [711.0, 274.0, 
713.0, 279.0], [404.0, 192.0, 405.0, 198.0], [311.0, 336.0, 308.0, 350.0], [572.0, 391.0, 557.0, 392.0], [328.0, 
100.0, 312.0, 97.0], [377.0, 138.0, 362.0, 139.0], [999.0, 684.0, 1002.0, 692.0], [287.0, 155.0, 283.0, 159.0], 
[474.0, 417.0, 465.0, 414.0], [1099.0, 170.0, 1097.0, 171.0], [350.0, 228.0, 355.0, 250.0]])

model = ransac_init(matches, plot=True)
for i in range(2,4):
    model = ransac(matches, model, i, plot=True) #hypo_outliers
plt.show()"""
