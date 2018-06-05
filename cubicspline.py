import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def cubicSpline(y0, y1, y2, y3, df1, iterations):
    t = 1
    A = np.zeros((12, 12),dtype=float)
    b = np.zeros((12, 1),dtype=float)
    
    ### Equations 0 to 5: Sampled measurements ###
    #eqn 0: f0(x0) = y0 = a00
    b[0] = y0
    A[0][0] = 1

    #eqn 1: f0(x1) = y1 = a00+a01t+a02t^2+a03t^3
    b[1] = y1
    A[1][0] = 1
    A[1][1] = t
    A[1][2] = t*t
    A[1][3] = t*t*t

    #eqn 2: f1(x1) = y1 = a10
    b[2] = y1
    A[2][4] = 1

    #eqn 3: f2(x2) = y2 = a10+a11t+a12t^2+a13t^3
    b[3] = y2
    A[3][4] = 1
    A[3][5] = t
    A[3][6] = t*t
    A[3][7] = t*t*t

    #eqn 4: f2(x2) = y2 = a20
    b[4] = y2
    A[4][8] = 1

    #eqn 5: f2(x3) = y3 = a20+a21t+a22t^2+a23t^3
    b[5] = y3
    A[5][8] = 1
    A[5][9] = t
    A[5][10] = t*t
    A[5][11] = t*t*t
    
    ### Equations 6 and 7: Known starting boundary conditions ###
    #eqn 6: f0'(x0) = a01 = given
    b[6] = df1
    A[6][1] = 1

    #eqn 7: reduce acceleration at ending with a reduction ratio alpha.
    #       f"23(t3) = 2a22+6a23t = alpha*f"23(t2) = alpha*2a22
    #       2(1-alpha)a22+6a23t=0
    b[7] = 0
    alpha = 1.0
    A[7][10] = 1 - alpha
    A[7][11] = 6*t
    #A[7][9]=0.5
    #A[7][10]=2*t
    #A[7][11]=3*t*t


    ### Equation 8 and 9: 1st order derivatives continuity ###
    #eqn 8: f0'(x1) = f1'(x1), a01+2a02t+3a03t^2 = a11
    b[8] = 0
    A[8][1] = 1
    A[8][2] = 2*t
    A[8][3] = 3*t*t
    A[8][5] = -1

    #eqn 9: f1'(x2) = f2'(x2), a11+2a12t+3a13t^2 = a21
    b[9] = 0
    A[9][5] = 1
    A[9][6] = 2*t
    A[9][7] = 3*t*t
    A[9][9] = -1

    ### Equation 10 and 11: 2nd order derivatives continuity ###
    #eqn 10: f0"(x1) = f1"(x1), 2a02+6a03t=2a12
    b[10] = 0
    A[10][2] = 2
    A[10][3] = 6*t
    A[10][6] = -2

    #eqn 11: f1"(x2) = f2"(x2), 2a12+6a13t=2a22
    b[11] = 0
    A[11][6] = 2
    A[11][7] = 6*t
    A[11][10] = -2

    x = la.solve(A,b)
    #print(x)

    xrange0 = np.arange(iterations, iterations+t, 0.01)
    yrange0 = []
    for i in range(len(xrange0)):
        n = xrange0[i]-xrange0[0]
        yrange0.append(x[0] + n*x[1] + n*n*x[2] + n*n*n*x[3])
    plt.plot(xrange0, yrange0, 'k:')

    xrange1 = np.arange(iterations+t, iterations+2*t, 0.01)
    yrange1 = []
    for i in range(len(xrange1)):
        n = xrange1[i]-xrange1[0]
        yrange1.append(x[4] + n*x[5] + n*n*x[6] + n*n*n*x[7])
    plt.plot(xrange1, yrange1, 'm')

    xrange2 = np.arange(iterations+2*t, iterations+3*t, 0.01)
    yrange2 = []
    for i in range(len(xrange2)):
        n = xrange2[i]-xrange2[0]
        yrange2.append(x[8] + n*x[9] + n*n*x[10] + n*n*n*x[11])
    plt.plot(xrange2, yrange2, 'k:')

    # Return 1st order derivative at x1 junction point
    # As new starting boundary conditions for next iteration
    # To ensure continuous velocity
    # 1st order derivative: f1'(x1) = a11
    return x[5]


df = 0
x_meas = np.arange(0, 6, 0.2)
y_meas = [6.12, 5.8 , 5.44, 5.13, 4.25,
          3.89, 3.26, 1.93, 0.27, -0.61,
          -1.43, -1.94,-1.7,-0.64,1.01,
          2.74, 4.33, 5.58, 6.89, 7.89,
          3.89, 8.92, 9.22, 9.7 , 10.11,
          10.23,9.58, 9.47, 8.39, 5.71]

# Sampling: anti-outlier
y_pivots = [y_meas[0]]
for i in range(1, 6):
    y_pivots.append((y_meas[i*5-1] + y_meas[i*5] + y_meas[i*5+1])/3)

plt.subplot(1,3,1)
plt.plot(x_meas[0:15], y_meas[0:15], 'x')
plt.plot([0,1,2,3], [y_pivots[0], y_pivots[1], y_pivots[2], y_pivots[3]], 'ro')
df = cubicSpline(y_pivots[0], y_pivots[1], y_pivots[2], y_pivots[3], df, 0)

plt.subplot(1,3,2)
plt.plot(x_meas[5:20], y_meas[5:20], 'x')
plt.plot([1,2,3,4], [y_pivots[1], y_pivots[2], y_pivots[3], y_pivots[4]], 'ro')
df = cubicSpline(y_pivots[1], y_pivots[2], y_pivots[3], y_pivots[4], df, 1)

plt.subplot(1,3,3)
plt.plot(x_meas[10:25], y_meas[10:25], 'x')
plt.plot([2,3,4,5], [y_pivots[2], y_pivots[3], y_pivots[4], y_pivots[5]], 'ro')
df = cubicSpline(y_pivots[2], y_pivots[3], y_pivots[4], y_pivots[5], df, 2)

plt.show()

