import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random 

def question1():
    # Open the data file and load the data
    path = './rand_points.txt'
    data = np.loadtxt(path)
    x, y, z = data[:,0], data[:,1], data[:,2]

    a, b = -4, 2
    x1, y1 = a*x+b*y, z
    plt.figure(1)
    plt.scatter(x1, y1, s=0.5)

    N = 50000  # The approximate number of points found in the text file
    x2, y2, z2 = np.asarray(random.sample(range(1, int(1e8)), N)), np.asarray(random.sample(range(1, int(1e8)), N)), np.asarray(random.sample(range(1, int(1e8)), N))
    print(type(x2))
    a2, b2 = 2, -2
    x_py, y_py = a2*x2+b2*y2, z2
    print(len(x_py), len(y_py))
    plt.figure(2)
    plt.scatter(x_py, y_py, s=0.5)
    plt.show()


def question2():
    y = np.linspace(0, np.pi/2, 1001) # Cutoff at zero
    y=0.5*(y[1:]+y[:-1])
    tany=2*np.tan(y)
    cosy=np.cos(y)

    # Plot the envelopes of the distributions
    p_lor=np.exp(-tany)/2/cosy**2
    p_exp = np.exp(-y)
    plt.figure(1)
    plt.plot(y, p_lor, label="Lorentzian transformation")
    plt.plot(y, p_exp, label="Exponential distribution")
    plt.legend(loc="upper left")

    n=1000000
    yy=np.random.uniform(low=0, high=np.pi/2, size=n)

    myp=np.exp(-np.tan(yy))/2/np.cos(yy)**2
    fac=1.01*p_lor.max()
    accept=(np.random.rand(n)*fac)<myp
    print('accept fraction is ', np.mean(accept))
    y_use=yy[accept]
    x_use=np.tan(y_use)
    aa, bb=np.histogram(x_use, np.linspace(0,np.pi/2,101))  # What determines this?
    b_cent=0.5*(bb[1:]+bb[:-1])
    pred=np.exp(-b_cent)
    pred=pred/pred.sum()
    aa=aa/aa.sum()
    plt.figure(2)
    plt.clf()
    plt.plot(b_cent,pred, color="red", label="Exponential distribution")
    plt.bar(b_cent,aa,0.15, label="bar plot")
    plt.show()


def question3():
    N = 1000000
    u, v = np.random.rand(N), np.random.rand(N)

    # Upper bound on u
    upper_bound_u = np.sqrt(np.exp(-v/u))

    # Keeping only the points that are between 0 and the upper bound condition
    useful_points = u < upper_bound_u

    # Update u and v based off the condition
    u, v = u[useful_points], v[useful_points]
    print("Fraction of pts kept: ", len(u)/len(upper_bound_u))
    plt.hist(v/u/N, bins=100, label='Histogram')
    plt.legend()
    plt.plot()
    plt.show()