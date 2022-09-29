from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

"""
Functions for Question 1
"""
# Code provided to us in class
def rk4_step(fun, x, y, h):
    k1, k2, k3, k4 = h*fun(x,y), h*fun(x+h/2,y+k1/2), h*fun(x+h/2,y+k2/2), h*fun(x+h,y+k3)
    dy=(k1+2*k2+2*k3+k4)/6
    return y+dy

def rk4_stepd(fun, x, y, h):
    # Compares two steps of length h/2, and uses them to cancel leading-order error term from RK4
    k1, k2, k3, k4 = h*fun(x,y), h*fun(x+h/2,y+k1/2), h*fun(x+h/2,y+k2/2), h*fun(x+h,y+k3)
    dy1 =( k1+2*k2+2*k3+k4)/6
    y1 = y+dy1

    # RK4 for a h/2 step size
    k5, k6, k7, k8 = (k1/2), h*fun(x+h/4, y+k5/2)/2, h*fun(x+h/4, y+k6/2)/2, h*fun(x+h/2, y+k7)/2
    dy2 = (k5+2*k6+2*k7+k8)/6 
    temp_y = y + dy2

    k9, k10, k11, k12 = h*fun(x+h/2, temp_y)/2, h*fun(x+3*h/4, temp_y+k9/2)/2, h*fun(x+3*h/4, temp_y+k10/2)/2, h*fun(x+h, temp_y+k11)/2
    dy3 = (k9+2*k10+2*k11+k12)/6
    y2 = temp_y + dy3

    return y2 +(y2-y1)/15

"""
Function for Question 2
"""
# TODO: Figure out why this is taking forever to run
def decay_solver(coeffs):
    # Set up the matrix using the coefficients
    n = len(coeffs)
    diags = np.multiply(coeffs, -1)
    lower_diags = coeffs[:-1]
    mat = np.diag(diags) + np.diag(lower_diags, -1)
    
    # Solve the equations (IVPs)
    iv = np.zeros(n)
    iv[0] = 1
    print(iv)

    # Create logspace (because the decay times are so different)
    time = np.logspace(-6, 20, 101)
    print(time)

    # Solve
    F = lambda t, s: np.dot(mat, s)
    sol = solve_ivp(F, [10**(-6), 10**(20)], iv, method='Radau', t_eval=time)
    # make plots?
    plt.figure(figsize = (12, 8))
    plt.plot(time, sol.y.T[:,0]/sol.y.T[])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    return 0

"""
Functions for Question 3
"""

def photogrammetry():

    pass