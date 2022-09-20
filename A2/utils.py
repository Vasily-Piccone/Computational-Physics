import scipy
import numpy as np

"""
Functions for question 1
"""

# Z is a number, the function we want will be returned as a function of u
def gen_dE(R, z):
    def dE(theta):
        radius = R
        return (z-radius*np.cos(theta))*np.sin(theta)/(radius**2 + z**2 - 2*radius*z*np.cos(theta))
    return dE

def integrator():
    pass



"""
Functions for question 2 
"""

def integrate_adaptive(fun, a, b, tol, extra=None):
    pass




"""
Functions for Question 3
"""

# a) 
