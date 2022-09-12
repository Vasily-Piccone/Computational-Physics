from cmath import nan
import sympy as sym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

"""
Question 2
"""
# Currently only handles one value of x.
# QUESTIONS: What if f''' is zero? How should we approach that?
def ndiff(fun, x, full=False):
    eps = 10*(-7)
    # calculate dx
    f = float(fun.subs(fvar, x))  # Value of f(x)
    fppp = fun.diff().diff().diff() # Calculating f'''
    f3p = float(fppp.subs(fvar, x)) # Value of f'''(x)
    
    # if this does equal zero, the derivative would also automatically be equal to zero
    if fppp.subs(fvar, x) != 0:
        dx = np.cbrt(np.abs(f*eps/f3p))
        print(dx)

    # calculate the derivative
    f_plus = float(fun.subs(fvar, x+dx))
    f_minus = float(fun.subs(fvar,x-dx))
    df = f_plus - f_minus
    if dx != 0:
        fp = df/(2*dx)
    else:
        if df < 0:
            fp = float('-inf')
        elif df > 0:
            fp = float('inf')
        else: 
            fp = nan
            print("The derivative of f at "+str(x)+" is undefined!")
    print(fp)

    if full == False:
        return fp
    else:
        # add safety checks if 0 or +/- inf
        # calculate rough error
        deriv = fun.diff()
        fp_actual = deriv.subs(fvar, x)
        error = fp_actual - fp
        return fp, dx, error
        # return derivative, dx, rough error

"""
Question 3
"""
# Is the V value provided included in the provided data? If not, throw an error -> Add proper error handling
def lakeshore(V, data):
    # Assuming that the data here is in the same format as the original file 
    voltage, temp = dat[:,1], dat[:,0]
    f = interp1d(temp, voltage, kind='cubic') # interpolated function

    # is the input a list or a single value?
    if isinstance(V, float) or isinstance(V, int):
        error = 0 # Define the calculation for this 
        t = f(V) # Calculate temperature
        return t, error
    elif isinstance(V,(list,pd.core.series.Series,np.ndarray)):
        error = []
        t_list = []
        # for each number in the list, interpolate
        for val in V:
            t_list.append(f(val))
        return t_list, error
    else:
        print("Please ensure that V is a list, an np.array, or a pandas array")
    # return the estimated temperature and the estimated error


"""
Question 4
"""
# interpolates for a single point
def poly_interpol():
    
    pass

"""
Helper Functions
"""
# Calculates the y_i given two lists of data using a Lagrange expansion.
"""TODO: 
- Add a check that ensures the lists are the same length (DONE)
- Calculate the estimated uncertainty
"""
def lagrange_i(x_i, x_list, y_list):
    y_i = 0
    if len(x_list) == len(y_list):
        for j in range(len(x_list)):
            l = 1
            for m in range(len(x_list)):
                if m != j:
                    l *= (x_i - x_list[m])/(x_list[j]-x_list[m])
            y_i = y_i + y_list[j]*l
    else: 
        print("The lists are of different sizes. Ensure your data is correct and try again.")
    return y_i


if __name__ == "__main__":

    # Question 2 Answer
    fvar = sym.Symbol("u")

    # func = sym.exp(fvar)
    # x = 0
    # q2 = ndiff(func, x)
    # print(q2)


    # Question 3 Answer 
    dat = np.loadtxt('./lakeshore.txt') # Might be good practice to add a try and catch statement here
    # Temperature | Voltage | dV/dT // Plot of the data
    print(dat)
    voltage, temp = dat[:,1], dat[:,0]

    # Debugging
    print("voltage:", voltage, "\n")
    print("temperature:", temp, "\n")
    v = 1.3
    # The order of these inputs is correct

    f = interp1d(temp, voltage, kind='cubic')
    print(f(1.7))
    voltage_num = 2.0
    vals, error = lakeshore(voltage_num, dat)
    print(vals, error)
    # plt.plot(voltage, temp, 'o', f(temp), voltage, '-')
    # plt.show()



    # Question 4 Answer
    no_pts = 100
    x = np.linspace(-np.pi/2, np.pi/2, no_pts)
    y = np.cos(x)


