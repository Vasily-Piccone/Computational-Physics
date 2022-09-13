from cmath import nan
import sympy as sym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

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
    f = interp1d(voltage, temp, kind='cubic') # interpolated function

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

TODO:
- Test all written interpolation methods
- Poly interpol (done)
- cubic interpol (done)
- rational function interpol (done)
- Lorentzian (between -1 and 1)
"""
# interpolates for a single point
def inter_compare(start, end, n: int, m: int, num_pts: int, func): # Make sure the function passed in is in a valid format 
    x = np.linspace(start, end, n+m-1)
    # Rational interpolation
    p, q = rat_fit(x,func,n,m)
    pred=rat_eval(p, q, x)

    xx=np.linspace(start, end, num_pts)
    yy_interp=rat_eval(p, q, xx) # Part of the code we want to plot
    error_rat = yy_interp-func

    # Polynomial interpolation
    pp = np.polyfit(x, func, n+m)
    yy_poly = np.polyval(pp, xx) # Want to plot
    error_poly = yy_poly-func

    # Cubic Spline
    spln = interpolate.splrep(x, func)
    yy_spline = interpolate.splev(xx, spln)
    error_spline = yy_spline-func

    # Plot all the errors together

"""
Helper Functions
"""
def rat_fit(x, y, n, m):
    assert(len(x)==n+m-1)
    assert(len(x)==len(y))
    mat=np.zeros([n+m-1, n+m-1])
    for i in range(n):
        mat[:,i]=x**i
    for i in range(1, m):
        mat[:,i-1+n]=-y*x**i
    pars=np.dot(np.linalg.inv(mat),y)
    p = pars[:n]
    q=pars[n:]
    return p,q

def rat_eval(p, q, x):
    top=0
    for i in range(len(p)):
        top=top+p[i]*x**i
    bot=1
    for i in range(len(q)):
        bot=bot+q[i]*x**(i+1)
    return top/bot


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
    voltage, temp = dat[:,1], dat[:,0]

    # Debugging
    print("voltage:", voltage, "\n")
    print("temperature:", temp, "\n")
    voltage_num = 0.5
    vals, error = lakeshore(voltage_num, dat)
    print(vals, error)
    plt.plot(voltage, temp)
    plt.plot(voltage_num, vals, marker='*', ls='none', ms=20)
    plt.show()


    # Question 4 Answer
    no_pts = 100
    x = np.linspace(-np.pi/2, np.pi/2, no_pts)
    y = np.cos(x)