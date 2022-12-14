from cmath import nan
import sympy as sym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import UnivariateSpline, interp1d


"""
Question 1
"""
# Returns the optimal error for the functions men
def opt_errors():
    eps = 10**(-15)
    # Calculate optimal errors using the calculated optimal dx
    g_r = np.random.randint(1, 9)
    print(g_r)
    dx1, dx2 = np.cbrt(3*eps*g_r), np.cbrt(3*g_r*eps/(0.01)**3) # Optimal dx's 
    opt_err1, opt_err2 = (g_r*eps)/dx1 +(dx1)**2/6, (g_r*eps)/dx2 +((0.01)**3)*(dx2)**2/(6)
    return opt_err1, dx1, opt_err2, dx2

def plot_errors(num_pts):
    min_dx_order = -8
    max_dx_order= -3
    dx = np.logspace(min_dx_order, max_dx_order, num=num_pts, endpoint=True) # 100 evenly-spaced points on a log scale
    print(dx)
    eps = 10**(-15)
    g = np.random.randint(1, 9, num_pts) # Random order unity integers
    err1 = (g*eps)/dx + (dx)**2/6
    err2 = (g*eps)/dx +(dx)**2/(6*(0.01)**3)
    return err1, err2, dx

"""
Question 2
"""
# Currently only handles one value of x.
# QUESTIONS: What if f''' is zero? How should we approach that?
def ndiff(fun, x, full=False):
    eps = 10*(-15)
    g = np.random.randint(1, 9)
    # calculate dx
    dx = np.cbrt(3*g*eps)
    
    # calculate the derivative
    f_plus = float(fun(x+dx))
    f_minus = float(fun(x-dx))
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

    # Assuming dx is small, only the 1/dx term will dominate
    if full == False:
        return fp
    else:
        error = g*eps/dx
        return fp, dx, error
        # return derivative, dx, rough error

"""
Question 3
"""
# Is the V value provided included in the provided data? If not, throw an error -> Add proper error handling
def lakeshore(V, data):
    # Assuming that the data here is in the same format as the original file 
    voltage, temp, dV = data[:,1], data[:,0], data[:,2]
    f = UnivariateSpline(temp, voltage) # interpolated function
    fp_real = f.derivative()
    fp_inter = UnivariateSpline(temp, dV)

    # is the input a list or a single value?
    if isinstance(V, float) or isinstance(V, int):
        error = fp_real(V) - fp_inter(V)
        t = f(V) # Calculate temperature
        return t, error
    elif isinstance(V,(list,pd.core.series.Series,np.ndarray)):
        error = []
        t_list = []
        # for each number in the list, interpolate
        for val in V:
            t_list.append(f(val))
            error.append(fp_real(val) - fp_inter(val))
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
- Lorentzian (between -1 and 1) (done)
    - Fixing the constant term in the denominator
    - Order increases, what happens?
    - Can you understand what has happened 
"""
# interpolates for a single point
# Please note the code for the interpolation methods is the same code from the lectures
def inter_compare(start, end, n: int, m: int, num_pts: int, func): # Make sure the function passed in is in a valid format 
    x = np.linspace(start, end, n+m-1)
    # Rational interpolation
    y = func(x)
    p, q = rat_fit(x,y,n,m)

    xx=np.linspace(start, end, num_pts)
    yy = func(xx)
    yy_interp=rat_eval(p, q, xx) # Part of the code we want to plot
    error_rat = yy_interp-yy

    # Polynomial interpolation
    pp = np.polyfit(x, y, n+m)
    yy_poly = np.polyval(pp, xx) # Want to plot
    error_poly = yy_poly-yy

    # Cubic Spline
    spln = interpolate.splrep(x, y)
    yy_spline = interpolate.splev(xx, spln)
    error_spline = yy_spline-yy
    # Plot all the errors together
    plt.clf();
    plt.plot(xx, error_spline)
    plt.plot(xx, error_poly)
    plt.plot(xx, error_rat)
    plt.legend(['spline error', 'polynomial interpol error', 'rational interpol error'])
    plt.show()

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
    print(mat)
    pars=np.dot(np.linalg.pinv(mat),y)
    p=pars[:n]
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

def lorentzian(x):
    return 1/(1+x**2)

if __name__ == "__main__":

    # Question 1 Answer 

    # err1, err2, dx = plot_errors(1000)
    # opt_err1, dx1, opt_err2, dx2 = opt_errors()
    # print(opt_err1, dx1, opt_err2, dx2)
    # Add the optimum dx to the plot
    # Fix the issue with the dx2

    # print(err1)
    # print("\n")
    # print(err2)
    # plt.loglog(dx, err2)
    # plt.title("Error for e^(0.01x) vs. dx")
    # plt.xlabel("dx")
    # plt.ylabel("Error for e^(0.01x)")
    # plt.plot(dx2, opt_err2, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
    # plt.legend(['Error','Optimal dx'])
    # plt.show()


    # Question 2 Answer (Fix question 2)
    # fvar = sym.Symbol("u")

    # func = sym.exp(fvar)
    # x = 0
    # q2 = ndiff(func, x)
    # print(q2)


    # # Question 3 Answer 
    # dat = np.loadtxt('./lakeshore.txt') # Might be good practice to add a try and catch statement here
    # # Temperature | Voltage | dV/dT // Plot of the data
    # voltage, temp = dat[:,1], dat[:,0]

    # Debugging
    # print("voltage:", voltage, "\n")
    # print("temperature:", temp, "\n")
    # voltage_num = 0.5
    # vals, error = lakeshore(voltage_num, dat)
    # print(vals, error)
    # plt.plot(voltage, temp)
    # plt.plot(voltage_num, vals, marker='*', ls='none', ms=20)
    # plt.show()

    # # Question 4 Answer
    no_pts = 101
    x = np.linspace(-np.pi/2, np.pi/2, no_pts)
    y = np.cos

    # Part a)
    # Prints out the plots
    inter_compare(-np.pi/2, np.pi/2, 4, 5, 101, y)

    # Part b)
    xb = np.linspace(-1, 1, no_pts)
    yb = lorentzian
    inter_compare(-1, 1, 4, 5, 101, yb)

    # When we switch from np.linalg.inv to np.lingalg.pinv we get a result that properly 