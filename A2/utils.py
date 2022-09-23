from distutils.util import copydir_run_2to3
from tokenize import Number
import scipy
import numpy as np

"""
Functions for question 1
"""
int_calls, int_adapt_calls = 0, 0

# Z is a number, the function we want will be returned as a function of u
def gen_dE(R, z):
    def dE(theta):
        radius = R
        return (z-radius*np.cos(theta))*np.sin(theta)/(radius**2 + z**2 - 2*radius*z*np.cos(theta))**(3/2)
    return dE

# The functions from this line to the functions for question 2 are taken from the class slides
def integrate(fun,a,b, dx_targ, ord=2):
    coeffs=integration_coeffs_legendre(ord+1)
    npt=np.int((b-a)/dx_targ)+1
    nn=(npt-1)%(ord)
    if nn > 0:
        npt=npt+(ord-nn)
    # assert(npt%(ord)==1)
    x=np.linspace(a, b, npt)
    dx=np.median(np.diff(x))
    dat=fun(x)
    # Testing
    print(dat[:-1])
    print([(npt-1)/(ord),ord])
    # 
    mat=np.reshape(dat[:-1],[(npt-1)/(ord),ord]).copy()
    mat[0,0]=mat[0,0]+dat[-1]
    mat[1:,0]=2*mat[1:,0]

    vec=np.sum(mat, axis=0)
    tot=np.sum(vec*coeffs[:-1])*dx
    return tot

def legendre_mat(npt):
    # Make a square legendre polynomial matrix of desired dimension
    x=np.linspace(-1,1,npt)
    mat=np.zeros([npt, npt])
    mat[:,0]=1.0
    mat[:,1]=x
    if npt>2:
        for i in range(1, npt-1):
            mat[:,i+1]=((2.0*i+1)*x*mat[:,i]-i*mat[:,i-1])/(i+1.0)
    return mat

def integration_coeffs_legendre(npt):
    # Find the integration coefficients using square
    # legendre polynomial matrix
    mat=legendre_mat(npt)
    mat_inv=np.linalg.inv(mat)
    coeffs=mat_inv[0,:]
    coeffs=coeffs/coeffs.sum()*(npt-1.0)
    return coeffs

"""
Functions for question 2 
"""
# adaptive integrator function from class
def integrate(fun,a,b,tol):
    global int_calls

    x=np.linspace(a,b,5)
    dx=x[1]-x[0]
    y=fun(x)
    #do the 3-point integral
    i1=(y[0]+4*y[2]+y[4])/3*(2*dx)
    i2=(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3*dx

    # Calculate how many times the function was called
    int_calls += 5

    myerr=np.abs(i1-i2)
    if myerr<tol:
        print("There were "+str(int_calls)+" function calls.")
        return i2
    else:
        mid=(a+b)/2
        int1=integrate(fun,a,mid,tol/2)
        int2=integrate(fun,mid,b,tol/2)
        return int1+int2

# Edit such that the function is not evaluated at multiple points
def integrate_adaptive(fun, a, b, tol, extra=None):
    # A variable to account for how many times we called the function 'fun'
    global int_adapt_calls 
    x = np.linspace(a, b, 5)

    if extra is None:
        y = fun(x)

        # increase the count as we did in the adaptive integrator function from class
        int_adapt_calls += 5
    else:
        # provide the end points of the new region which have already been calculated
        y = [extra[0], fun(x[1]), extra[1], fun(x[3]), extra[2]]

        # increase the count of function evals by 2
        int_adapt_calls += 2

    # Setting up the integral as we did in class:
    dx=x[1]-x[0]

    #do the 3-point integral
    i1=(y[0]+4*y[2]+y[4])/3*(2*dx)
    i2=(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3*dx

    # Definining the error
    myerr=np.abs(i1-i2)

    if myerr < tol:
        print("There were "+str(int_adapt_calls)+" function calls.")
        return i2
    else:
        left_area, right_area = integrate_adaptive(fun, a, x[2], tol/2, y[0:3]), integrate_adaptive(fun, x[2], b, tol/2, y[2:])
        return left_area + right_area

"""
Functions for Question 3
"""

# a) 
def log2fit(value: Number, calc_val=False):
    x = np.linspace(-1, 1, 1001)
    y = np.log2((x+3)/4)
    log2_val = 0

    cheb_coeffs=np.polynomial.chebyshev.chebfit(x,y, deg=25)
    cheb_new = []
    # Remove the coefficients below the tolerated threshold of 10^(-6)
    for coeff in cheb_coeffs:
        if abs(coeff) > 10**(-6):
            cheb_new.append(coeff)

    # Convert the chebyshev coefficients to polynomial coefficients
    poly_coeffs=np.polynomial.chebyshev.cheb2poly(cheb_new)
    poly = np.polynomial.Polynomial(poly_coeffs)

    # Re-shifting the x axis to be between 
    x_p = (x+3)/4
    y_cheb = poly(x)
    if calc_val and (value >= 0.5 and value <= 1):
        log2_val = poly(4*(value)-3)
    return x_p, y_cheb, log2_val

# b)
def mylog2(num: Number):
    if num <= 0:
        print("This number is less than or equal to zero. Please try another number")
    else:
        mantissa, exp = np.frexp(num)
        mantissa2, exp2 = np.frexp(np.e)
        # Since the mantissa will always be between 0.5 and 1, we can use our function which fits log2 between 0.5 and 1
        x, y, a = log2fit(mantissa, calc_val=True)
        x1, y1, b = log2fit(mantissa2, calc_val=True)
        print(b)
        val = (a+exp)/(b+exp2)
        return val