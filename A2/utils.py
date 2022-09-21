from tokenize import Number
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

def integrate_adaptive(fun, a, b, tol, extra=None):
    pass




"""
Functions for Question 3
"""

# a) 
def log2fit():
    x = np.linspace(-1, 1, 11)
    y = np.log2((x+3)/4)
    
    log2coeffs=np.polynomial.chebyshev.chebfit(x,y, deg=10)
    log2=np.polynomial.chebyshev.Chebyshev.fit(x,y, deg=10)
    return x, log2
    pass

# b)
def mylog2(num: Number):
    if num <= 0:
        print("This number is less than or equal to zero. Please try another number")
    else:
        mantissa, exp = np.frexp(num)
        print(mantissa, exp)
    pass