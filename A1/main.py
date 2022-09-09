from cmath import nan
import sympy as sym
import numpy as np


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

if __name__ == "__main__":
    fvar = sym.Symbol("u")

    func = sym.exp(fvar)
    x = 0
    q2 = ndiff(func, x)
    print(q2)