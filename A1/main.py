import sympy as sym
from utils import *
v = sym.Symbol('v')

class a1:
    global eps

    def __init__(self):
        pass

    def taylor_roundoff():
        pass

    # Currently only handles one value of x.
    def ndiff(fun, x, full=False):
        # Have some code in here that determines whether eps is 10^-7 or 10^-16
        if False: #type(fun).__name__ != 'sympy.core.mul.Mul':  # This code is to catch errors will need to be fixed
            print("A function was not of the sympy library. Other function types are not supported.")
            #TODO: return derivative, return error, return dx
            return 0
        else:
            if full == False:
                eps = 10**(-7)
                fpp = fun.diff().diff()

                # determine which of the core atttributes this is 
                print(eps*fun.subs(v, x)/fpp.subs(v,x))
                dx = np.sqrt(np.abs(eps*fun.subs(v, x)/fpp.subs(v,x)))
                deriv = (fun.subs(v, x+dx)-fun.subs(v, x-dx))/(2*dx)
                return deriv
            else:
                pass # Fill this in with the rest of the nonsense

    def lakeshore(V, data):  # V is either a number or an array
        pass


# This is all for the question 4
    def spline_comparison():
        pass
    
    def poly_spline():
        pass

    def cubic_spline():
        pass

    def rational_interpol():
        pass
# print(sym.diff(v**5))

# expr = sym.cos(v) + 1
# print(expr.diff().diff())
# print(expr.subs(v, 1))

# # Can use has(v) to determine which variables are being used
# arguments = expr.has(v)
# print(arguments)

func = sym.cos(v*v)
x = 0 
assignment = a1()

val = a1.ndiff(func, x)
print(val)

