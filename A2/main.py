from utils import *
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Solution for Q1
    z_vec = np.linspace(0, 10, 1001) # Remember that this must contain the R value
    a , b = 0, np.pi
    r = 3

    scipy_ans = []
    integrator_ans = []
    coloumb_ans = np.zeros(1001)
    # for i in range(1001):
    #     if z_vec[i] > r:
    #         # This isn't the exact representation, but we're curious about qualitative behaviour
    #         coloumb_ans[i] = 2*r**2/z_vec[i]**2 

    for zi in z_vec:
        func = gen_dE(R=r, z=zi)
        y1 = scipy.integrate.quad(func, a, b)
        if zi != 0:
            y2 = integrate(func, a, b, 10**(-5))
            integrator_ans.append(y2)  
        scipy_ans.append(r**2*y1[0])
    
    # This currently shows the plot with the integrated solution using the scipy.integrate.quad method.
    plt.plot(z_vec, scipy_ans, label='Scipy answer')
    plt.plot(z_vec, coloumb_ans, label='Analytic answer')
    plt.xlabel("z distance")
    plt.ylabel("field strength")
    plt.legend(loc="upper left")
    plt.show()

    # Solution for Q2
    # defining some test functions
    a, b = 0, np.pi
    tol = 10**(-5)
    fun1 = np.exp
    fun2 = np.cos
    fun3 = lambda x: 1/(np.exp(-x)+1)  

    slow1, fast1 = integrate(fun1, a, b, tol), integrate_adaptive(fun1, a, b, tol, extra=None)
    print(slow1, fast1)
    print(int_calls, int_adapt_calls) 

    slow2, fast2 = integrate(fun2, a, b, tol), integrate_adaptive(fun2, a, b, tol, extra=None)
    print(slow2, fast2)
    print(int_calls, int_adapt_calls) 

    slow3, fast3 = integrate(fun3, a, b, tol), integrate_adaptive(fun3, a, b, tol, extra=None)
    print(slow3, fast3)
    print(int_calls, int_adapt_calls) 


    # Solution for Q3 
    # Q3a)
    x, y, value = log2fit(0.5, calc_val=True)
    # Uncomment this section to see the plot of log2 between 0.5 and 1
    # plt.plot(x, y)
    # plt.show()

    # Q3b)
    value = mylog2(np.e)
    print(value)