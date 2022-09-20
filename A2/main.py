from utils import *
import numpy as np
import matplotlib.pyplot as plt




if __name__ == "__main__":

    # Solution for Q1
    z_vec = np.linspace(0, 10, 11) # Remember that this must contain the R value
    a , b = 0, np.pi
    
    scipy_ans = []
    integrator_ans = []
    for zi in z_vec:
        func = gen_dE(R=1, z=zi)
        y1 = scipy.integrate.quad(func, a, b)
        # if zi != 0:
        #     y2 = integrate(func, a, b, 10**(-5))
        #     integrator_ans.append(y2)  
        scipy_ans.append(y1[0])
    '''
    TODO: 
    - Add the integrator method result 
    - Add the coloumb potential
    ''' 
    # This currently shows the plot with the integrated solution using the scipy.integrate.quad method.
    plt.plot(z_vec, scipy_ans)
    # plt.plot(z_vec, )
    plt.show()

    # Solution for Q2

    # Solution for Q3 
    """
    TODO:
    - Write out some more information here, and properly outline the solution of the problem
    - 
    """
    x, log2 = log2fit()
    print(log2)
    plt.plot(x, log2(x))
    plt.show()

