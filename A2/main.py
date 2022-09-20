from utils import *
import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":

    # Solution for Q1
    z_vec = np.linspace(0, 10, 11) # Remember that this must contain the R value
    a , b = 0, np.pi

    ans = []
    for zi in z_vec:
        func = gen_dE(R=1, z=zi)
        y = scipy.integrate.quad(func, a, b)
        ans.append(y[0])

    # Solution for Q2


'''TODO: 
- Add the integrator method result 
- Add the coloumb potential
''' 
# This currently shows the plot with the integrated solution using the scipy.integrate.quad method.
plt.plot(z_vec, ans)
plt.show()