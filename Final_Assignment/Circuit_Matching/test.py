import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from thruster import *

print(sp.special.jv(1, 2.405)) # This works

length = 100
t= np.linspace(0, length, 101)
noise = np.random.normal(0,1,101)

power = 300*np.heaviside(length - t, length - t) 

plt.figure(1)
plt.plot(t, noise)

plt.figure(2)
plt.plot(t, noise + power)
plt.show()