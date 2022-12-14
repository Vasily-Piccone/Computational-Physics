import numpy as np
import matplotlib.pyplot as plt
from poisson import *


# Size of the grid
N = 100

# Defining A matrix
diag = 1*np.ones(N)
off_diag = -1/4*np.ones(N-1)
A = np.diag(diag,k=0) + np.diag(off_diag,k=1) + np.diag(off_diag,k=-1)

# Initial guess (x vector)
x_g = np.zeros(N)
x_g[0] = 1

# Initial guess (b vector)
b = np.zeros((N,1))
b[0] = 1

# Nonsense
mat = np.ones(shape=(N, N))
L = np.tril(mat, k=0)
print(L)

M_inv = L@L.T
print(M_inv)

A_hat, x_hat, b_hat = L.T@A@L, np.linalg.pinv(L)@x_g, L.T@b 

x = LinearCG(A=A_hat, b=b_hat, x0=x_hat)
print(x)





# # NumNodes length = 19


# Showing the solution
# plt.imshow(x)
# plt.colorbar()
# plt.show()