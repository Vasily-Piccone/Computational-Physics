
# imports 
import numpy as np
import scipy.signal as sp

j = complex(0,1)

# Question 1: Function that shifts an array using a convolution
def convo_shift(arr, shift_by: int):
    L = len(arr)  # Define length
    f = np.zeros(shape=(L,1))  # Make a copy of the array

    # To take into account instances where the shift is larger than the array length
    if shift_by > L:
        shift_by %= L

    # Defining the impulse array
    impulse = sp.unit_impulse(L, shift_by)
    f = np.fft.irfft(np.fft.fft(arr)*np.fft.fft(impulse), L)

    return f

# Question 2
# a) correlation function of two arrays
def cor_fun(f: list, g: list):
    return np.fft.irfft(np.fft.fft(f)*np.conjugate(np.fft.fft(g)))


# b) NOTE: the assumption is f is Gaussian
def cor_fun2(f: list, shift_by: int):
    fa = np.copy(f)
    g = convo_shift(fa, shift_by)
    return cor_fun(g, g)

# Question 3: Write a routine to take an FFT-based convolution of two arrays without any danger of wrapping around. 
# You may wish to add zeros to the end of the input arrays. 
def no_wrap_convolve(f: list, g: list):
    L_f, L_g = len(f), len(g)
    l1, l2, pad = np.copy(f), np.copy(g), np.zeros(np.abs(L_f - L_g))  # Why do I have to add 1 for this to work?
    if L_g > L_f:
        l1 = np.append(l1, pad)
    elif L_f > L_g:
        l2 = np.append(l2, pad)
    
    # Adding extra zeros to the back to further prevent wrap-around
    l1_padded = np.append(l1, pad)
    l2_padded = np.append(l2, pad)

    return np.fft.irfft(np.fft.rfft(l1_padded)*np.fft.rfft(l2_padded))

# This looks good
def dft(x: list):
    N = len(x)
    X, n = np.zeros(N, dtype=np.complex128), np.arange(N)
    for k in range(N):
        for i in range(N):
            X[k] += x[i]*np.exp(-2*np.pi*j*k*n[i]/N)
    return X

# For some reason, this did not yield promising results 
geometric_ft = lambda k, N:  (1-np.exp(-2*np.pi*j*k))/(1-np.exp(-2*np.pi*j*k/N))

# Question 5 functions
