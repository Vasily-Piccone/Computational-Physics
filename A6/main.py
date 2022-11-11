from utils import *
import matplotlib.pyplot as plt

j = complex(0,1)

if __name__ == "__main__":

    # Have to fix the shifting function. Seems to be bugging out
    x = np.linspace(-10, 10, 1001)
    y = np.exp(-0.5*(x)**2)
    y_shift = convo_shift(arr=y, shift_by=500)
    y_ss = convo_shift(arr=y_shift, shift_by=-500)
    # Uncomment when ready to save
    # plt.plot(x,y, x, y_shift)
    # plt.legend(['Original', 'Shifted'])
    # plt.savefig('./Shifted_Demo.png')


    # Question 2a): Add the plot of the Gaussian
    # gaussian_cor = cor_fun(f=y, g=y)
    # plt.plot(x, gaussian_cor)
    # plt.savefig('gaussian_correlation.png')

    # Question 2b): Can be completed once the time-shifting function is completed
    # shift = -250
    # gaussian_cor2 = cor_fun2(f=y, shift_by=shift)
    # Uncomment when you are ready to put the assignment together and better observe plots
    # plt.plot(x, gaussian_cor2)
    # plt.show()

    # Question 3: 
    # g1 = np.exp(-0.5*x**2)
    # x2 = np.linspace(-10, 10, 2001)
    # g2 = np.exp(-0.5*x2**2)
    # convo_gauss_3 = no_wrap_convolve(g1, g2)
    # xx2 = np.linspace(-10, 10, len(convo_gauss_3))
    # plt.plot(xx2, convo_gauss_3)
    # plt.show()

    # Question 4c): 
    k1, N = 6.5, 101 # Defining the constants
    x4, F_analytic = np.linspace(-10, 10, N), np.zeros(N, dtype=np.complex128) # Defining the arrays that we need

    # The numerical dft from numpy
    sin_kx = np.sin(2*np.pi*k1*x4/N)  # This is the sine wave
    k_vals= np.arange(N)[:N//2+1]
    F_semi_num = dft(sin_kx)  # This is the dft function I wrote (looks good right now)

    # The numpy fft
    F_num = np.fft.rfft(sin_kx)  

    # Plotting of the ffts
    # plt.plot(k_vals, np.abs(F_num), '-', k_vals, np.abs(F_semi_num[:N//2+1]), '--')
    # plt.legend(['F_numerical', 'F_semi_numerical'])
    # plt.plot(k_vals, np.abs(F_num)-np.abs(F_semi_num[:N//2+1]))
    # plt.show()


    # Question 4d):
    window = 0.5 - 0.5*np.cos(2*np.pi*x4)
    window_f = np.fft.rfft(window)
    fixed_fft = no_wrap_convolve(f=F_num, g=window_f)
    dup_fixed_fft = np.copy(fixed_fft)
    np.append(dup_fixed_fft, fixed_fft[-1])
    print(len(dup_fixed_fft))
    plt.plot(k_vals, dup_fixed_fft)
    plt.show()


    # Question 4e):



    # Question 5a):
    