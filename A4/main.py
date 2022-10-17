import numpy as np
import matplotlib.pyplot as plt
import statistics as stats
from utils import *

if __name__ == "__main__":
    # Question 1a
    #________________________________________________________________
    data=np.load('./mcmc/sidebands.npz')
    t=data['time']
    d=data['signal']

    # Plot of the raw data. Uncomment to compare with the plots of other models
    # plt.plot(t,d)
    
    p_guess = [1.5, 0.000192, 0.00001]
    p_calc=newtons_method_lorentz(data, p_guess, t)
    y, grad = calc_lorentz(p_calc, t)
    # Uncomment the following two lines to see the plot, or reference the assignment pdf
    # plt.plot(t, y)
    # plt.show()  


    # Question 1b
    # ________________________________________________________________
    sigma = stats.stdev(d)
    standard_error = sigma/np.sqrt(len(d))  # computing standard error 
    print(standard_error)

    # Computing the covariance matrix 
    # cov_inv = np.identity(len(t))/standard_error**2
    # cov = np.linalg.pinv(grad.transpose()@cov_inv@grad)
    # errs = np.sqrt(np.diag(cov))
    # print("The errors on a, t0 and w respectively are:", errs)
    # UNCOMMENT BEFORE SUBMITTING


    # Question 1c
    # ________________________________________________________________
    p_calc=newtons_method_lorentz_numerical(data, p_guess, t)
    yc, gradc = calc_lorentz(p_calc, t)
    print("The best-fit parameters for the Lorentzian using numerical derivatives are: ", p_calc)
    # Uncomment to see the plot, or refer to the assignment pdf
    # plt.plot(t, yc)
    # plt.show()

    # Question 1d
    # ________________________________________________________________
    p_1d = [1.5, 0.3, 0.35, 0.000192, 0.00005, 0.00001]  #a, b, c, t0, t1, w
    p_calc_1d=newtons_method_q1d(data, p_1d, t) 
    yd, gradd=calc_lorentzian_sum(p_calc_1d, t)
    print("The best-fit parameters for the sum of Lorentzians using numerical derivatives are: ", p_calc_1d)

    # Uncomment the following two lines to get the plot or refer to the assignment pdf
    # plt.plot(t, yd)
    # plt.show()

    # Computing the covariance matrix. Standard error was re-used from question 1b
    cov_inv_d = np.identity(len(t))/standard_error**2
    cov_d = np.linalg.pinv(gradd.transpose()@cov_inv_d@gradd)
    errs_d = np.sqrt(np.diag(cov_d))
    # print("The errors on a, b, c, t0, t1, and w respectively are:", errs_d)
    # UNCOMMENT BEFORE SUBMITTING. COMMENTED OUT TO SAVE RUN TIME

    # Question 1e)
    # __________________________________________________________________
    residual = d - yd
    # Uncomment to see this plot, or refer to assignment pdf
    # plt.plot(t, residual)
    # plt.show()

    # Question 1f)
    # __________________________________________________________________
    # Calculating random, slightly perturbed p arrays
    # p_rand1, p_rand2, p_rand3 = p_calc_1d + cov_d@np.random.randn(len(p_calc_1d)), p_calc_1d + cov_d@np.random.randn(len(p_calc_1d)), p_calc_1d + cov_d@np.random.randn(len(p_calc_1d))

    # Calculate their corresponding Lorentzians and residues (Uncomment when submitting)
    # ye1, grad_e1 = calc_lorentzian_sum(p_rand1, t)
    # ye2, grad_e2 = calc_lorentzian_sum(p_rand2, t)
    # ye3, grad_e3 = calc_lorentzian_sum(p_rand3, t)

    # res1 = d - ye1
    # res2 = d - ye2
    # res3 = d - ye3

    # Calculate corresponding chi squared, and compare it with the chi squared from question 1d
    # chisq1, chisq2, chisq3 = res1.transpose()@cov_inv_d@res1, res2.transpose()@cov_inv_d@res2, res3.transpose()@cov_inv_d@res3
    chisq = residual.transpose()@cov_inv_d@residual
    print(chisq)


    # Question 1g
    # ____________________________________________________
    print("------------")
    print("question 1g)")
    step_size_g = np.sqrt(np.diag(cov_d))  # Determine the step size using the covariance matrix from d)
    steps = 20000
    p_1g = [1.44299239, 0.103910780, 0.0647325300, 0.000192578522, 0.0000445671630, 0.0000160651094] # Best fit params obtained from question 1d
    chain, chisq_vals = mcmc(p=p_1g, t=t, signal=d, noise=standard_error, step_size=step_size_g, num_steps=steps)
    a_vals, b_vals, c_vals, t0_vals, t1_vals, omega_vals, i_vals = chain[:,0], chain[:,1], chain[:,2], chain[:,3], chain[:,4], chain[:,5], np.arange(0, steps, 1)
    plt.plot(i_vals, b_vals)
    plt.show()