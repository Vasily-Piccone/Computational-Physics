from utils import *
import time
import matplotlib .pyplot as plt

# differentiate the get_spectrum function to get A
if __name__ == "__main__":
    # Question 1
    dirname = os.path.dirname(__file__)
    planck_1 = os.path.join(dirname, 'mcmc/COM_PowerSpect_CMB-TT-full_R3.01.txt')
    planck=np.loadtxt(planck_1, skiprows=1)
    multipole, var_multipole, d_uncertainty = planck[:,0], planck[:,1], (planck[:,2]+planck[:,3])/2
    number_of_points = len(multipole)

    params = [6.90000000e+01, 2.20000000e-02, 1.20000000e-01, 6.00000000e-02, 2.15972543e-09, 9.50000000e-01]# [69, 0.022, 0.12, 0.06, 2.1e-9, 0.95]

    params_test = [6.89999913e+1, 3.55206536e-02, 1.16992286e-01, 6.00063315e-02, 3.54844104e-09, 9.49991204e-01]
  

    # Question 2
    # Some error in the steps not changing. Either change your gradient calculator or change your 
    # Newtons function

    start_q2 = time.time()
    # Re-using parameters from the previous example
    # Calculate the spectrum and gradient
    model, grad = get_spectrum(params)[:number_of_points], model_params(params, num_pts=number_of_points)

    # Calculate the parameter values using Newton's method and calculate the errors
    p_vals= newtons(var_multipole, params, number_of_points, iterations=25)
    cov, errs = errors(grad, model)
    print("The p-values are: ", p_vals)
    print("The associated errors are: ", errs)
    file_q2 = open("planck_fit_params.txt", "w+")
    parameter_str, err_str = str(p_vals), str(errs)
    content = "Parameters: " + parameter_str +"\n" + "Errors: " + err_str +"\n"
    file_q2.write(content)
    file_q2.close()
    print("Question 2 took ", time.time()-start_q2, "seconds to run.")

    # Question 3
    start_q3 = time.time()
    param_error = np.sqrt(np.diag(cov))
    print("Error: ",param_error)
    params_q3 = np.copy(params)
    size_of_steps = np.copy(param_error)

    markov_chain_length = 10000
    out_chain, chisq_vec = mcmc(params=params_q3, data=var_multipole, function=chi_sq, noise=d_uncertainty, step_size=size_of_steps, num_steps=markov_chain_length)
    print(out_chain.shape, chisq_vec.shape)
    print("Paramaters: ", np.mean(out_chain, axis=0), "Parameter errors: ", np.std(out_chain, axis=0))

    # Write to the file
    new_array = np.zeros([markov_chain_length, len(params)+1])
    new_array[:,0] = np.copy(chisq_vec)
    new_array[:,[1,2,3,4,5,6]] = np.copy(out_chain)
    np.savetxt("planck_chain.txt", new_array)
    print("Question 3 took ", time.time()-start_q3, "seconds to run.")
    a_vals, b_vals, c_vals, t0_vals, t1_vals, omega_vals, i_vals = out_chain[:,0], out_chain[:,1], out_chain[:,2], out_chain[:,3], out_chain[:,4], out_chain[:,5], np.arange(0, markov_chain_length, 1)

    plt.loglog(i_vals,np.abs(np.fft.fft(b_vals))**2, label='Baryon Density Chain')
    plt.xlabel('Logarithm of Chain Index')
    plt.ylabel(r'$\Omega_bh^{2}$')
    plt.legend()
    plt.savefig('Question3.png')


    # Question 4
    # Load in the text file from question 3
    start_q4 = time.time()
    data = np.loadtxt("planck_chain.txt")
    chisq_vector = data[:,0]
    out_chain = data[:,[1,2,3,4,5,6]]
    v1, v2, v3, v4, v5, v6 = np.mean(out_chain[:,0]), np.mean(out_chain[:,1]), np.mean(out_chain[:,2]), np.mean(out_chain[:,3]), np.mean(out_chain[:,4]), np.mean(out_chain[:,5])
    err1, err2, err3, err4, err5, err6 = np.std(out_chain[:,0]), np.std(out_chain[:,1]), np.std(out_chain[:,2]), np.std(out_chain[:,3]), np.std(out_chain[:,4]), np.std(out_chain[:,5])
    print("Values: ")
    print(v1, err1, "\n")
    print(v2, err2, "\n")
    print(v3, err3, "\n")
    print(v4, err4, "\n")
    print(v5, err5, "\n")
    print(v6, err6, "\n")

    h = v1/100
    omb_q3 = v2/h**2
    omc_q3 = v3/h**2
    om_lambda = 1 - omb_q3 - omc_q3
    print("Omega_lambda: ", om_lambda)
    sigma_lambda = np.sqrt(omb_q3**2 + omc_q3**2)
    print("sigma_lambda: ", sigma_lambda)

    # Define covariance matrix and error. Picking after 115 for better chi squared values
    cov_q4 = np.cov(out_chain[115:].transpose())
    error_q4 = np.sqrt(np.diag(cov_q4))

    # Calculating probabilities
    print(out_chain[0,115:9999])
    prob_i, prob_f = np.random.normal(out_chain.transpose()[0][115:]), np.exp(-0.5*(out_chain.transpose()[3][115:]-tau)**2)/tau_error
    prob_new = prob_i + prob_f

    weights = prob_new/np.sum(prob_new) # Normalizing the new probability 
    importance_params = np.sum(weights*out_chain[115:].transpose(),axis=1)
    importance_errs = np.sqrt(np.sum(weights*((out_chain[115:]-importance_params)**2).transpose(), axis=1))
    print("Importance sampling parameters: ", importance_params, "\n")
    print("Importance sampling errors: ", importance_errs, "\n")

    # Re-reunning mcmc
    best_fit_params, step_size = np.copy(importance_params), np.copy(error_q4)
    out_chain_q4, chisq_vector_q4 = mcmc(params=best_fit_params, data=var_multipole, function=chi_sq_constrained, noise=d_uncertainty, step_size=step_size, num_steps=10000)

    # Create the object which needs to be saved
    new_array_q4 = np.zeros([markov_chain_length, len(params)+1])
    new_array_q4[:,0] = np.copy(chisq_vector_q4)
    new_array_q4[:,[1,2,3,4,5,6]] = np.copy(out_chain_q4)


    # Load in the data
    data = np.loadtxt("planck_chain_tauprior.txt")
    out_chain = data[:, [1,2,3,4,5,6]]

    v1, v2, v3, v4, v5, v6 = np.mean(out_chain[:,0]), np.mean(out_chain[:,1]), np.mean(out_chain[:,2]), np.mean(out_chain[:,3]), np.mean(out_chain[:,4]), np.mean(out_chain[:,5])
    err1, err2, err3, err4, err5, err6 = np.std(out_chain[:,0]), np.std(out_chain[:,1]), np.std(out_chain[:,2]), np.std(out_chain[:,3]), np.std(out_chain[:,4]), np.std(out_chain[:,5])
    print("Values (Q4): ")
    print(v1, err1, "\n")
    print(v2, err2, "\n")
    print(v3, err3, "\n")
    print(v4, err4, "\n")
    print(v5, err5, "\n")
    print(v6, err6, "\n")

    i_vals = np.arange(0, markov_chain_length, 1)
    plt.loglog(i_vals,np.abs(np.fft.fft(out_chain[:,1]))**2, label='Baryon Density Chain')
    plt.xlabel('Logarithm of Chain Index')
    plt.ylabel(r'$\Omega_bh^{2}$')
    plt.legend()
    plt.savefig('Question4_fft.png')

    np.savetxt('planck_chain_tauprior.txt',new_array_q4)
    print("Q4 took ", time.time()-start_q4, " seconds to run.")