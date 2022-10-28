from sympy import multigamma
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

    params = [69, 0.022, 0.12, 0.06, 2.1e-9, 0.95]


    # Question 2
    # Some error in the steps not changing. Either change your gradient calculator or change your 
    # Newtons function

    # start_q2 = time.time()
    # # Re-using parameters from the previous example
    # # Calculate the spectrum and gradient
    model, grad = get_spectrum(params)[:number_of_points], model_params(params, num_pts=number_of_points)
    # # Calculate the parameter values using Newton's method and calculate the errors
    # p_vals= newtons(var_multipole, params, number_of_points, iterations=25)
    cov, errs = errors(grad, model)
    # print("The p-values are: ", p_vals)
    # print("The associated errors are: ", errs)
    # file_q2 = open("planck_fit_params.txt", "w+")
    # parameter_str, err_str = str(p_vals), str(errs)
    # content = "Parameters: " + parameter_str +"\n" + "Errors: " + err_str +"\n"
    # file_q2.write(content)
    # file_q2.close()
    # print("Question 2 took ", time.time()-start_q2, "seconds to run.")

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
    print(new_array)
    np.savetxt("planck_chain.txt", new_array)
    print("Question 3 took ", time.time()-start_q3, "seconds to run.")
    a_vals, b_vals, c_vals, t0_vals, t1_vals, omega_vals, i_vals = out_chain[:,0], out_chain[:,1], out_chain[:,2], out_chain[:,3], out_chain[:,4], out_chain[:,5], np.arange(0, markov_chain_length, 1)
    plt.plot(i_vals, a_vals)
    plt.show()


