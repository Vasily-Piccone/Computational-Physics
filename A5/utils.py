import numpy as np
import camb
import stats
import os

# The length of the data file found in question 1
dirname = os.path.dirname(__file__)
planck_1 = os.path.join(dirname, 'mcmc/COM_PowerSpect_CMB-TT-full_R3.01.txt')
planck=np.loadtxt(planck_1, skiprows=1)
multipole, var_multipole, d_uncertainty = planck[:,0], planck[:,1], (planck[:,2]+planck[:,3])/2
number_of_points = len(multipole)

# Constrained optical parameter
tau = 0.054
tau_error = 0.0074

# Re-using the get_spectrum function from prof. Sievers' example code:
def get_spectrum(pars,lmax=3000):
    H0, ombh2, omch2, tau, As, ns = pars
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau) # determining universe-type based on the model
    pars.InitPower.set_params(As=As,ns=ns,r=0) # creating the data from the parameters 
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt[2:]

# Numerical derivative calculation (Re-used from assignment 4)
dt = 10**(-8)
dfdx = lambda f, t: 1/(2*dt)*(f(t+dt) - f(t-dt))

# This function calculates the A matrix needed for the non-linear best-fit 
def model_params(params, num_pts):
    H0, ombh2, omch2, tau, As, ns = params
    
    # Re-parameterizing function in terms of best-fot parameters
    # Concatenate an element containing the variable with the rest of the variable
    fH0 = lambda x1 : get_spectrum(np.concatenate((np.array([x1]), np.array([ombh2, omch2, tau, As, ns]))))[:number_of_points]
    fombh2 = lambda x2 : get_spectrum(np.concatenate((np.array([H0]), np.array([x2]), np.array([omch2, tau, As, ns]))))[:number_of_points]
    fomch2 = lambda x3 : get_spectrum(np.concatenate((np.array([H0, ombh2]), np.array([x3]), np.array([tau, As, ns]))))[:number_of_points]
    ftau = lambda x4 : get_spectrum(np.concatenate((np.array([H0, ombh2, omch2]), np.array([x4]),np.array([As, ns]))))[:number_of_points]
    fAs = lambda x5 : get_spectrum(np.concatenate((np.array([H0, ombh2, omch2, tau]), np.array([x5]), np.array([ns]))))[:number_of_points]
    fns = lambda x6 : get_spectrum(np.concatenate((np.array([H0, ombh2, omch2, tau, As]), np.array([x6]))))[:number_of_points]
    
    # Calculating the gradient (A) matrix
    grad = np.zeros([num_pts, len(params)])
    grad[:,0], grad[:,1], grad[:,2] = dfdx(fH0, H0), dfdx(fombh2, ombh2), dfdx(fomch2, omch2)
    grad[:,3], grad[:,4], grad[:,5] = dfdx(ftau, tau), dfdx(fAs, As), dfdx(fns, ns)
    return grad

# Newton's method to solve the non-linear best fit problem for Q2
def newtons(data_vector, params, num_pts, iterations=25):
    cov_inv = np.identity(d_uncertainty.size)/(d_uncertainty**2)
    data = data_vector
    p_guess = np.copy(params)

    # Beginning of the Newton's method loop
    for i in range(iterations):
        spec = get_spectrum(p_guess)[:num_pts]
        r = data - spec
        r = np.matrix(r).transpose()
        # Calculate gradient and inverse covariance matrix
        grad = model_params(p_guess, num_pts)
        grad = np.matrix(grad)
        # Chilling
        dp = np.linalg.pinv(grad.transpose()*cov_inv*grad)*(grad.transpose()*cov_inv*r)
        for j in range(len(p_guess)):
            p_guess[j] += dp[j]
        p_guess = np.asarray(p_guess)
        print(p_guess, dp, i)
    return p_guess

def errors(grad, data_vector):
    sigma = stats.stdev(data_vector)
    standard_error = sigma/np.sqrt(len(data_vector))  # computing standard error 
    cov_inv = np.identity(len(data_vector))/standard_error**2 # Computing the inverse of the covariance matrix
    cov = np.linalg.pinv(grad.transpose()@cov_inv@grad, rcond = 1e-16)
    errs = np.sqrt(np.diag(cov))
    return cov, errs 


# MCMC sampler for question 3
def mcmc(params, data, function, noise, step_size, num_steps=1000):
    chisq_i = function(params, data, noise)

    chain, chisq_vector = np.zeros([num_steps, len(params)]), np.zeros(num_steps)

    for i in range(num_steps):
        # incementing the params and calculating chi squared values
        cur_params = params + step_size*np.random.randn(len(params))
        chisq_f = function(cur_params, data, noise)
        d_chisq = chisq_f - chisq_i
        
        # acceptance probability
        prob = np.exp(-0.5*d_chisq)
        acceptance_cond = np.random.rand(1)<prob
        
        # Defining whether or not the chain iteration is accepted
        if acceptance_cond:
            
            # Updating values
            params = cur_params
            chisq_i = chisq_f
         
        # Saving the parameters and chisquares to some output arrays    
      
    
    return chain, chisq_vector

def chi_sq(params, data, noise):
    pred = get_spectrum(params)[:number_of_points]
    chisq = np.sum(((data-pred)/noise)**2)
    return chisq

def chi_sq_constrained(params, data, noise):
    # Unconstrained chi squared
    chisq = chi_sq(params, data, noise)
    chisq_optical = ((params[3] - tau)**2/(tau_error**2))
    chisq_tot = chisq + chisq_optical
    return chisq_tot