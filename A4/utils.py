import numpy as np

def newtons_method_lorentz(data, p, t): 
    x_true = data['signal']
    iter = 0
    p_g = np.copy(p)
    for i in range(100):
        iter +=1
        pred, grad = calc_lorentz(p_g,t)  
        r = x_true - pred  
        r = np.matrix(r).transpose()  
        grad = np.matrix(grad)
        dp = np.linalg.pinv(grad.transpose()*grad)*(grad.transpose()*r)

        for i in range(len(p_g)):
            p_g[i] += dp[i]
        p_g = np.asarray(p_g)
        print(p_g, dp, iter)
    return p_g

# Newton's method with the numerical derivatives
def newtons_method_lorentz_numerical(data, p, t):
    x_true = data['signal']
    iter = 0
    p_g = np.copy(p)
    for i in range(100):
        iter +=1
        pred, grad = calc_lorentz_numerical(p_g,t)  
        r = x_true - pred  
        r = np.matrix(r).transpose()  
        grad = np.matrix(grad)
        dp = np.linalg.pinv(grad.transpose()*grad)*(grad.transpose()*r)
 
        for i in range(len(p_g)):
            p_g[i] += dp[i]
        p_g = np.asarray(p_g)
        print(p_g, dp, iter)
    return p_g


# This is taken from the "modelling" slides
def calc_lorentz(p, t):
    a, t0, w = p[0], p[1], p[2]
    y = a/(1+((t-t0)/w)**2)
    grad = np.zeros([len(t),len(p)])
    grad[:,0] = 1/(1+((t-t0)/w)**2)
    grad[:,1] = (2*a*w**2*(t-t0)**2)/(w**2+(t-t0)**2)**2
    grad[:,2] = (2*a*w*(t-t0)**2)/(w**2+(t-t0)**2)**2
    return y, grad

# Numerical Lorentz 
def calc_lorentz_numerical(p, t):
    a, t0, w = p[0], p[1], p[2]
    y = a/(1+((t-t0)/w)**2)
    # We can re-parameterize the function in terms of a, t0, and w
    fa = lambda u: u/(1+((t-t0)/w)**2)
    ft0 = lambda v: a/(1+((t-v)/w)**2)
    fw = lambda q: a/(1+((t-t0)/q)**2)

    # We now numerically define the gradient
    grad = np.zeros([len(t),len(p)])
    grad[:,0] = dfdx(fa, a)
    grad[:,1] = dfdx(ft0, t0)
    grad[:,2] = dfdx(fw, w)

    return y, grad

# Numerical derivative function
dt = 10**(-8)
dfdx = lambda f, t: 1/(2*dt)*(f(t+dt) - f(t-dt))


# Functions from Q1d
# _______________________________________________________
def calc_lorentzian_sum(p, t):
    a, b, c, t0, t1, w = p
    y = a/(1+((t-t0)/w)**2) + b/(1+((t-t0+t1)/w)**2) + c/(1+((t-t0-t1)/w)**2)
    # Reparameterizing the functions in terms of best-fit parameters
    fa = lambda x1: x1/(1+((t-t0)/w)**2) + b/(1+((t-t0+t1)/w)**2) + c/(1+((t-t0-t1)/w)**2)
    fb = lambda x2: a/(1+((t-t0)/w)**2) + x2/(1+((t-t0+t1)/w)**2) + c/(1+((t-t0-t1)/w)**2)
    fc = lambda x3: a/(1+((t-t0)/w)**2) + b/(1+((t-t0+t1)/w)**2) + x3/(1+((t-t0-t1)/w)**2)
    ft0 = lambda x4: a/(1+((t-x4)/w)**2) + b/(1+((t-x4+t1)/w)**2) + c/(1+((t-x4-t1)/w)**2)
    ft1 = lambda x5: a/(1+((t-t0)/w)**2) + b/(1+((t-t0+x5)/w)**2) + c/(1+((t-t0-x5)/w)**2)
    fw = lambda x6: a/(1+((t-t0)/x6)**2) + b/(1+((t-t0+t1)/x6)**2) + c/(1+((t-t0-t1)/x6)**2)

    grad = np.zeros([len(t), len(p)])
    grad[:,0], grad[:,1], grad[:,2] = dfdx(fa, a), dfdx(fb, b), dfdx(fc, c)
    grad[:,3], grad[:,4], grad[:,5] = dfdx(ft0, t0), dfdx(ft1, t1), dfdx(fw, w)

    return y, grad

# Newton's method function for the sum of Lorentzians 
def newtons_method_q1d(data, p, t):
    x_true = data['signal']
    iter = 0
    p_g = np.copy(p)
    for i in range(100):
        iter +=1
        pred, grad = calc_lorentzian_sum(p_g,t)  
        r = x_true - pred  
        r = np.matrix(r).transpose()  
        grad = np.matrix(grad)
        dp = np.linalg.pinv(grad.transpose()*grad)*(grad.transpose()*r)
 
        for i in range(len(p_g)):
            p_g[i] += dp[i]
        p_g = np.asarray(p_g)
        print(p_g, dp, iter)
    return p_g

# MCMC Function
def mcmc(p, t, signal, noise, step_size, num_steps):
    # Calculate initial chi sqaured value
    chisq_i = chisq(p, t, signal, noise)

    # Define the arrays which will hold the Monte-Carlo results and the chi squared values at each step
    chain = np.zeros([num_steps, len(p)])
    chisq_vals = np.zeros(num_steps)

    # loop through the mcmc 
    for i in range(num_steps):
        # Update the p value
        p_new = p + step_size*np.random.randn(len(p))
        chisq_f = chisq(p_new, t, signal, noise)
        if chisq_f < chisq_i:
            accept = True
        else:
            dchisq = chisq_f-chisq_i
            # print("dchisq:", dchisq, "iteration:", i)
            if np.random.rand(1)<np.exp(-0.5*(dchisq)):
                accept = True
            else: 
                accept=False
        if accept:
            p, chisq_i = p_new, chisq_f
        chain[i,:] = p
        chisq_vals[i] = chisq_i

    return chain, chisq_vals

# Chi squared function
def chisq(p, t, signal, noise):
    y, grad = calc_lorentzian_sum(p, t)
    chisquared = np.sum(((signal-y/noise)**2))
    return chisquared

def get_step(trial_step):
    if len(trial_step.shape)==1:
        return np.random.randn(len(trial_step))*trial_step
    else:
        L=np.linalg.cholesky(trial_step)
        return L@np.random.randn(trial_step.shape[0])