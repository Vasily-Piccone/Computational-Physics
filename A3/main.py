from utils import *

if __name__ == "__main__":
    """
    Solution for Question 1
    """
    h = 0.2
    x = np.linspace(-20, 20, 200)
    y0 = 1

    rhs = lambda x, y: y/(1+x**2)

    sol1 = [y0]
    sol2 = [y0]
    analytic = 1/(np.exp(np.arctan(-20)))*np.exp(np.arctan(x))
    for i in range(1, len(x)):
        sol1.append(rk4_step(rhs, x[i], sol1[i-1], h)) 
        sol2.append(rk4_stepd(rhs, x[i], sol2[i-1], h))

    err1 = analytic - sol1
    err2 = analytic - sol2
    # plt.plot(x, err1, x, err2)
    # plt.legend(['sol1 error', 'sol2 error'], loc="upper left")
    # plt.show()

    """
    Solution for Question 2
    """
    # U238, Th234, Pro234, U234, Th230, Ra226, Rd222, Pl218, Pb214, Bs214, Pl214, Pb210, Bs210, Pl210 -> Then Lead
    # Might make sense to rescale everything here, and then rescale to retrive the answer
    u238 = (365*24*60*60)*4468*10**9
    Th234 = 24.10*24*60*60
    Pr234 = 6.70*60*60
    u234 = 245500*(365*24*60*60)
    Th230 = 75380*(365*24*60*60)
    Ra226 = 1600*(365*24*60*60)
    Rd222 = 3.8235*24*60*60
    Po218 = 3.10*60
    Pb214 = 26.8*60
    Bi214 = 19.9*60
    Po214 = 164.3*10**(-6)
    Pb210 = 22.3*365*24*60*60
    Bi210 = 5015.0*365*24*60*60
    Po210 = 138376*24*60*60
    Pb206 = (365*24*60*60)*4468*10**9

    halflife = [u238, Th234, Pr234, u234, Th230, Rd222, Po218, Pb214, Bi214, Po214, Pb210, Bi210, Po210, Pb206] # In seconds
    lamb = [np.log(2)/(halflife[i]) for i in range(len(halflife))]

    # b = decay_solver(lamb) # Uncomment to get the results. Look at the function "decay_solver" to change between plot a and b

    """
    Solution to Question 3
    """
    path = "./dish_zenith.txt"
    a, x0, y0, z0, cov = photogrammetry(path)
    print(a, x0, y0, z0)
    print(cov)
    uncert_a = cov[0,0]
    f, f_err = 1/(4*a), np.sqrt(uncert_a/(4*a**2))
    print(f, f_err)
