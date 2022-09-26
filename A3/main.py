from utils import *

if __name__ == "__main__":
    """
    Solution for Question 1
    """


    """
    Solution for Question 2
    """
    # U238, Th234, Pro234, U234, Th230, Ra226, Rd222, Pl218, Pb214, Bs214, Pl214, Pb210, Bs210, Pl210 -> Then Lead
    u238 = (365*24*60*60)*4468*10**9
    Th234 = 24.10*24*60*60
    Pr234 = 6.70*60*60
    u234 = 245500*(365*24*60*60)
    Th230 = 75380*(365*24*60*60)
    Ra226 = 1600**(365*24*60*60)
    Rd222 = 3.8235*24*60*60
    Po218 = 3.10*60
    Pb214 = 26.8*60
    Bi214 = 19.9*60
    Po214 = 164.3*10**(-6)
    Pb210 = 22.3*365*24*60*60
    Bi210 = 5015.0*365*24*60*60
    Po210 = 138376*60*60

    halflife = [u238, Th234, Pr234, u234, Th230, Rd222, Po218, Pb214, Bi214, Po214, Pb210, Bi210, Po210] # In seconds
    lamb = [np.log(2)/(halflife[i]) for i in range(len(halflife))]
    
    print(lamb)
    b = decay_solver(lamb)