import numpy as np
import matplotlib.pyplot as plt
from gas import *
import scipy as sp

import warnings
warnings.filterwarnings('ignore') # To remove the warnings associated with very large and very small numbers

"""
@Author: Vasily G. Piccone

N => Number of turns,
l_w=> length of wire,
b => thickness of dielectric enclosing the cylinder,
R => radius of thruster,
n_s => number density of plasma ~ 10^10 /cm^3 = 10^16 / m^3, 
nu_eff => average rate of collision (s^-1) 
"""

q_e = 1.602e-19  # charge of an electron
mu0 = 4 * np.pi * 10 ** (-7)  # Permeability of free space
m_e = 9.11e-31  # mass of an electron
nu_eff = 2.5e7  # s^-1 (assumption: nu_eff for argon and nitrogen will be similar) -> What are the values for these for other gases?

alpha = 0.25 # Duty cycle 0.25
Tau = 125e-6  # [s] Pulse period (1/pulse frequency) 1000e-6

class thruster(gas):
    def __init__(self, voltage, N, l_w, b, R, species: str, T_w):
        super().__init__(species, T_w) # Inheriting the gas class

        self.n_s = 10e18  # default plasma number density m^-3 
        self.sigma_eff = q_e ** 2 * self.n_s / (m_e * nu_eff)  # default effective conductivity
        self.delta_p = np.sqrt(m_e / (mu0 * self.n_s)) / q_e  # default collision-less skin depth
        
        # Defining the dimensions of the plasma
        self.R = R  
        self.l_w = l_w
        self.N = N
        self.b = b

        # Circuit parameters
        self.voltage = voltage
        self.R_t = 50  # Ohms (resistance of the power supply)
        self.C_2 = None  # this will be updated once the power matching function is called
        self.C_1 = 90e-8  # In units of F (90 nF) - Value of capacitor
        
        # TODO: find a robust formula for when b < R
        self.L_s = (mu0 * np.pi * N ** 2) / self.l_w * ((self.b / self.R) ** 2 - 1)
        self.R_s = (N ** 2 * 2 * np.pi * self.R) / (self.sigma_eff / self.delta_p)

        # Defining helper functions
        self.h_L = lambda Te: 0.86/(np.sqrt(3 + (self.l_w/2/self.lambda_i) + (0.86*self.l_w*self.uB(Te))/(np.pi*self.g(Te)*self.D_i)**2))  # normalized axial sheath edge density
        self.h_R = lambda Te: 0.8/np.sqrt(4 + (self.R/self.lambda_i)+(0.8*self.R*self.uB(Te)/(2.405*sp.special.jv(1, 2.405)*self.g(Te)*self.D_i)**2))  # normalized radial sheath edge density
        self.d_eff = lambda Te: 0.5*self.R*self.l_w/(self.R*self.h_L(Te) + self.l_w*self.h_R(Te))  # effective plasma size

        # Different states of the plasma
        self.n_s_pulsed = None # Keep blank until it is solved for

    #  Prints out the equivalent circuit parameters (resistance and inductance) of the ion thruster
    def circuit_param(self):
        print(f"The circuit params are (L_s, R_s) = {self.L_s, self.R_s}")

    # Calculates the capacitance C_2 needed to match the thruster's equivalent circuit for maximum power delivery
    def power_matched_capacitor(self, freq_i, freq_f, num_pts):
        freq_pts = np.linspace(freq_i, freq_f, num=num_pts)
        matched_cap = []
        for i in range(num_pts):
            w = (2 * np.pi * freq_pts[i])
            X = 1 / (w * self.C_1) + self.L_s * w
            cap = (1 / w) * X / (self.R_s ** 2 + X ** 2)
            matched_cap.append(cap)
        return freq_pts, matched_cap
    

    # This function has to inherit all gas values from the thruster, which the thruster should inherit from the gas
    def ode(self, t, y: list): 
        # Unpack the initial conditions
        ne, Te = y[0], y[1]

        # The power is defined as a pulse
        P = lambda t: Pave/alpha*np.heaviside(Tau*alpha - t, Tau*alpha - t)   # Is this correct? 

        # Adding gaussian noise to the power supply
        noise = 100*np.random.normal(0,1)

        We = lambda ne, Te: (3/2)*ee*ne*Te*np.pi*self.R**2*self.l_w  # This should go into the thruster 
        Vs = lambda Te: Te*np.log(self.m_kg/(2*np.pi*m_e))

        # particle conservation terms
        nu_iz, nu_loss = self.k_iz(Te)*ng, self.uB(Te)/self.d_eff(Te)  # Ionization rate [1/s], Loss rate [1/s]
        dndt = ne*(nu_iz - nu_loss)  # Particle balance

        # I have ne and Te as function inputs 
        dTedt = Te*(((P(t)+ noise)/We(ne, Te)) - ((2/3)*(self.e_c(Te)/Te) + 1)*nu_iz - ((2/3)*((Vs(Te) + (5/2)*Te)/Te) - 1)*nu_loss)  # Power balance
        
        dydt = [dndt, dTedt]
        return dydt

    # Add support here for default vs. pulsed plots for the matched capacitor calculation
    def solve_ode(self, init_cond: list, func, t_range):
        sol = sp.integrate.solve_ivp(fun=func, t_span=t_range, y0=init_cond)
        return sol
    

    # There must be a better way to structure this
    def pulsed_ns_circuit_params(self, ns: list, frequency):
        # Set the plasma number density for a power pulse to ns
        self.n_s_pulsed = ns 
        sigma_eff = q_e ** 2 * ns / (m_e * nu_eff)  # calculate the effective conductivity
        delta_p = np.sqrt(m_e / (mu0 * ns)) / q_e  # calculate the collision-less skin depth
        L_s = (mu0 * np.pi * self.N ** 2) / self.l_w * ((self.b / self.R) ** 2 - 1)  # Plasma inductance
        R_s = (self.N ** 2 * 2 * np.pi * self.R) / (sigma_eff / delta_p)  # Plasma Resistance

        w = (2 * np.pi * frequency)  # Angular frequency
        X = 1 / (w * self.C_1) + L_s * w  # Reactance 
        cap = (1 / w) * X / (R_s ** 2 + X ** 2)  # Matching capacitor value

        return cap
    
# Standard error
def standard_error(data):
    return np.std(data)/np.sqrt(len(data))

# Where things actually run
if __name__ == '__main__':
    VOLTAGE: float = 446.0  # voltage across terminals
    WINDINGS: int = 3  # number of windings in coil
    RADIUS: float = 0.03  # meters, radius of thruster tube through which plasma travels
    WIRE_LENGTH: float = WINDINGS*np.pi*RADIUS**2 + 0.1  # meters, where the 0.1 factor is the additional length for attaching the coil to the circuit
    DIELECTRIC_R: float = 0.003175  # thickness of dielectric in meters
    B: float = DIELECTRIC_R + RADIUS # Total radius of coil
    propellant: str = 'Ar'  # Selected propellant


    # instantiating thrusters 
    mk_1 = thruster(VOLTAGE, WINDINGS, WIRE_LENGTH, B, RADIUS, propellant, T_w=600) # Argon
    mk_2 = thruster(VOLTAGE, WINDINGS, WIRE_LENGTH, B, RADIUS, species='N2', T_w=600)  # Diatomic Nitrogen
    mk_3 = thruster(VOLTAGE, WINDINGS, WIRE_LENGTH, B, RADIUS, species='Xe', T_w=600)  # Xenon

    # mk_1.circuit_param()

    # Solving the pulsed plasma (Te_0 = 1 eV, n_0 = 2.5e17 /m^3)
    vals_Ar = mk_1.solve_ode([2.5e17, 1], mk_1.ode, (0, Tau)) # Solve the ODE. Tau is the length of time we are solving the equation over
    vals_N2 = mk_2.solve_ode([2.5e17, 1], mk_2.ode, (0, Tau))
    vals_Xe = mk_3.solve_ode([2.5e17, 1], mk_3.ode, (0, Tau))


    # Get plasma number density, electron temperature, and time
    t_Ar, yAr, t_N2, yN2, t_Xe, yXe = vals_Ar.t, vals_Ar.y, vals_N2.t, vals_N2.y, vals_Xe.t, vals_Xe.y
    ne_Ar, Te_Ar = vals_Ar.y[0], vals_Ar.y[1] 
    ne_N2, Te_N2 = vals_N2.y[0], vals_N2.y[1]
    ne_Xe, Te_Xe = vals_Xe.y[0], vals_Xe.y[1]

    # Plot plasma number density versus time
    plt.figure(1)
    plt.title("Plasma Numer Density (1e18)$/m^3$")
    plt.plot(t_Ar, ne_Ar/1e18, label='Argon (Ar)') # Renormalizing to include plasma number density and electron temperature on the same plot
    plt.plot(t_N2, ne_N2/1e18, label='Diatomic Nitrogen (N2)') 
    plt.plot(t_Xe, ne_Xe/1e18, label="Xenon (Xe)")
    plt.xlabel("Time (seconds)")
    plt.legend()

    # Plot Electron Temperature versus time
    plt.figure(2)
    plt.title("Electron Temperature (K)")
    plt.plot(t_Ar, Te_Ar, label='Argon (Ar)') # Renormalizing to include plasma number density and electron temperature on the same plot
    plt.plot(t_N2, Te_N2, label='Diatomic Nitrogen (N2)') 
    plt.plot(t_Xe, Te_Xe, label="Xenon (Xe)")
    plt.xlabel("Time (seconds)")
    plt.legend()

    # Calculate the circuit parameters given the pulsed plasma number density from above, at an RF frequency of 2.5 MHz
    capacitor_vals_Ar = mk_1.pulsed_ns_circuit_params(ns=ne_Ar, frequency = 2.5e6)
    capacitor_vals_N2 = mk_1.pulsed_ns_circuit_params(ns=ne_N2, frequency = 2.5e6)
    capacitor_vals_Xe = mk_1.pulsed_ns_circuit_params(ns=ne_Xe, frequency = 2.5e6)

    err_Ar, err_N2, err_Xe = standard_error(capacitor_vals_Ar), standard_error(capacitor_vals_N2), standard_error(capacitor_vals_Xe)
    C_ar, C_n2, C_Xe = np.mean(capacitor_vals_Ar), np.mean(capacitor_vals_N2), np.mean(capacitor_vals_Xe)
    print("Ar: ", C_ar, "+/-", err_Ar, "\n")
    print("N2: ", C_n2, "+/-", err_N2, "\n")
    print("Xe: ", C_Xe, "+/-", err_Xe, "\n")
    
    # Plot said values
    fig3 = plt.figure(3)
    plt.plot(t_Ar, capacitor_vals_Ar, label="Argon (Ar)")
    plt.plot(t_N2, capacitor_vals_N2, label="Diatomic Nitrogen (N2)")
    plt.plot(t_Xe, capacitor_vals_Xe, label="Xenon (Xe)")
    plt.legend()
    plt.xlabel("Time (seconds)")
    plt.ylabel("Capacitance (F)")

    plt.show()