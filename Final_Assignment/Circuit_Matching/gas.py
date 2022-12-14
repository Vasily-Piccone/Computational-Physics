import numpy as np
from scipy.special import logsumexp  # Use this to fix the functions which are giving you Runtime Error
# This class defines gases of a single species
# gases = {'Ar': [], 'N2': [], 'Xe': []}


# Physical constants
kb = 1.380649e-23  # [J/kg*K] Boltzmann constant
ee = 1.6e-19  # [C] or [J/eV], electron charge constant
m_e = 9.1e-31  # [kg], electron mass


Pave = 500    #[W] Time averaged power absorbed by the plasma. Not nessisarily the RF power input but ideally it is.
pg_Torr = 5e-3           #[Torr] Gas pressure in the discharge in Torr
pg = pg_Torr*133.3       #[Pa] Gas pressure in the discharge. Only used to compute the number density for the example problem.
Tg = 600      # [K] Gas temperature

# Initial Calculations
ng = pg/(kb*Tg)    #[1/m^3] Gas number density in the discharge

class gas:
    # Initialize gas. T_w is the temperature of the plasma channel wall [K]
    def __init__(self, species: str, T_w):
        self.species = species
        if species == 'N2':
            # Set physical parameters and functions for the gas
            self.m_amu = 28  # [amu] Mass of the gas species in amu (N2)
            self.m_kg = self.m_amu/(6.022e+26)  # [kg] Mass of the gas species in kg
            self.e_loss_ion = 15.581  # [eV] Energy loss from ionization.
            self.e_loss_exc = 6.17  # [eV] Energy loss from excitation.

            # Diatomic Nitrogen rate equations
            self.k_iz = lambda Te: 7.76e-15*Te**(0.79)*np.exp(-16.75/Te)   # [m^3/s] Ionization rate coefficient for (N2).
            self.k_ex = lambda Te: 8.06e-16*Te**(-0.306)*np.exp(-8.87/Te)  # [m^3/s] First excitation mode (6.17eV) rate coefficient for (N2). 
            self.k_el = lambda Te: 1.04e-13*Te**(0.43)*np.exp(-0.206/Te)   # [m^3/s] Elastic collision rate coefficient for (N2).

        elif species == 'Ar':
            self.m_amu = 40  # [amu] Mass of the gas species in amu (Ar)
            self.m_kg = self.m_amu/(6.022e+26)  # [kg] Mass of the gas species in kg
            self.e_loss_ion = 15.76  # [eV] Energy loss from ionization.
            self.e_loss_exc = 12.14  # [eV] Energy loss from excitation.

            # Argon Rate equations
            self.k_iz = lambda Te: 2.34e-14*Te**0.59*np.exp(-17.44/Te)   # [m^3/s] Ionization rate coefficient for (Ar). 
            self.k_ex = lambda Te: 2.48e-14*Te**0.33*np.exp(-12.78/Te)   # [m^3/s] First excitation mode rate coefficient for (Ar).
            self.k_el = lambda Te: 2.336e-14*Te**1.609*np.exp(0.0618*(np.log(Te)**2) - 0.1171*(np.log(Te)**3))    # [m^3/s] Elastic collision rate coefficient for (Ar). 

        elif species == 'Xe':
            self.m_amu = 131.3 # [amu] Mass of the gas species in amu (Ar)
            self.m_kg = self.m_amu/(6.022e+26)  # [kg] Mass of the gas species in kg
            self.e_loss_ion = 12.1  # [eV] Energy loss from ionization.
            self.e_loss_exc = 10.2  # [eV] Energy loss from excitation.

            # Will the gamma require special error handling since the other species do not have a gamma?
            self.gamma = 1.66  # Ratio of specific heats of the gas 
            
            # Xenon Rate equations
            self.k_iz = lambda Te: np.heaviside(5-Te, 5-Te)*(1e-20*((3.97 + 0.643*Te - 0.0368*Te**2)*np.exp(-12.127/Te))*np.sqrt(8*ee*Te/(np.pi*m_e))) + np.heaviside(Te-5,Te-5)*(1e-20*(-1.031e-4*Te + 6.386*np.exp(-12.127/Te))*np.sqrt(8*ee*Te/(np.pi*m_e)));   # [m^3/s] Ionization rate coefficient for (Xe).   
            self.k_ex = lambda Te: 1.9310e-19*(np.exp(-11.6/Te)/np.sqrt(Te))*np.sqrt(8*ee*Te/(np.pi*m_e));   # [m^3/s] First excitation mode rate coefficient for (Xe).
            self.k_el = lambda Te: 0*Te;    # [m^3/s] Elastic collision rate coefficient for (Xe). Could not find, but has a small impact on results

        # Other important constants
        self.sigi = 1e-18  #  ion atom scattering cross section [m^2] -
        self.lambda_i = 1/(ng*self.sigi)  # 
        self.T_i = T_w/11594  # ion temperature [eV]
        self.Q_in = 8.28072e-16/np.sqrt(16*ee*self.T_i/np.pi/self.m_kg)  # ion-neutral collision cross section from Tsay 2005
        self.nu_in = np.sqrt(8*ee*self.T_i/np.pi/self.m_kg)*ng*self.Q_in  # ion-neutral collision frequency Tsay 2005
        self.D_i = ee*self.T_i/(self.m_kg*self.nu_in)  # ion diffusion coefficient from Tsay 2005

        # Functions needed for the ODE
        self.g = lambda Te: Te/self.T_i  # ratio between electron and ion temperature
        self.uB = lambda Te: np.sqrt(ee*Te/self.m_kg)  #  Bohm velocity [m/s]
        self.e_c = lambda Te: (1/self.k_iz(Te))*(self.k_iz(Te)*self.e_loss_ion + self.k_ex(Te)*self.e_loss_exc + self.k_el(Te)*(3*m_e/self.m_kg)*Te)  # Collisional energy lost per ion produced [eV]