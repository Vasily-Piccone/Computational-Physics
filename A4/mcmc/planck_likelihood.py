import numpy as np
import camb
from matplotlib import pyplot as plt
import time
import os

dirname = os.path.dirname(__file__)

def get_spectrum(pars,lmax=3000):
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau) # determining universe-type based on the model
    pars.InitPower.set_params(As=As,ns=ns,r=0) # creating the data from the parameters 
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt[2:]


plt.ion()

planck_1 = os.path.join(dirname, './COM_PowerSpect_CMB-TT-full_R3.01.txt')
pars=np.asarray([69, 0.022, 0.12, 0.06, 2.1e-9, 0.95])
planck=np.loadtxt(planck_1, skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
print(len(spec))
errs=0.5*(planck[:,2]+planck[:,3])
model=get_spectrum(pars)
model=model[:len(spec)]
print(model)
resid=spec-model
chisq=np.sum( (resid/errs)**2)
print("chisq is ",chisq," for ",len(resid)-len(pars)," degrees of freedom.")
# Why doesn't the code run after this line?
#read in a binned version of the Planck PS for plotting purposes

planck_2 = os.path.join(dirname, './COM_PowerSpect_CMB-TT-binned_R3.01.txt')
planck_binned=np.loadtxt(planck_2,skiprows=1)
errs_binned=0.5*(planck_binned[:,2]+planck_binned[:,3])
plt.plot(ell,model)
plt.errorbar(planck_binned[:,0],planck_binned[:,1],errs_binned,fmt='.')
plt.savefig("beans.png")
