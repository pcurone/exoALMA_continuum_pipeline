##########################################################################
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

from emcee import EnsembleSampler
import corner
import pickle
from scipy import signal

# Convert from arcsec and deg to radians
from galario import arcsec, deg
from galario.double import chi2Profile, sampleProfile, get_image_size
from uvplot import COLUMNS_V0

from uvplot import UVTable
##########################################################################

target = 'AA_Tau'

##################################
#####   IMPORT THE UVTABLE   #####
##################################

print('Importing uvtable')
uvtable_filename = target+'_galario_uvtable.txt'
u, v, Re, Im, w = np.require(np.loadtxt(uvtable_filename, unpack=True), requirements='C')
wle = 0.0009041583162952121  # in meters, read from the second line in the uvtable.txt
u /= wle
v /= wle


##########################################
#####   DEFINE THE MODEL FUNCTIONS   #####
##########################################


def FourRingProfile_PointSource(f0, f1, r1, sigma1, f2, r2, sigma2, f3, r3, sigma3, f4, r4, sigma4, Rmin, dR, nR):

    # radial grid
    R = np.linspace(Rmin, Rmin + dR*nR, nR, endpoint=False)

    return f0 * signal.unit_impulse(nR) + f1 * np.exp(-(R-r1)**2./(2.*sigma1**2.)) + f2 * np.exp(-(R-r2)**2./(2.*sigma2**2.)) + f3 * np.exp(-(R-r3)**2./(2.*sigma3**2.)) + f4 * np.exp(-(R-r4)**2./(2.*sigma4**2.))



#####################################################
#####   DEFINE THE PRIOR PROBABILITY FUNCTION   #####
#####################################################

'''
Log of uniform prior probability function
'''
def lnpriorfn(p, par_ranges):
    for i in range(len(p)):
        if p[i] < par_ranges[i][0] or p[i] > par_ranges[i][1]:
            return -np.inf

    return 0.0


#########################################################
#####   DEFINE THE POSTERIOR PROBABILITY FUNCTION   #####
#########################################################

def lnpostfn(p, p_ranges, Rmin, dR, nR, nxy, dxy, u, v, Re, Im, w):
    """ Log of posterior probability function """

    lnprior = lnpriorfn(p, p_ranges)  # apply prior
    if not np.isfinite(lnprior):
        return -np.inf

    # unpack the parameters
    f0, f1, r1, sigma1, f2, r2, sigma2, f3, r3, sigma3, f4, r4, sigma4, inc, PA, dRA, dDec = p
    
    f0 = 10.**f0        # convert from log to real space
    f1 = 10.**f1
    f2 = 10.**f2
    f3 = 10.**f3 
    f4 = 10.**f4 
    
    # convert to radians
    sigma1 *= arcsec
    sigma2 *= arcsec
    sigma3 *= arcsec
    sigma4 *= arcsec
    r1 *= arcsec
    r2 *= arcsec
    r3 *= arcsec
    r4 *= arcsec
    Rmin *= arcsec
    dR *= arcsec
    inc *= deg
    PA *= deg
    dRA *= arcsec
    dDec *= arcsec

    # compute the model brightness profile
    f = FourRingProfile_PointSource(f0, f1, r1, sigma1, f2, r2, sigma2, f3, r3, sigma3, f4, r4, sigma4, Rmin, dR, nR)

    chi2 = chi2Profile(f, Rmin, dR, nxy, dxy, u, v, Re, Im, w,
                                inc=inc, PA=PA, dRA=dRA, dDec=dDec)

    #print(np.min(f),np.max(f),np.isnan(f.any()),chi2)

    return -0.5 * chi2 + lnprior



########################################
#####   DETERMINE THE IMAGE SIZE   #####
########################################

# compute the optimal image size
nxy, dxy = get_image_size(u, v, verbose=True, f_min=5., f_max=2.5)


###############################
#####   SETUP THE MCMC    #####
###############################

# radial grid parameters
Rmin = 1e-4  # arcsec
dR = 0.0001    # arcsec
nR = 40000


# f1, r1, sigma1, inc, PA, dRA, dDec
# initial guess for the parameters
p0 = [15.5,   10.06, 0.35, 0.054,     9.97, 0.28, 0.037,   9.65, 0.66, 0.02,    9.5, 0.42, 0.39,   58.52, 93.8, -0.0055, 0.005] #  
# parameter space domain
p_ranges = [[1., 20.],
            
            [1., 15.],
            [0., 0.6], 
            [0., 0.8], 

            [1., 15.],
            [0., 1.0], 
            [0., 1.0], 

            [1., 15.],
            [0., 1.0], 
            [0., 1.0], 

            [1., 15.],
            [0., 1.5], 
            [0.0, 1.5],

            [0., 90.],
            [0., 180.],
            [-2, 2], 
            [-2, 2]]

ndim = len(p_ranges)

nthreads = 30
nwalkers = 100
nsteps = 5001

# initialize the walkers with an ndim-dimensional Gaussian ball
pos = [p0 + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

sampler = EnsembleSampler(nwalkers, ndim, lnpostfn, args=[p_ranges, Rmin, dR, nR, nxy, dxy, u, v, Re, Im, w], threads=nthreads)


#############################
#####   RUN THE MCMC    #####
#############################

# execute the MCMC
print('Walking...')
pos, prob, state = sampler.run_mcmc(pos, nsteps, progress=True)


################################
#####   SAVE THE RESULTS   #####
################################

samples = sampler.chain[:, -1000:, :].reshape((-1, ndim))

results_galario_fullrun = {'emcee_sampler_samples' : samples,
                           'model_Rmin' : Rmin,
                           'model_dR' : dR,
                           'model_nR' : nR,
                           'model_p_ranges' : p_ranges,
                           'model_p0' : p0,
                           'model_nwalkers' : nwalkers,
                           'model_nsteps' : nsteps,
                           'uvdata_u' : u,
                           'uvdata_v' : v,
                           'uvdata_Re' : Re,
                           'uvdata_Im' : Im,
                           'uvdata_w' : w,
                           'wavelength' : wle
                            }


pickle.dump(results_galario_fullrun, open('galario_fullrun_FourRingProfile_PointSource_'+str(int(nwalkers))+'walker_'+str(int(nsteps))+'steps.pkl', 'wb'))


###########################
#####   CORNER PLOT   #####
###########################

fig = corner.corner(samples, labels=["$f_0$", "$f_1$",  r"$r_1$", r"$\sigma_1$", "$f_2$",  r"$r_2$", r"$\sigma_2$", "$f_3$",  r"$r_3$", r"$\sigma_3$", "$f_4$",  r"$r_4$", r"$\sigma_4$", "inc", "PA", r"$\Delta RA$", r"$\Delta Dec$"],
                    show_titles=True, quantiles=[0.16, 0.50, 0.84],
                    label_kwargs={'labelpad':20, 'fontsize':15}, fontsize=8, title_fmt = '.5f')

#plt.show()
plt.savefig('Cornerplot_galario_FourRingProfile_PointSource_'+str(int(nwalkers))+'walker_'+str(int(nsteps))+'steps.pdf', bbox_inches='tight')


#######################
#####   UV-PLOT   #####
#######################

bestfit = [np.percentile(samples[:, i], 50) for i in range(ndim)]

f0, f1, r1, sigma1, f2, r2, sigma2, f3, r3, sigma3, f4, r4, sigma4, inc, PA, dRA, dDec = bestfit

f0 = 10.**f0        # convert from log to real space
f1 = 10.**f1        
f2 = 10.**f2 
f3 = 10.**f3 
f4 = 10.**f4 

# convert to radians
sigma1 *= arcsec
sigma2 *= arcsec
sigma3 *= arcsec
sigma4 *= arcsec
r1 *= arcsec
r2 *= arcsec
r3 *= arcsec
r4 *= arcsec
Rmin *= arcsec
dR *= arcsec
inc *= deg
PA *= deg
dRA *= arcsec
dDec *= arcsec

# compute the model brightness profile
f = FourRingProfile_PointSource(f0, f1, r1, sigma1, f2, r2, sigma2, f3, r3, sigma3, f4, r4, sigma4, Rmin, dR, nR)


# compute the visibilities of the bestfit model
vis_mod = sampleProfile(f, Rmin, dR, nxy, dxy, u, v,
                                 inc=inc, PA=PA, dRA=dRA, dDec=dDec)

uvbin_size = 20e3     # uv-distance bin, units: wle

# observations uv-plot
uv = UVTable(uvtable=[u*wle, v*wle, Re, Im, w], wle=wle, columns=COLUMNS_V0)
uv.apply_phase(-dRA, -dDec)         # center the source on the phase center
uv.deproject(inc, PA)
axes = uv.plot(linestyle='.', color='k', label='Data', uvbin_size=uvbin_size, linewidth=1, alpha=0.8)

# model uv-plot
uv_mod = UVTable(uvtable=[u*wle, v*wle, vis_mod.real, vis_mod.imag, w], wle=wle, columns=COLUMNS_V0)
uv_mod.apply_phase(-dRA, -dDec)     # center the source on the phase center
uv_mod.deproject(inc, PA)
uv_mod.plot(axes=axes, linestyle='-', color='r', label='Model', yerr=False, uvbin_size=uvbin_size)

axes[0].hlines(0,0,3000, linewidth=1, alpha=0.5, color='gray', linestyle='dashed')
axes[1].hlines(0,0,3000, linewidth=1, alpha=0.5, color='gray', linestyle='dashed')


axes[0].legend(fontsize=18, frameon=True)
for side in axes[0].spines.keys():  # 'top', 'bottom', 'left', 'right'
    axes[0].spines[side].set_linewidth(3)

for side in axes[1].spines.keys():  # 'top', 'bottom', 'left', 'right'
    axes[1].spines[side].set_linewidth(3)
axes[0].tick_params(which='both',right=True,top=True, width=3, length=6,labelsize=14, direction='in',pad=5)
axes[1].tick_params(which='both',right=True,top=True, width=3, length=6,labelsize=14, direction='in',pad=5)

#plt.show()
plt.savefig('Radialprofile_galario_FourRingProfile_PointSource_'+str(int(nwalkers))+'walker_'+str(int(nsteps))+'steps.pdf', bbox_inches='tight')           



##################################
#####   CHECK CONVERGENCE    #####
##################################

fig, axes = plt.subplots(ndim, figsize=(12, 20), sharex=True)
labels=["$f_0$", "$f_1$",  r"$r_1$", r"$\sigma_1$", "$f_2$",  r"$r_2$", r"$\sigma_2$", "$f_3$",  r"$r_3$", r"$\sigma_3$", "$f_4$",  r"$r_4$", r"$\sigma_4$", "inc", "PA", r"$\Delta RA$", r"$\Delta Dec$"]
samples_tot = sampler.get_chain()
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples_tot[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples_tot))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
plt.savefig('Walkers_path_galario_FourRingProfile_PointSource_'+str(int(nwalkers))+'walker_'+str(int(nsteps))+'steps.pdf', bbox_inches='tight')           


# Check auto-correlation time
tau = sampler.get_autocorr_time()

f = open('autocorr_time.txt', 'w')
f.write(f'Autocorrelation time estimated with sampler.get_autocorr_time()\n{tau}')
f.close()
