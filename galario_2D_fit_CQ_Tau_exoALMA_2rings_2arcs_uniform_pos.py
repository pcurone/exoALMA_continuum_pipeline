##########################################################################
import numpy as np
from numba import njit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from scipy import ndimage
import os
import time

from emcee import EnsembleSampler
import corner
import pickle
from scipy import signal

# Convert from arcsec and deg to radians
from galario import arcsec, deg
from galario.double import chi2Image, sampleImage, get_image_size
from uvplot import COLUMNS_V0

from uvplot import UVTable
##########################################################################

target = 'CQ_Tau'

##################################
#####   IMPORT THE UVTABLE   #####
##################################

print('Importing uvtable')
uvtable_filename = target+'_galario_uvtable.txt'
u, v, Re, Im, w = np.require(np.loadtxt(uvtable_filename, unpack=True), requirements='C')
wle = 0.0009042042814161718  # in meters, read from the second line in the uvtable.txt
u /= wle
v /= wle

########################################
#####   DETERMINE THE IMAGE SIZE   #####
########################################

# compute the optimal image size (nxy=number of pixels, dxy=pixel size in radians)
nxy, dxy = get_image_size(u, v, verbose=True, f_min=5., f_max=2.5)

# define the image grid in radians
xarr = np.linspace(-nxy/2*dxy, nxy/2*dxy, nxy)
yarr = np.linspace(-nxy/2*dxy, nxy/2*dxy, nxy)
x,y = np.meshgrid(xarr, yarr)

#############################################
#####   DEFINE THE 2D MODEL FUNCTIONS   #####
#############################################

# define a Gaussian ring in 2D on the image array (with numba to make it faster)
@njit(fastmath=True, parallel=True)
def ring_2D(I0,R0,width,inc_rad):
    r = ((x/np.cos(inc_rad))**2.+(y)**2.)**0.5 
    imagemap = I0 * np.exp(-((r-R0)**2.)/(2.*(width)**2.))
    return imagemap


# Define a 2D 'Gaussian' arc on the image array (with numba to make it faster)
'''
2D arc defined as a Gaussian 2D ring with an exponential cutoff on the azimuthal coordinate

theta0 = angle in deg of the central point of the arc
defined with same convention of the PA, 0=up, 90=left, 180=down, 270=right

theta_width = angle in deg of the azimuthal extension of the arc,
or better, angle after which there is the exponential cutoff
'''
@njit(fastmath=True, parallel=True)
def arc_2D(I0,R0,R_width, theta0_rad, theta_width_rad,inc_rad):
    r = ((x/np.cos(inc_rad))**2.+(y)**2.)**0.5 
    theta = np.arctan2(y,x/np.cos(inc_rad)) - theta0_rad
    for j in range(nxy):
        for k in range(nxy):
            if theta[j,k] >  np.pi:
                theta[j,k] = 2*np.pi - theta[j,k]
            if theta[j,k] <  -np.pi:
                theta[j,k] = -2*np.pi - theta[j,k]
    imagemap = I0 * np.exp(-((r-R0)**2.)/(2.*(R_width)**2.)) * np.exp(-(theta)**2./(2.*theta_width_rad**2.)) 
    return imagemap


# define the image model as sum of two rings and two arcs (with numba to make it faster)
@njit(fastmath=True, parallel=True)
def Image2D_2rings_2arcs(f1,r1,sigma1,   f2,r2,sigma2,   f_arc1,r_arc1,sigmar_arc1,th_arc1,sigmath_arc1,  f_arc2,r_arc2,sigmar_arc2,th_arc2,sigmath_arc2,   inc_rad):
    model_image = ring_2D(f1,r1,sigma1,inc_rad) + ring_2D(f2,r2,sigma2,inc_rad) + arc_2D(f_arc1,r_arc1,sigmar_arc1, th_arc1, sigmath_arc1,inc_rad) + arc_2D(f_arc2,r_arc2,sigmar_arc2, th_arc2, sigmath_arc2,inc_rad) 
    return model_image


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

def lnpostfn_2D(p, p_ranges, dxy, u, v, Re, Im, w):
    """ Log of posterior probability function """

    lnprior = lnpriorfn(p, p_ranges)  # apply prior
    if not np.isfinite(lnprior):
        return -np.inf

    # unpack the parameters
    f1,r1,sigma1,   f2,r2,sigma2,   f_arc1,r_arc1,sigmar_arc1,th_arc1,sigmath_arc1,  f_arc2,r_arc2,sigmar_arc2,th_arc2,sigmath_arc2,   inc, PA, dRA, dDec = p
    
    f1 = (10.0**f1) * dxy**2        # convert from log10[Jy/sr] to Jy/pixel
    f2 = (10.0**f2) * dxy**2
    f_arc1 = (10.**f_arc1) * dxy**2
    f_arc2 = (10.**f_arc2) * dxy**2
    
    # convert to radians
    r1 *= arcsec
    r2 *= arcsec
    r_arc1 *= arcsec
    r_arc2 *= arcsec
    sigma1 *= arcsec
    sigma2 *= arcsec
    sigmar_arc1 *= arcsec
    sigmar_arc2 *= arcsec

    th_arc1 = (90+th_arc1) * deg
    th_arc2 = (90+th_arc2) * deg
    sigmath_arc1 *= deg
    sigmath_arc2 *= deg

    inc *= deg
    PA *= deg
    dRA *= arcsec
    dDec *= arcsec

    # compute the model brightness profile
    f = Image2D_2rings_2arcs(f1,r1,sigma1,   f2,r2,sigma2,   f_arc1,r_arc1,sigmar_arc1,th_arc1,sigmath_arc1,  f_arc2,r_arc2,sigmar_arc2,th_arc2,sigmath_arc2,   inc)

    chi2_2D = chi2Image(f, dxy, u, v, Re, Im, w,
                     PA=PA, dRA=dRA, dDec=dDec, origin='lower')

    return -0.5 * chi2_2D + lnprior


###############################
#####   SETUP THE MCMC    #####
###############################


# initial guess for the parameters
# f1(log10[Jy/sr]), r1(arcsec), sigma1(arcsec),
# f2(log10[Jy/sr]), r2(arcsec), sigma2(arcsec),
# f_arc1(log10[Jy/sr]), r_arc1 (arcsec), sigmar_arc1(arcsec), th_arc1(deg), sigmath_arc1(deg),
# f_arc2(log10[Jy/sr]), r_arc2 (arcsec), sigmar_arc2(arcsec), th_arc2(deg), sigmath_arc2(deg),
# inc(deg), PA(deg), dRA(arcsec), dDec(arcsec)
p0 = [10.55, 0.28, 0.1,     9.27, 0.48, 0.18,   10.4, 0.3, 0.08, 10, 40,     10.4, 0.3, 0.08, 160, 30,    35.0, 55.0, -0.0056, -0.0053] #  
# parameter space domain
p_ranges = [[1., 20.],
            [0., 1.0], 
            [0., 1.0], 

            [1., 20.],
            [0., 1.0], 
            [0., 1.0], 

            [1., 20.],
            [0., 1.0], 
            [0., 1.0],
            [0., 360],
            [0., 360],   

            [1., 20.],
            [0., 1.0], 
            [0., 1.0],
            [0., 360],
            [0., 360],   

            [0., 90.],
            [0., 180.],
            [-2, 2], 
            [-2, 2]]

ndim = len(p_ranges)

nthreads = 30
nwalkers = 50
nsteps = 1000


# initialize the walkers in a uniform interval for a broader exploration of the parameter space
pos = []
for i in range(nwalkers):
    pos.append(np.array([np.random.uniform(low=9, high=15),
                         np.random.uniform(low=0.2, high=0.45),
                         np.random.uniform(low=0., high=0.2),

                         np.random.uniform(low=5, high=11),
                         np.random.uniform(low=0., high=0.8),
                         np.random.uniform(low=0.1, high=1.0),

                         np.random.uniform(low=5, high=15),
                         np.random.uniform(low=0.2, high=0.45),
                         np.random.uniform(low=0., high=0.2),
                         np.random.uniform(low=0., high=90),
                         np.random.uniform(low=0., high=120),

                         np.random.uniform(low=5, high=15),
                         np.random.uniform(low=0.2, high=0.45),
                         np.random.uniform(low=0., high=0.2),
                         np.random.uniform(low=90., high=270),
                         np.random.uniform(low=0., high=120),

                         np.random.uniform(low=0., high=45),
                         np.random.uniform(low=0., high=180),
                         np.random.uniform(low=-0.1, high=0.1),
                         np.random.uniform(low=-0.1, high=0.1),
                         ]))

sampler = EnsembleSampler(nwalkers, ndim, lnpostfn_2D, args=[p_ranges, dxy, u, v, Re, Im, w], threads=nthreads)

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


pickle.dump(results_galario_fullrun, open('galario_fullrun_Image2D_2rings_2arcs_'+str(int(nwalkers))+'walker_'+str(int(nsteps))+'steps.pkl', 'wb'))


###########################
#####   CORNER PLOT   #####
###########################

fig = corner.corner(samples, labels=["$f_1$",  r"$r_1$", r"$\sigma_1$",       "$f_2$",  r"$r_2$", r"$\sigma_2$",         "$f_{\mathrm{Arc}1}$",  r"$r_{\mathrm{Arc}1}$", r"$\sigma_{\mathrm{Arc}1}$", r"$\theta_{\mathrm{Arc}1}$", r"$\sigma_{\theta\mathrm{Arc}1}$",     "$f_{\mathrm{Arc}2}$",  r"$r_{\mathrm{Arc}2}$", r"$\sigma_{\mathrm{Arc}2}$", r"$\theta_{\mathrm{Arc}2}$", r"$\sigma_{\theta\mathrm{Arc}2}$",     "inc", "PA", r"$\Delta RA$", r"$\Delta Dec$"],
                    show_titles=True, quantiles=[0.16, 0.50, 0.84],
                    label_kwargs={'labelpad':20, 'fontsize':15}, fontsize=8, title_fmt = '.5f')

#plt.show()
plt.savefig('Cornerplot_galario_Image2D_2rings_2arcs_'+str(int(nwalkers))+'walker_'+str(int(nsteps))+'steps.pdf', bbox_inches='tight')


#####################################
#####   IMAGE OF THE 2D MODEL   #####
#####################################

bestfit = [np.percentile(samples[:, i], 50) for i in range(ndim)]

f1,r1,sigma1,   f2,r2,sigma2,   f_arc1,r_arc1,sigmar_arc1,th_arc1,sigmath_arc1,  f_arc2,r_arc2,sigmar_arc2,th_arc2,sigmath_arc2,   inc, PA, dRA, dDec = bestfit

f1 = (10.0**f1) * dxy**2        # convert from log10[Jy/sr] to Jy/pixel
f2 = (10.0**f2) * dxy**2
f_arc1 = (10.**f_arc1) * dxy**2
f_arc2 = (10.**f_arc2) * dxy**2

# convert to radians
r1 *= arcsec
r2 *= arcsec
r_arc1 *= arcsec
r_arc2 *= arcsec
sigma1 *= arcsec
sigma2 *= arcsec
sigmar_arc1 *= arcsec
sigmar_arc2 *= arcsec

th_arc1 = (90+th_arc1) * deg
th_arc2 = (90+th_arc2) * deg
sigmath_arc1 *= deg
sigmath_arc2 *= deg

inc *= deg
PA *= deg
dRA *= arcsec
dDec *= arcsec

# compute the model brightness image
f = Image2D_2rings_2arcs(f1,r1,sigma1,   f2,r2,sigma2,   f_arc1,r_arc1,sigmar_arc1,th_arc1,sigmath_arc1,  f_arc2,r_arc2,sigmar_arc2,th_arc2,sigmath_arc2,   inc)


# Plot the image
fig= plt.figure(figsize=(7,5.9))
ax =fig.add_subplot(111)

rotate_img = ndimage.rotate(f, -PA/deg, reshape=False)
plt.imshow(rotate_img, cmap = 'inferno', extent=[nxy/2*dxy/arcsec, -nxy/2*dxy/arcsec, -nxy/2*dxy/arcsec, nxy/2*dxy/arcsec],origin='lower')

cbar = plt.colorbar(pad=0.02)
cbar.outline.set_linewidth(2)
cbar.ax.tick_params(which='major', labelsize=14,width=2.5, length=6,direction='in')
cbar.ax.tick_params(which='minor', labelsize=14,width=1.5, length=4,direction='in')
cbar.set_label('Intensity [Jy/pixel]', fontsize = 17, labelpad=26)
cbar.ax.minorticks_on()

lim = 1.2
ax.set_xlim(lim,-lim)
ax.set_ylim(-lim,lim)

index_ticks = 0.5
ax.xaxis.set_major_locator(MultipleLocator(index_ticks))
ax.xaxis.set_minor_locator(MultipleLocator(index_ticks/4))
ax.yaxis.set_major_locator(MultipleLocator(index_ticks))
ax.yaxis.set_minor_locator(MultipleLocator(index_ticks/4)) 
ax.tick_params(which='major',axis='both',right=True,top=True, labelsize=14, pad=5,width=2.5, length=6,direction='in',color='w')
ax.tick_params(which='minor',axis='both',right=True,top=True, labelsize=14, pad=5,width=1.5, length=4,direction='in',color='w')
ax.set_xlabel('RA offset  ($^{\prime\prime}$)', fontsize = 17, labelpad=10)
ax.set_ylabel('Dec offset  ($^{\prime\prime}$)', fontsize = 17, labelpad=10)

for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(1)
    
plt.savefig('Model_Image2D_2rings_2arcs_'+str(int(nwalkers))+'walker_'+str(int(nsteps))+'steps.pdf', bbox_inches='tight')  


#######################
#####   UV-PLOT   #####
#######################

# compute the visibilities of the bestfit model
vis_mod = sampleImage(f, dxy, u, v, PA=PA, dRA=dRA, dDec=dDec, origin='lower')

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
plt.savefig('Radialprofile_galario_Image2D_2rings_2arcs_'+str(int(nwalkers))+'walker_'+str(int(nsteps))+'steps.pdf', bbox_inches='tight')           



##################################
#####   CHECK CONVERGENCE    #####
##################################

fig, axes = plt.subplots(ndim, figsize=(12, 20), sharex=True)
labels=["$f_1$",  r"$r_1$", r"$\sigma_1$",       "$f_2$",  r"$r_2$", r"$\sigma_2$",         "$f_{\mathrm{Arc}1}$",  r"$r_{\mathrm{Arc}1}$", r"$\sigma_{\mathrm{Arc}1}$", r"$\theta_{\mathrm{Arc}1}$", r"$\sigma_{\theta\mathrm{Arc}1}$",     "$f_{\mathrm{Arc}2}$",  r"$r_{\mathrm{Arc}2}$", r"$\sigma_{\mathrm{Arc}2}$", r"$\theta_{\mathrm{Arc}2}$", r"$\sigma_{\theta\mathrm{Arc}2}$",     "inc", "PA", r"$\Delta RA$", r"$\Delta Dec$"]
samples_tot = sampler.get_chain()
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples_tot[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples_tot))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
plt.savefig('Walkers_path_galario_Image2D_2rings_2arcs_'+str(int(nwalkers))+'walker_'+str(int(nsteps))+'steps.pdf', bbox_inches='tight')           


# Check auto-correlation time
tau = sampler.get_autocorr_time()

f = open('autocorr_time.txt', 'w')
f.write(f'Autocorrelation time estimated with sampler.get_autocorr_time()\n{tau}')
f.close()
