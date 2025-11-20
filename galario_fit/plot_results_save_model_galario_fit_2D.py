##########################################################################
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from scipy import ndimage
import os
import time

from emcee import EnsembleSampler, autocorr
from emcee.backends import HDFBackend
import corner

import galario
# Convert from arcsec and deg to radians
from galario import arcsec, deg
# Use GPU if available, otherwise fall back to CPU
if galario.HAVE_CUDA:
    from galario.double_cuda import chi2Image, sampleImage, get_image_size
else:
    from galario.double import chi2Image, sampleImage, get_image_size
  
from uvplot import COLUMNS_V0
from uvplot import UVTable

from galario_2D_model_functions import model_registry
##########################################################################


##################### MODIFY HERE #####################
target = 'CQ_Tau'

uvtable_filename = f'../data/{target}_galario_uvtable.txt'

# Select model
selected_model = "2rings_2arcs"

nwalkers = 100


##################################
#####   IMPORT THE UVTABLE   #####
##################################

# Start the timer
start_time = time.time()

print('Importing uvtable')
wle = np.loadtxt(uvtable_filename, skiprows=2, max_rows=1)
u, v, Re, Im, w = np.require(np.loadtxt(uvtable_filename, skiprows=4, unpack=True), requirements='C')


########################################
#####   DETERMINE THE IMAGE SIZE   #####
########################################

# compute the optimal image size (nxy=number of pixels, dxy=pixel size in radians)
nxy, dxy = get_image_size(u, v, verbose=True, f_min=5., f_max=2.5)

# define the image grid in radians
xarr = np.linspace(-nxy/2*dxy, nxy/2*dxy, nxy)
yarr = np.linspace(-nxy/2*dxy, nxy/2*dxy, nxy)
x,y = np.meshgrid(xarr, yarr)


###############################
#####   SETUP THE MCMC    #####
###############################

model_info = model_registry[selected_model]
model_function = model_info["function"]
base_parameters = model_info["parameters"]
param_labels = model_info["labels"]
params = [param for param in base_parameters]
params.extend(["inc", "PA", "dRA", "dDec"])  
ndim = len(params)

# Backend setup and initialization
backend_filename = f"{target}_galario_2D_emcee_backend_{selected_model}_{nwalkers}walkers.h5"
backend = HDFBackend(backend_filename)


####################################
#####   READ RESULTING CHAIN   #####
####################################

flat_chain = backend.get_chain(flat=True)
niters = 1000

# Extract the last nsamples samples
if len(flat_chain) >= niters*nwalkers:
    print(f"Selecting only walker's positions in the last {niters} iterations")
    samples = flat_chain[-niters*nwalkers:]
else:
    print(F"Warning: The chain contains fewer iterations than the selected niters ({niters}).")
    samples = flat_chain  # Use all available samples

# Print or analyze the last nsamples samples
#print("Shape of last nsamples  samples:", samples.shape)

###########################
#####   CORNER PLOT   #####
###########################

fig = corner.corner(samples, labels=param_labels,
                    show_titles=True, quantiles=[0.16, 0.50, 0.84],
                    label_kwargs={'labelpad':20, 'fontsize':15}, fontsize=8, title_fmt = '.5f')

#plt.show()
plt.savefig(f'products/{target}_cornerplot_galario_2D_{selected_model}_{nwalkers}walkers_{backend.iteration}iters.pdf', bbox_inches='tight')


#####################################
#####   IMAGE OF THE 2D MODEL   #####
#####################################

bestfit = [np.percentile(samples[:, i], 50) for i in range(ndim)]

param_dict = {key: val for key, val in zip(params, bestfit)}

# Convert specific parameters to required units
for key in ["sigma0", "sigma1", "sigma2", "sigma3", "sigma4", "sigma5", "sigma6", 
            "r1", "r2", "r3", "r4", "r5", "r6", 
            "r_arc1", "r_arc2", "r_arc3", "r_arc4", "r_arc5", "r_arc6",
            "sigmar_arc1", "sigmar_arc2", "sigmar_arc3", "sigmar_arc4", "sigmar_arc5", "sigmar_arc6", 
            "dRA", "dDec"]:
    if key in param_dict:
        param_dict[key] *= arcsec

for key in ["sigmaphi_arc1", "sigmaphi_arc2", "sigmaphi_arc3", "sigmaphi_arc4", "sigmaphi_arc5", "sigmaphi_arc6",
            "inc", "PA"]:
    if key in param_dict:
        param_dict[key] *= deg

for key in ["phi_arc1", "phi_arc2", "phi_arc3", "phi_arc4", "phi_arc5", "phi_arc6"]:
    if key in param_dict:
        param_dict[key] = (90+param_dict[key]) * deg

for key in ["f0", "f1", "f2", "f3", "f4", "f5", "f6",
            "f_arc1", "f_arc2", "f_arc3", "f_arc4", "f_arc5", "f_arc6"]:
    if key in param_dict:
        param_dict[key] = 10.0**param_dict[key] * dxy**2

# Extract the necessary parameters for chi2Image
inc = param_dict["inc"]
PA = param_dict["PA"]
dRA = param_dict["dRA"]
dDec = param_dict["dDec"]
funtion_param_dict = {key: val for key, val in param_dict.items() if key not in ["inc", "PA", "dRA", "dDec"]}
funtion_param_dict["inc_rad"] = inc

# Compute the model brightness profile
f = model_function(x,y,**funtion_param_dict)


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
ax.xaxis.set_minor_locator(MultipleLocator(index_ticks/5))
ax.yaxis.set_major_locator(MultipleLocator(index_ticks))
ax.yaxis.set_minor_locator(MultipleLocator(index_ticks/5)) 
ax.tick_params(which='major',axis='both',right=True,top=True, labelsize=14, pad=5,width=2.5, length=6,direction='in',color='w')
ax.tick_params(which='minor',axis='both',right=True,top=True, labelsize=14, pad=5,width=1.5, length=4,direction='in',color='w')
ax.set_xlabel('RA offset  ($^{\prime\prime}$)', fontsize = 17, labelpad=10)
ax.set_ylabel('Dec offset  ($^{\prime\prime}$)', fontsize = 17, labelpad=10)

for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(1)
    
plt.savefig(f'products/{target}_model_Image2D_{selected_model}_{nwalkers}walkers_{backend.iteration}iters.pdf', bbox_inches='tight')  



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
plt.savefig(f'products/{target}_radialprofile_galario_2D_{selected_model}_{nwalkers}walkers_{backend.iteration}iters.pdf', bbox_inches='tight')


########################################################
#####   GENERATE ASCII UVTABLE OF GALARIO MODEL    #####
########################################################

print('Get uvtable with galario model')

# Create the formatted header manually
header = (
    f"# Visbilities obtained from the galario model {selected_model} with {nwalkers} walkers and {backend.iteration} total iterations\n"
    f"# Columns:\tu[wavelength]\tv[wavelength]\tRe(V)[Jy]\tIm(V)[Jy]\tweight"
)
# Save the data
np.savetxt(
    f"products/{target}_uvtable_galario_model_{selected_model}_{nwalkers}walkers_{backend.iteration}iters.txt",
    np.column_stack([u, v, vis_mod.real, vis_mod.imag, w]),
    fmt='%10.6e',
    delimiter='\t',
    header=header,
    comments=""  # Removes the default '#' added by savetxt to the header
)


##################################
#####   CHECK CONVERGENCE    #####
##################################

fig, axes = plt.subplots(ndim, figsize=(12, 20), sharex=True)
labels=param_labels
nonflat_chain = backend.get_chain()
for i in range(ndim):
    ax = axes[i]
    ax.plot(nonflat_chain[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(nonflat_chain))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
plt.savefig(f'products/{target}_walker_path_galario_{selected_model}_{nwalkers}walkers_{backend.iteration}iters.pdf', bbox_inches='tight')

# Calculate autocorrelation time and check chain length
try:
    tau = autocorr.integrated_time(nonflat_chain, tol=0)
    print(f"Autocorrelation time for each parameter: {tau}")
    print(f"Mean autocorrelation time: {np.mean(tau)}")

    # Check if the chain is sufficiently long
    nsteps = nonflat_chain.shape[0]  # Number of steps in the chain
    if nsteps < 50 * np.max(tau):
        print("Warning: Chain is too short for reliable results. "
              f"Chain length is {nsteps}, but should be at least {50 * np.max(tau):.0f} "
              "to reliably estimate autocorrelation time.")
except Exception as e:
    print(f"Error calculating autocorrelation time: {e}")


# End the timer
end_time = time.time()
elapsed_time = (end_time - start_time) / 60  # Convert seconds to minutes

# Print runtime and number of walkers
print(f"Run completed in {elapsed_time:.2f} minutes with {nwalkers} walkers.")
