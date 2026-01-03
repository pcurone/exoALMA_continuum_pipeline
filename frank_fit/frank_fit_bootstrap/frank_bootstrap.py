# Script to run many times the frank fit varying the geometrical parameters in order to get a reasonable estimate of the uncertainty of the frank fit

import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from astropy.io import fits
from astropy.visualization import (AsinhStretch, LinearStretch, ImageNormalize)
import scipy
from tqdm import tqdm

import frank
from frank.radial_fitters import FrankFitter
from frank.geometry import FixedGeometry
from frank.utilities import convolve_profile, sweep_profile
from frank.io import save_fit, load_sol
from frank.make_figs import make_full_fig
sys.path.append('..')
import diskdictionary as disk

#frank.enable_logging()

# controls
target = 'AA_Tau'

robust = 0.5    # robust value of your fiducial image
n_iter = 10     # number of bootstrap iterations

# Get major axis of synthesized beam value
dhdu = fits.open(f'../CLEAN/robust{robust}/{target}_data_robust{robust}.fits')
dimg, hd = np.squeeze(dhdu[0].data), dhdu[0].header
bmaj = 3600 * hd['BMAJ']

# Select the sigmas for the Gaussian distribution of the geometrical parameters for the bootstrap
sigma_inc = 1  # deg
sigma_PA = 1  # deg
sigma_dRA = bmaj/(2*np.sqrt(2*np.log(2)))/3   # fwhm to sigma/3 arcsec (3 sigmas within 1 beam, ~10 mas in accordance to estimates from ALMA Technical Handbook)
sigma_dDec = bmaj/(2*np.sqrt(2*np.log(2)))/3  # fwhm to sigma/3 arcsec (3 sigmas within 1 beam, ~10 mas in accordance to estimates from ALMA Technical Handbook)

# wsmooth and alpha range
wsmooth_range = [1e-4, 1e-3, 1e-2, 1e-1]
alpha_range = [1.05, 1.10, 1.20, 1.30]

# load the visibility data
dat = np.load(f'../../data/{target}_continuum.vis.npz')
u, v, vis, wgt = dat['u'], dat['v'], dat['Vis'], dat['Wgt']

# Best values for geometrical parameters from galario fit
inc = disk.disk[target]['incl']
PA = disk.disk[target]['PA']
dRA = dRA=disk.disk[target]['dx']
dDec = dDec=disk.disk[target]['dy']

# Start the timer
start_time = time.time()

with open('Int_bootstrap.txt', 'w') as file_Int:
    for i in tqdm(range(n_iter)):
        try:
            # Generate new parameters randomly from the Gaussian distributions
            i_inc = np.random.normal(inc, sigma_inc)
            i_PA = np.random.normal(PA, sigma_PA)
            i_dRA = np.random.normal(dRA, sigma_dRA)
            i_dDec = np.random.normal(dDec, sigma_dDec)

            # Pick hyperparameters randomly from the defined lists
            i_wsmooth = np.random.choice(wsmooth_range)
            i_alpha = np.random.choice(alpha_range)
    
            # set the disk viewing geometry with new parameters
            geom = FixedGeometry(i_inc, i_PA, i_dRA, i_dDec)
    
            # configure the fitting code setup
            FF = FrankFitter(Rmax=1.5*disk.disk[target]['rout'], geometry=geom, 
                             N=disk.disk[target]['hyp-Ncoll'], 
                             alpha=i_alpha, 
                             weights_smooth=i_wsmooth,
                             method='LogNormal')
    
            # fit the visibilities
            sol = FF.fit(u, v, vis, wgt)
        
            # save useful plot of the fit
            #priors = {'alpha': disk.disk[target]['hyp-alpha'],
            #          'wsmooth': disk.disk[target]['hyp-wsmth'],
            #          'Rmax': 1.5*disk.disk[target]['rout'],
            #          'N': disk.disk[target]['hyp-Ncoll'],
            #          'p0': 1e-35}
            #make_full_fig(u, v, vis, wgt, sol, bin_widths=[1e4, 5e4], priors=priors, save_prefix=f'./{target}_{i}')
    
            
            # Convert the arrays to strings and write them to the files
            if i == 0:
                np.savetxt('R_bootstrap.txt', [sol.r], delimiter=' ')
            np.savetxt(file_Int, [sol.I], delimiter=' ')
            #file_Int.write('\n')
        except Exception as e:
            # Print the error message and continue with the next iteration
            print(f'Fit failed in iteration {i} for inc={i_inc:.2f}, PA={i_PA:.2f}, dRA={i_dRA:.5f}, dDec={i_dDec:.5f}')
            print(f'Error: {e}')
           
# End the timer
end_time = time.time()
total_time = end_time - start_time

print(f"Bootstrapping completed in {(total_time/3600):.2f} hours!")
