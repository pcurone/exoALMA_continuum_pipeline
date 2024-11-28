'''
Script for testing the hyperparameters effect on the frank model.
Testing both the LogNormal and Normal methods with different values of alpha and wsmooth, while keeping Rmax and N fixed
'''

import os, sys, time
import numpy as np
import matplotlib.pyplot as plt

import frank
from frank.radial_fitters import FrankFitter
from frank.geometry import FixedGeometry
from frank.utilities import convolve_profile, sweep_profile
from frank.io import save_fit, load_sol
from frank.make_figs import make_full_fig
frank.enable_logging()
sys.path.append('..')
import diskdictionary as disk

# controls
target = 'CX_Tau'

methods = ['LogNormal', 'Normal']
wsmooth_range = [1e-4, 1e-3, 1e-2, 1e-1]
alpha_range = [1.05, 1.10, 1.20, 1.30]


for method in methods:
    for wsmooth in wsmooth_range:
        for alpha in alpha_range:
    
            print('....')
            print(f'frank fit with method {method}, wsmooth={wsmooth}, and alpha={alpha}')
            print('....')

            # load the visibility data
            dat = np.load(f'../../data/{target}_continuum.vis.npz')
            u, v, vis, wgt = dat['u'], dat['v'], dat['Vis'], dat['Wgt']

            # set the disk viewing geometry
            geom = FixedGeometry(disk.disk[target]['incl'], disk.disk[target]['PA'], 
                                    dRA=disk.disk[target]['dx'], 
                                    dDec=disk.disk[target]['dy'])

            # configure the fitting code setup
            FF = FrankFitter(Rmax=disk.disk[target]['Rmax']*disk.disk[target]['rout'], 
                                geometry=geom, 
                                N=disk.disk[target]['hyp-Ncoll'], 
                                alpha=alpha, 
                                weights_smooth=wsmooth,
                                method=method)

            # fit the visibilities
            sol = FF.fit(u, v, vis, wgt)
            
            # Producing the figure with the results
            make_full_fig(u, v, vis, wgt, sol, bin_widths=[1e4, 5e4], save_prefix=f'./{target}_method{method}_wsmooth{wsmooth}_alpha{alpha}')

            # save the fit
            #save_fit(u, v, vis, wgt, sol, prefix=f'./{target}_method{method}_wsmooth{wsmooth}_alpha{alpha}')

            print('....')
            print('Finished frank fit with method {method}, wsmooth={wsmooth}, and alpha={alpha}')
            print('....')
   
