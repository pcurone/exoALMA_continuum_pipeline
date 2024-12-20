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

import frank
from frank.radial_fitters import FrankFitter
from frank.geometry import FixedGeometry
from frank.utilities import convolve_profile, sweep_profile
from frank.io import save_fit, load_sol
from frank.make_figs import make_full_fig
frank.enable_logging()
from gofish import imagecube  # for the CLEAN profile
sys.path.append('.')
import diskdictionary as disk
from myutils import Jybeam_to_Jysr, Jysr_to_Tb, Jysr_to_Tb_err, Jysr_to_Tb_RJ, Jysr_to_Tb_RJ_err


# controls
target = 'AA_Tau'

im_dat = True   # CLEAN the observed data
frank  = True   # Run the frank fit 
im_res = True   # CLEAN the residuals (observed data - frank model) 
annotate_res = True     # Annotate the residual image with the geometrical parameters that have been used
im_mdl = True    # CLEAN the frank model


#####  Plot settings #####
img_lim = 1.3     # multiple of rout, used to set the limit of the images and the intesity radial profile
maj_ticks = 0.5          # arcsec, spacing of the major ticks in figures
min_ticks = maj_ticks/5  # arcsec, spacing of the minor ticks in figures
res_vmin, res_vmax = -5, 5    # min and max SNR values in the residual image

FOV_gofish = 5     # Clip the image cube down to a specific field-of-view spanning a range ``FOV``, where ``FOV`` is in [arcsec]
dr_gofish = None     #  Width of the annuli to split the integrated region into in [arcsec]. Default is quater of the beam major axis


###############################
##### CLEAN THE DATA
###############################
if im_dat:
    t0 = time.time()
    print('....')
    print('Imaging the data')
    print('....')
    os.environ['TARGET'] = target
    os.system('casa --nogui --nologger --nologfile -c data_imaging.py')
    print('....')
    print('Finished imaging the data')
    print('....')
    t1 = time.time()


##########################
##### PLOT THE IMAGE
##########################

# load data
dhdu = fits.open(f"CLEAN/robust{disk.disk[target]['crobust']}/{target}_data_robust{disk.disk[target]['crobust']}.fits")
dimg, hd = np.squeeze(dhdu[0].data), dhdu[0].header

# parse coordinate frame indices into physical numbers
RA = 3600 * hd['CDELT1'] * (np.arange(hd['NAXIS1']) - (hd['CRPIX1'] - 1)) 
DEC = 3600 * hd['CDELT2'] * (np.arange(hd['NAXIS2']) - (hd['CRPIX2'] - 1))
dRA, dDEC = np.meshgrid(RA - disk.disk[target]['dx'], 
                        DEC - disk.disk[target]['dy'])
freq = hd['CRVAL3']

# disk-frame polar coordinates
inclr = np.radians(disk.disk[target]['incl'])
PAr = np.radians(disk.disk[target]['PA'])
xd = (dRA * np.cos(PAr) - dDEC * np.sin(PAr)) / np.cos(inclr)
yd = (dRA * np.sin(PAr) + dDEC * np.cos(PAr))
r, theta = np.sqrt(xd**2 + yd**2), np.degrees(np.arctan2(yd, xd))

# beam parameters
bmaj, bmin, bPA = 3600 * hd['BMAJ'], 3600 * hd['BMIN'], hd['BPA']

# image setups
rout = disk.disk[target]['rout']
im_bounds = (dRA.max(), dRA.min(), dDEC.min(), dDEC.max())
dRA_lims, dDEC_lims = [img_lim*rout, -img_lim*rout], [-img_lim*rout, img_lim*rout]

#####  Plot the data image (Intensity) #####
plt.style.use('default')
fig = plt.figure(figsize=(7.0, 5.9))
gs  = gridspec.GridSpec(1, 2, width_ratios=(1, 0.04))

# image (sky-plane)
ax = fig.add_subplot(gs[0,0])

# intensity limits, and stretch
norm_data_int = ImageNormalize(vmin=0, vmax=np.nan_to_num(dimg*1e3).max(), stretch=AsinhStretch())
cmap = 'inferno'

im = ax.imshow(dimg*1e3, origin='lower', cmap=cmap, extent=im_bounds, 
               norm=norm_data_int, aspect='equal')

# beam
beam = Ellipse((dRA_lims[0] + 0.1*np.diff(dRA_lims), 
                dDEC_lims[0] + 0.1*np.diff(dDEC_lims)), bmaj, bmin, angle=90-bPA, 
                edgecolor='w', lw=1.5, facecolor='none', hatch='/////')
ax.add_artist(beam)

# limits, labeling
ax.set_xlim(dRA_lims)
ax.set_ylim(dDEC_lims)
ax.set_xlabel(r'RA offset  ($^{\prime\prime}$)', fontsize = 17, labelpad=10)
ax.set_ylabel(r'Dec offset  ($^{\prime\prime}$)', fontsize = 17, labelpad=10)

# axes style
ax.xaxis.set_major_locator(MultipleLocator(maj_ticks))
ax.xaxis.set_minor_locator(MultipleLocator(min_ticks))
ax.yaxis.set_major_locator(MultipleLocator(maj_ticks))
ax.yaxis.set_minor_locator(MultipleLocator(min_ticks)) 
ax.tick_params(which='major',axis='both',right=True,top=True, labelsize=14, pad=5,width=2.5, length=6,direction='in',color='w')
ax.tick_params(which='minor',axis='both',right=True,top=True, labelsize=14, pad=5,width=1.5, length=4,direction='in',color='w')
for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(1)


# add a scalebar
cbax = fig.add_subplot(gs[:,1])
cb = Colorbar(ax=cbax, mappable=im, orientation='vertical',
              ticklocation='right')
cb.outline.set_linewidth(2)
cb.ax.tick_params(which='major', labelsize=14,width=2.5, length=6,direction='in')
cb.ax.tick_params(which='minor', labelsize=14,width=1.5, length=4,direction='in')
cb.set_label('Intensity (mJy beam$^{-1}$)', rotation=270, labelpad=26, fontsize = 17)
cb.ax.minorticks_on()

# adjust layout
fig.subplots_adjust(wspace=0.02)
fig.subplots_adjust(left=0.11, right=0.89, bottom=0.1, top=0.98)
fig.savefig(f"figs/{target}_dataimage_robust{disk.disk[target]['crobust']}.pdf", bbox_inches='tight')


##################################
##### FRANK VISIBILITY MODELING 
##################################
if frank:
        
    print('....')
    print('Performing visibility modeling')
    print('....')

    # load the visibility data
    dat = np.load(f'../data/{target}_continuum.vis.npz')
    u, v, vis, wgt = dat['u'], dat['v'], dat['Vis'], dat['Wgt']

    # set the disk viewing geometry
    geom = FixedGeometry(disk.disk[target]['incl'], disk.disk[target]['PA'], 
                         dRA=disk.disk[target]['dx'], 
                         dDec=disk.disk[target]['dy'])

    # configure the fitting code setup
    FF = FrankFitter(Rmax=disk.disk[target]['Rmax']*disk.disk[target]['rout'], 
                     geometry=geom, 
                     N=disk.disk[target]['hyp-Ncoll'], 
                     alpha=disk.disk[target]['hyp-alpha'], 
                     weights_smooth=disk.disk[target]['hyp-wsmth'],
                     method=disk.disk[target]['frank_method'])

    # fit the visibilities
    sol = FF.fit(u, v, vis, wgt)

    make_full_fig(u, v, vis, wgt, sol, bin_widths=[1e4, 5e4], save_prefix='fits/'+target)

    # save the fit
    save_fit(u, v, vis, wgt, sol, prefix=f'fits/{target}')

    print('....')
    print('Finished visibility modeling')
    print('....')
   


####################################
##### CLEAN THE RESIDUALS
####################################
if im_res:
    t0 = time.time()
    print('....')
    print('Imaging residuals')
    print('....')
    os.environ['TARGET'] = target
    os.system('casa --nogui --nologger --nologfile -c resid_imaging.py')
    print('....')
    print('Finished imaging residuals')
    print('....')
    t1 = time.time()
    


###############################
##### PLOT THE +/- RESIDUALS
###############################

# residuals color map
c2 = plt.cm.Reds(np.linspace(0, 1, 32))
c1 = plt.cm.Blues_r(np.linspace(0, 1, 32))
c1 = np.vstack([c1, np.ones((10, 4))])
colors = np.vstack((c1, c2))
mymap = mcolors.LinearSegmentedColormap.from_list('eddymap', colors)
separations = 20   # discrete colorbar
discrete_colors = plt.get_cmap(mymap)(np.linspace(0,1,separations))
discrete_cmap = mcolors.ListedColormap(discrete_colors) 

print('....')
print('Making residual +/- plot')
print('....')

# load residual image
rhdu = fits.open(f"CLEAN/robust{disk.disk[target]['crobust']}/{target}_resid_robust{disk.disk[target]['crobust']}.fits")
rimg = np.squeeze(rhdu[0].data)

dhdu = fits.open(f"CLEAN/robust{disk.disk[target]['crobust']}/{target}_data_robust{disk.disk[target]['crobust']}.fits")
hd = dhdu[0].header

# parse coordinate frame indices into physical numbers
RA = 3600 * hd['CDELT1'] * (np.arange(hd['NAXIS1']) - (hd['CRPIX1'] - 1)) 
DEC = 3600 * hd['CDELT2'] * (np.arange(hd['NAXIS2']) - (hd['CRPIX2'] - 1))
dRA, dDEC = np.meshgrid(RA - disk.disk[target]['dx'], 
                        DEC - disk.disk[target]['dy'])
freq = hd['CRVAL3']

# disk-frame polar coordinates
inclr = np.radians(disk.disk[target]['incl'])
PAr = np.radians(disk.disk[target]['PA'])
xd = (dRA * np.cos(PAr) - dDEC * np.sin(PAr)) / np.cos(inclr)
yd = (dRA * np.sin(PAr) + dDEC * np.cos(PAr))
r, theta = np.sqrt(xd**2 + yd**2), np.degrees(np.arctan2(yd, xd))

# beam parameters
bmaj, bmin, bPA = 3600 * hd['BMAJ'], 3600 * hd['BMIN'], hd['BPA']

# image setups
rout = disk.disk[target]['rout']
im_bounds = (dRA.max(), dRA.min(), dDEC.min(), dDEC.max())
dRA_lims, dDEC_lims = [img_lim*rout, -img_lim*rout], [-img_lim*rout, img_lim*rout]

# set up plot
plt.style.use('classic')
fig = plt.figure(figsize=(7.0, 5.9))
gs  = gridspec.GridSpec(1, 2, width_ratios=(1, 0.04))

# image (sky-plane)
ax = fig.add_subplot(gs[0,0])
norm = ImageNormalize(vmin=res_vmin, vmax=res_vmax, stretch=LinearStretch())
im = ax.imshow(1e3*rimg / disk.disk[target]['RMS'], origin='lower', 
                cmap=discrete_cmap, extent=im_bounds, 
                norm=norm, aspect='equal')

# annotations
tt = np.linspace(-np.pi, np.pi, 91)
inclr = np.radians(disk.disk[target]['incl'])
PAr = np.radians(disk.disk[target]['PA'])
rEllipse = disk.disk[target]['rEllipse']
for ir in range(len(rEllipse)):
    # mark gap boundaries
    xe, ye = rEllipse[ir] * np.cos(tt) * np.cos(inclr), rEllipse[ir] * np.sin(tt)
    ax.plot( xe * np.cos(PAr) + ye * np.sin(PAr),
        -xe * np.sin(PAr) + ye * np.cos(PAr), '-', color='lightgray',
        lw=0.8, alpha=0.8)

# mark Rout
xb, yb = rout * np.cos(tt) * np.cos(inclr), rout * np.sin(tt)
ax.plot( xb * np.cos(PAr) + yb * np.sin(PAr),
        -xb * np.sin(PAr) + yb * np.cos(PAr), '--', color='lightgray',
        lw=0.8, alpha=0.5)


# beam
beam = Ellipse((dRA_lims[0] + 0.1*np.diff(dRA_lims), 
                dDEC_lims[0] + 0.1*np.diff(dDEC_lims)), bmaj, bmin, angle=90-bPA, 
                edgecolor='k', lw=1.5, facecolor='none', hatch='/////')
ax.add_artist(beam)

# limits, labeling
ax.set_xlim(dRA_lims)
ax.set_ylim(dDEC_lims)
ax.set_xlabel(r'RA offset  ($^{\prime\prime}$)', fontsize = 17, labelpad=10)
ax.set_ylabel(r'Dec offset  ($^{\prime\prime}$)', fontsize = 17, labelpad=10)

# axes style
ax.xaxis.set_major_locator(MultipleLocator(maj_ticks))
ax.xaxis.set_minor_locator(MultipleLocator(min_ticks))
ax.yaxis.set_major_locator(MultipleLocator(maj_ticks))
ax.yaxis.set_minor_locator(MultipleLocator(min_ticks)) 
ax.tick_params(which='major',axis='both',right=True,top=True, labelsize=14, pad=5,width=2.5, length=6,direction='in',color='k')
ax.tick_params(which='minor',axis='both',right=True,top=True, labelsize=14, pad=5,width=1.5, length=4,direction='in',color='k')
for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(2)

if annotate_res:
    ax.text(0.05, 0.9, f'inc={disk.disk[target]["incl"]:.1f}°, PA = {disk.disk[target]["PA"]:.1f}°, \
            \ndx = {disk.disk[target]["dx"]:.4f}\'\', dy = {disk.disk[target]["dy"]:.4f}\'\' ', 
            transform=ax.transAxes)

# add a scalebar
cbax = fig.add_subplot(gs[:,1])
cb = Colorbar(ax=cbax, mappable=im, orientation='vertical',
                ticklocation='right') 
cb.outline.set_linewidth(2)
cb.ax.tick_params(which='major', labelsize=14,width=2.5, length=6,direction='in')
cb.ax.tick_params(which='minor', labelsize=14,width=1.5, length=4,direction='in')
cb.set_label('Residual S/N', rotation=270 , labelpad=26, fontsize = 17)
cb.ax.minorticks_on()

# adjust layout
fig.subplots_adjust(wspace=0.02)
fig.subplots_adjust(left=0.11, right=0.89, bottom=0.1, top=0.98)
fig.savefig(f"figs/{target}_resid_robust{disk.disk[target]['crobust']}.pdf", bbox_inches='tight')



######################################
##### PLOT THE SWEEP FRANK MODEL
######################################

sol = load_sol(f'fits/{target}_frank_sol.obj')
r_frank = sol.r
Inu_frank = sol.I

sweep_img = sweep_profile(r_frank, Inu_frank, project=True, geom=sol.geometry, dr=1e-3)
Tb_sweep_img = Jysr_to_Tb(sweep_img[0], freq)
im_bounds_sweep = (sweep_img[1], -sweep_img[1], -sweep_img[2], sweep_img[2])

### Plot the data image
plt.style.use('default')
fig = plt.figure(figsize=(7.0, 5.9))
gs  = gridspec.GridSpec(1, 2, width_ratios=(1, 0.04))
ax = fig.add_subplot(gs[0,0])

# intensity limits, and stretch
norm = ImageNormalize(vmin=0, vmax=1, stretch=AsinhStretch())
cmap = 'inferno'

im = ax.imshow(sweep_img[0]/np.nan_to_num(sweep_img[0]).max(), origin='lower', cmap=cmap, extent=im_bounds_sweep, 
               norm=norm, aspect='equal')

#annotations
tt = np.linspace(-np.pi, np.pi, 91)
inclr = np.radians(disk.disk[target]['incl'])
PAr = np.radians(disk.disk[target]['PA'])
rEllipse = disk.disk[target]['rEllipse']
for ir in range(len(rEllipse)):
    # mark gap boundaries
    xe, ye = rEllipse[ir] * np.cos(tt) * np.cos(inclr), rEllipse[ir] * np.sin(tt)
    ax.plot( xe * np.cos(PAr) + ye * np.sin(PAr),
        -xe * np.sin(PAr) + ye * np.cos(PAr), '-', color='lightgray',
        lw=0.8, alpha=0.8)

# mark Rout
xb, yb = rout * np.cos(tt) * np.cos(inclr), rout * np.sin(tt)
ax.plot( xb * np.cos(PAr) + yb * np.sin(PAr),
        -xb * np.sin(PAr) + yb * np.cos(PAr), '--', color='lightgray',
        lw=0.8, alpha=0.5)

# limits, labeling
ax.set_xlim(dRA_lims)
ax.set_ylim(dDEC_lims)
ax.set_xlabel(r'RA offset  ($^{\prime\prime}$)', fontsize = 17, labelpad=10)
ax.set_ylabel(r'Dec offset  ($^{\prime\prime}$)', fontsize = 17, labelpad=10)

# axes style
ax.xaxis.set_major_locator(MultipleLocator(maj_ticks))
ax.xaxis.set_minor_locator(MultipleLocator(min_ticks))
ax.yaxis.set_major_locator(MultipleLocator(maj_ticks))
ax.yaxis.set_minor_locator(MultipleLocator(min_ticks)) 
ax.tick_params(which='major',axis='both',right=True,top=True, labelsize=14, pad=5,width=2.5, length=6,direction='in',color='w')
ax.tick_params(which='minor',axis='both',right=True,top=True, labelsize=14, pad=5,width=1.5, length=4,direction='in',color='w')
for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(1)


# add a scalebar
cbax = fig.add_subplot(gs[:,1])
cb = Colorbar(ax=cbax, mappable=im, orientation='vertical',
              ticklocation='right')
cb.outline.set_linewidth(2)
cb.ax.tick_params(which='major', labelsize=14,width=2.5, length=6,direction='in')
cb.ax.tick_params(which='minor', labelsize=14,width=1.5, length=4,direction='in')
cb.set_label('Normalized intensity', rotation=270, labelpad=26, fontsize = 17)
cb.ax.minorticks_on()

# adjust layout
fig.subplots_adjust(wspace=0.02)
fig.subplots_adjust(left=0.11, right=0.89, bottom=0.1, top=0.98)

fig.savefig(f'figs/{target}_Sweep_modelimage.pdf', bbox_inches='tight')


#################################
##### BRIGHTNESS RADIAL PROFILE
#################################

# plot the radial profile and compare with the CLEAN map

sol = load_sol(f'fits/{target}_frank_sol.obj')
r_frank = sol.r
Inu_frank = sol.I

# Convolve the frank profile with the CLEAN beam
# Units: [arcsec], [arcsec], [deg]
clean_beam = {'bmaj':bmaj, 'bmin':bmin, 'beam_pa':bPA}
Inu_frank_convolved = convolve_profile(r_frank, Inu_frank, disk.disk[target]['incl'], disk.disk[target]['PA'], clean_beam)

# convert to brightness temperatures (full Planck law)
Tb_frank = Jysr_to_Tb(Inu_frank, freq)
Tb_frank_convolved = Jysr_to_Tb(Inu_frank_convolved, freq)
# convert to brightness temperatures (R-J limit)
Tb_frank_RJ = Jysr_to_Tb_RJ(Inu_frank, freq)
Tb_frank_convolved_RJ = Jysr_to_Tb_RJ(Inu_frank_convolved, freq)

# Obtain the CLEAN profile using the imagecube function from gofish
cube = imagecube(f"CLEAN/robust{disk.disk[target]['crobust']}/{target}_data_robust{disk.disk[target]['crobust']}.fits", FOV=FOV_gofish)
r_clean, I_clean, dI_clean = cube.radial_profile(x0=disk.disk[target]['dx'], y0=disk.disk[target]['dy'], inc=disk.disk[target]['incl'], PA=disk.disk[target]['PA'], dr=dr_gofish)
# convert to brightness temperatures (full Planck law)
Tb_clean, dTb_clean = Jysr_to_Tb_err(Jybeam_to_Jysr(I_clean, bmin, bmaj), Jybeam_to_Jysr(dI_clean, bmin, bmaj), freq)
# convert to brightness temperatures (R-J limit)
Tb_clean_RJ, dTb_clean_RJ = Jysr_to_Tb_RJ_err(Jybeam_to_Jysr(I_clean, bmin, bmaj), Jybeam_to_Jysr(dI_clean, bmin, bmaj), freq)


##### Plot (full Planck) #####
fig, axs = plt.subplots(1, 2, figsize=(18,5))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.35)

axs[0].hlines(y=0, xmin=0, xmax=img_lim*rout, color='gray', linestyle='dashed', linewidth=1)

for ax in axs.flat:
    ax.fill_between(r_clean, Tb_clean-dTb_clean, Tb_clean+dTb_clean, color='gray', alpha=0.5)
    ax.plot(r_clean, Tb_clean, 'k', lw=3, label='CLEAN')
    ax.plot(r_frank, Tb_frank, 'r', lw=3, label='Frank logarithmic fit')
    ax.plot(r_frank[Tb_frank_convolved>0], Tb_frank_convolved[Tb_frank_convolved>0], 'b', lw=3, label='Frank convolved')

    ax.set_xlim([0, img_lim*rout])
    ax.xaxis.set_major_locator(MultipleLocator(maj_ticks))
    ax.xaxis.set_minor_locator(MultipleLocator(min_ticks))

    ax.tick_params(which='major',axis='both',right=True,top=True, labelsize=14, pad=7,width=2.5, length=6,direction='in',color='k')
    ax.tick_params(which='minor',axis='both',right=True,top=True, labelsize=14, pad=7,width=1.5, length=4,direction='in',color='k')
    ax.set_xlabel(r'$R \,\,$ [arcsec]', fontsize = 17, labelpad=10)
    ax.set_ylabel('Brightness temperature [K]', fontsize = 17, labelpad=10)
    ax.legend(fontsize=13)
    
    for side in ax.spines.keys():
        ax.spines[side].set_linewidth(3) 

axs[1].set_yscale('log')
axs[1].set_ylim([1.1, np.amax(Tb_frank)*1.3]) 

fig.savefig(f"figs/{target}_Tb_profile_robust{disk.disk[target]['crobust']}.pdf", bbox_inches='tight')


##### Plot (R-J limit) #####
fig, axs = plt.subplots(1, 2, figsize=(18,5))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.35)

axs[0].hlines(y=0, xmin=0, xmax=img_lim*rout, color='gray', linestyle='dashed', linewidth=1)

for ax in axs.flat:
    ax.fill_between(r_clean, Tb_clean_RJ-dTb_clean_RJ, Tb_clean_RJ+dTb_clean_RJ, color='gray', alpha=0.5)
    ax.plot(r_clean, Tb_clean_RJ, 'k', lw=3, label='CLEAN')
    ax.plot(r_frank, Tb_frank_RJ, 'r', lw=3, label='Frank logarithmic fit')
    ax.plot(r_frank[Tb_frank_convolved_RJ>0], Tb_frank_convolved_RJ[Tb_frank_convolved_RJ>0], 'b', lw=3, label='Frank convolved')

    ax.set_xlim([0, img_lim*rout])
    ax.xaxis.set_major_locator(MultipleLocator(maj_ticks))
    ax.xaxis.set_minor_locator(MultipleLocator(min_ticks))

    ax.tick_params(which='major',axis='both',right=True,top=True, labelsize=14, pad=7,width=2.5, length=6,direction='in',color='k')
    ax.tick_params(which='minor',axis='both',right=True,top=True, labelsize=14, pad=7,width=1.5, length=4,direction='in',color='k')
    ax.set_xlabel(r'$R \,\,$ [arcsec]', fontsize = 17, labelpad=10)
    ax.set_ylabel('Brightness temperature [K]', fontsize = 17, labelpad=10)
    ax.legend(fontsize=13)
    
    for side in ax.spines.keys():
        ax.spines[side].set_linewidth(3) 

axs[1].set_yscale('log')
axs[1].set_ylim([0.01, np.amax(Tb_frank_RJ)*1.3]) 

fig.savefig(f"figs/{target}_Tb_profile_RJ_robust{disk.disk[target]['crobust']}.pdf", bbox_inches='tight')



##########################
##### CLEAN THE MODEL
##########################
if im_mdl:
    t0 = time.time()
    print('....')
    print('Imaging model')
    print('....')
    os.environ['TARGET'] = target
    os.system('casa --nogui --nologerr --nologfile -c model_imaging.py') 
    print('....')
    print('Finished imaging model')
    print('....')
    t1 = time.time()



######################################
##### PLOT THE CLEANED FRANK MODEL
######################################

# load model
dhdu = fits.open(f"CLEAN/robust{disk.disk[target]['crobust']}/{target}_model_robust{disk.disk[target]['crobust']}.fits")
dimg, hd = np.squeeze(dhdu[0].data), dhdu[0].header

# parse coordinate frame indices into physical numbers
RA = 3600 * hd['CDELT1'] * (np.arange(hd['NAXIS1']) - (hd['CRPIX1'] - 1)) 
DEC = 3600 * hd['CDELT2'] * (np.arange(hd['NAXIS2']) - (hd['CRPIX2'] - 1))
dRA, dDEC = np.meshgrid(RA - disk.disk[target]['dx'], 
                        DEC - disk.disk[target]['dy'])
freq = hd['CRVAL3']

# disk-frame polar coordinates
inclr = np.radians(disk.disk[target]['incl'])
PAr = np.radians(disk.disk[target]['PA'])
xd = (dRA * np.cos(PAr) - dDEC * np.sin(PAr)) / np.cos(inclr)
yd = (dRA * np.sin(PAr) + dDEC * np.cos(PAr))
r, theta = np.sqrt(xd**2 + yd**2), np.degrees(np.arctan2(yd, xd))

# beam parameters
bmaj, bmin, bPA = 3600 * hd['BMAJ'], 3600 * hd['BMIN'], hd['BPA']

# image setups
rout = disk.disk[target]['rout']
im_bounds = (dRA.max(), dRA.min(), dDEC.min(), dDEC.max())
dRA_lims, dDEC_lims = [img_lim*rout, -img_lim*rout], [-img_lim*rout, img_lim*rout]

#####  Plot the data image (Intensity) #####
plt.style.use('default')
fig = plt.figure(figsize=(7.0, 5.9))
gs  = gridspec.GridSpec(1, 2, width_ratios=(1, 0.04))

# image (sky-plane)
ax = fig.add_subplot(gs[0,0])

# intensity limits, and stretch
cmap = 'inferno'

im = ax.imshow(dimg*1e3, origin='lower', cmap=cmap, extent=im_bounds, 
               norm=norm_data_int, aspect='equal')

# beam
beam = Ellipse((dRA_lims[0] + 0.1*np.diff(dRA_lims), 
                dDEC_lims[0] + 0.1*np.diff(dDEC_lims)), bmaj, bmin, angle=90-bPA, 
                edgecolor='w', lw=1.5, facecolor='none', hatch='/////')
ax.add_artist(beam)

# limits, labeling
ax.set_xlim(dRA_lims)
ax.set_ylim(dDEC_lims)
ax.set_xlabel(r'RA offset  ($^{\prime\prime}$)', fontsize = 17, labelpad=10)
ax.set_ylabel(r'Dec offset  ($^{\prime\prime}$)', fontsize = 17, labelpad=10)

# axes style
ax.xaxis.set_major_locator(MultipleLocator(maj_ticks))
ax.xaxis.set_minor_locator(MultipleLocator(min_ticks))
ax.yaxis.set_major_locator(MultipleLocator(maj_ticks))
ax.yaxis.set_minor_locator(MultipleLocator(min_ticks)) 
ax.tick_params(which='major',axis='both',right=True,top=True, labelsize=14, pad=5,width=2.5, length=6,direction='in',color='w')
ax.tick_params(which='minor',axis='both',right=True,top=True, labelsize=14, pad=5,width=1.5, length=4,direction='in',color='w')
for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(1)

# add a scalebar
cbax = fig.add_subplot(gs[:,1])

cb = Colorbar(ax=cbax, mappable=im, orientation='vertical',
              ticklocation='right')
cb.outline.set_linewidth(2)
cb.ax.tick_params(which='major', labelsize=14,width=2.5, length=6,direction='in')
cb.ax.tick_params(which='minor', labelsize=14,width=1.5, length=4,direction='in')
cb.set_label('Intensity (mJy beam$^{-1}$)', rotation=270, labelpad=26, fontsize = 17)
cb.ax.minorticks_on()

# adjust layout
fig.subplots_adjust(wspace=0.02)
fig.subplots_adjust(left=0.11, right=0.89, bottom=0.1, top=0.98)
fig.savefig(f"figs/{target}_CLEAN_modelimage_robust{disk.disk[target]['crobust']}.pdf", bbox_inches='tight')
