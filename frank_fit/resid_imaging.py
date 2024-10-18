import os, sys
import numpy as np
execfile('ImportMS.py', globals())
sys.path.append('.')
import diskdictionary as disk

# read which disk this is about
target = str(np.loadtxt('whichdisk.txt', dtype='str'))

mask_ra = disk.disk[target]['mask_ra']
mask_dec = disk.disk[target]['mask_dec']
mask_pa = disk.disk[target]['PA'] #position angle of mask in degrees
mask_semimajor = disk.disk[target]['rout']*1.5 #semimajor axis of mask in arcsec
mask_semiminor = mask_semimajor*np.cos(disk.disk[target]['incl'] * np.pi/180)
mask = f"ellipse[[{mask_ra},{mask_dec}], [{mask_semimajor}arcsec, {mask_semiminor}arcsec], {mask_pa}deg]"

# load model and residual visibilities into MS format
ImportMS('data/'+target+'_continuum.ms', 
         'fits/'+target+'_frank_uv_fit', make_resid=True)

# Perform the imaging
imagename = 'data/'+target+'_resid'
for ext in ['.image*', '.mask', '.model*', '.pb*', '.psf*', '.residual*',
            '.sumwt*', '.alpha*']:
    os.system('rm -rf '+imagename+ext)
tclean(vis='data/'+target+'_continuum.resid.ms', imagename=imagename, 
       specmode='mfs', deconvolver=disk.disk[target]['cdeconvolver'],
       scales=disk.disk[target]['cscales'], nterms=disk.disk[target]['cnterms'],mask=mask,  
       imsize=disk.disk[target]['cimsize'], cell=disk.disk[target]['ccell'], gain=disk.disk[target]['cgain'],
       cycleniter=disk.disk[target]['ccycleniter'], cyclefactor=disk.disk[target]['ccyclefactor'], 
       weighting='briggs', robust=disk.disk[target]['crobust'], uvtaper=disk.disk[target]['ctaper'],
       niter=disk.disk[target]['cniter'], threshold=f"{disk.disk[target]['cthresh']*disk.disk[target]['RMS']}mJy", 
       savemodel='none')

# Export FITS files of the original + JvM-corrected images
exportfits(imagename+'.image', imagename+'.fits', overwrite=True)

os.system('rm -rf *.last')