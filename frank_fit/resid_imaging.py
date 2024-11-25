import os, sys
import numpy as np
execfile('ImportMS.py', globals())
sys.path.append('.')
import diskdictionary as disk

# Get target from environment variable
target = os.environ['TARGET']

ACA_disks = ["DM_Tau", "LkCa_15", "HD_34282", "J1604",
                     "J1615", "V4046_Sgr", "AA_Tau"]

if any(target == ACA_disk for ACA_disk in ACA_disks):
              grid = 'mosaic'
else:
              grid = 'standard'

mask_ra = disk.disk[target]['mask_ra']
mask_dec = disk.disk[target]['mask_dec']
mask_pa = disk.disk[target]['PA'] #position angle of mask in degrees
mask_semimajor = disk.disk[target]['rout']*1.5 #semimajor axis of mask in arcsec
mask_semiminor = mask_semimajor*np.cos(disk.disk[target]['incl'] * np.pi/180)
mask = f"ellipse[[{mask_ra},{mask_dec}], [{mask_semimajor}arcsec, {mask_semiminor}arcsec], {mask_pa}deg]"

# load model and residual visibilities into MS format
ImportMS('../data/'+target+'_continuum.ms', 
         'fits/'+target+'_frank_uv_fit', make_resid=True)

# Perform the imaging
imagename = 'CLEAN/'+target+'_resid'
for ext in ['.image*', '.mask', '.model*', '.pb*', '.psf*', '.residual*',
            '.sumwt*', '.alpha*']:
    os.system('rm -rf '+imagename+ext)
tclean(vis='CLEAN/'+target+'_continuum.resid.ms', imagename=imagename, specmode='mfs', 
       deconvolver=disk.disk[target]['cdeconvolver'], scales=disk.disk[target]['cscales'], 
       nterms=disk.disk[target]['cnterms'],mask=mask, imsize=disk.disk[target]['cimsize'], 
       gridder=grid, cell=disk.disk[target]['ccell'], gain=disk.disk[target]['cgain'],
       cycleniter=disk.disk[target]['ccycleniter'], cyclefactor=disk.disk[target]['ccyclefactor'], 
       weighting='briggs', robust=disk.disk[target]['crobust'], uvtaper=disk.disk[target]['ctaper'],
       niter=disk.disk[target]['cniter'], threshold=f"{disk.disk[target]['cthresh']*disk.disk[target]['RMS']}mJy", 
       interactive=False, savemodel='none')

# Export FITS files of the original + JvM-corrected images
exportfits(imagename+'.image', imagename+'.fits', overwrite=True)

os.system('rm -rf *.last')