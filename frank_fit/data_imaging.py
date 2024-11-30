import os, sys
import numpy as np
execfile('ImportMS.py', globals())
sys.path.append('.')
import diskdictionary as disk

# Get target from environment variable
target = os.environ['TARGET']

# Set the right folder
folder_path = f"CLEAN/robust{disk.disk[target]['crobust']}"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder created: {folder_path}")

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
noise_annulus_in, noise_annulus_out = disk.disk[target]['rout']*2, disk.disk[target]['rout']*4
noise_annulus = f"annulus[[{mask_ra}, {mask_dec}],['{noise_annulus_in}arcsec', '{noise_annulus_out}.arcsec']]"

# Perform the imaging
imagename = f"{folder_path}/{target}_data_robust{disk.disk[target]['crobust']}"
for ext in ['.image*', '.mask', '.model*', '.pb*', '.psf*', '.residual*', 
            '.sumwt*', '.alpha*']:
    os.system('rm -rf '+imagename+ext)
tclean(vis='../data/'+target+'_continuum.ms', imagename=imagename, specmode='mfs', 
       deconvolver=disk.disk[target]['cdeconvolver'], scales=disk.disk[target]['cscales'], 
       nterms=disk.disk[target]['cnterms'],mask=mask, imsize=disk.disk[target]['cimsize'], 
       gridder=grid, cell=disk.disk[target]['ccell'], gain=disk.disk[target]['cgain'],
       cycleniter=disk.disk[target]['ccycleniter'], cyclefactor=disk.disk[target]['ccyclefactor'], 
       weighting='briggs', robust=disk.disk[target]['crobust'], uvtaper=disk.disk[target]['ctaper'],
       niter=disk.disk[target]['cniter'], threshold=f"{disk.disk[target]['cthresh']*disk.disk[target]['RMS']}mJy", 
       interactive=False, savemodel='none')

# Export FITS files 
exportfits(f'{imagename}.image', f'{imagename}.fits', overwrite=True)

# Read fits image and save image properties
headerlist = imhead(f'{imagename}.image', mode = 'list')
beammajor = headerlist['beammajor']['value']
beamminor = headerlist['beamminor']['value']
beampa = headerlist['beampa']['value']
target_stats = imstat(imagename = f'{imagename}.image', region = mask)
target_flux = target_stats['flux'][0]
peak_intensity = target_stats['max'][0]
rms = imstat(imagename = f'{imagename}.image', region = noise_annulus)['rms'][0]
SNR = peak_intensity/rms
print(f"#{imagename}.image")
print(f"#Beam {beammajor:.3f} arcsec x {beamminor:.3f}, PA {beampa:.2f} deg")
print(f"#Flux inside disk mask: {target_flux*1000:.2f} mJy")
print(f"#Peak intensity of source: %.2f mJy/beam" % (peak_intensity*1000,))
print(f"#rms (in annulus between {noise_annulus_in:.3f} and {noise_annulus_out:.3f} arcsec): {rms*1000:.2e} mJy/beam")
print(f"#Peak SNR: {SNR:.2f}")

# Write info of the image to txt file
with open(f"{folder_path}/Info_image_data_{target}_robust{disk.disk[target]['crobust']}.txt", 'w') as Info_txt:
    Info_txt.write(f'# {imagename}.image\n')
    Info_txt.write(f'# Beammajor(arcsec)    Beamminor(arcsec)    PA(deg)    Flux_in_mask(mJy)    Peak_int(mJy/beam)    rms(mJy/beam)    Peak_SNR\n')
    Info_txt.write(f'{beammajor:.3f}   {beamminor:.3f}   {beampa:.2f}   {target_flux*1e3:.2f}   {peak_intensity*1e3:.2f}   {rms*1e3:.2e}   {SNR:.2f}')

os.system('rm -rf *.last')
