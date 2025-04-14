disk = {}

#AA_Tau_continuum_robust-0.5_threshold3.0sigma_clean.image
#Beam 0.070 arcsec x 0.056 arcsec (-13.88 deg)
#Flux inside disk mask: 193.35 mJy
#Peak intensity of source: 2.89 mJy/beam
#rms: 5.61e-02 mJy/beam
#Peak SNR: 51.57

disk['AA_Tau'] = {'name': 'AA_Tau',
                 'label': 'AA_Tau',
                 'distance': 134.7,     # parsec, distance of the disk
                 'incl': 58.53531,     # deg, inclination of the disk
                 'PA': 93.77080,     # deg, position angle of the disk
                 'dx': -0.00546,     # arcsec, RA offset between the disk center and the phase center
                 'dy': 0.00483,     # arcsec, Dec offset between the disk center and the phase center
                 'rout': 1.23,     # arcsec, radius of the contour at which signal is 2xnoise_rms
                 'Rmax': 1.5,     # multiple of rout, frank hyperparameter Rmax
                 'hyp-alpha': 1.30,     # frank hyperparameter alpha
                 'hyp-wsmth': 0.01,     # frank hyperparameter w_smooth
                 'hyp-Ncoll': 400,     # frank hyperparameter N
                 'hyp-p0': 1e-35 ,     # frank hyperparameter p0 (1e-35 for LogNormal, 1e-15 for Normal)
                 'frank_method': 'LogNormal',     # frank method (LogNormal or Normal)
                 'mask_ra': '04h34m55.42980s',     # RA of the CLEAN mask center
                 'mask_dec': '24.28.52.5645',     # Dec of the CLEAN mask center
                 'cdeconvolver': 'multiscale',     # CLEAN deconvolver
                 'cscales': [0, 8, 15, 30, 80],     # pixels, CLEAN scales for multiscale deconvolver
                 'cnterms': 1,     # CLEAN Number of Taylor coefficients in the spectral model (for deconvolver mtmfs)
                 'ccell': '0.01arcsec',     # CLEAN cell size
                 'cimsize': 1024,     # CLEAN image size 
                 'cgain': 0.1,     # CLEAN gain parameter
                 'ccycleniter': 300,     # CLEAN Maximum number of minor-cycle iterations before triggering a major cycle
                 'ccyclefactor': 1.0,     # CLEAN Scaling on PSF sidelobe level to compute the minor-cycle stopping threshold
                 'crobust': 0.5,     # CLEAN robust parameter
                 'ctaper': [],     # CLEAN uv-taper
                 'cniter': 50000,     # CLEAN Maximum number of iterations 
                 'cthresh': 1.0,     # Multiple of the noise rms, CLEAN threshold
                 'RMS': 0.056,   # mJy, noise rms
                 'rEllipse': [0.06, 0.23, 0.44, 0.62],     # arcsec, ellipses to be drawn in the residual image
}
