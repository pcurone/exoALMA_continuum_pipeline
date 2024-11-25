import numpy as np

# Enable/disable numba
use_numba = True
if use_numba:
    from numba import njit

# Helper to conditionally apply njit
def conditional_njit(func):
    return njit(fastmath=True, parallel=True)(func) if use_numba else func


##########################################
#####   DEFINE THE MODEL FUNCTIONS   #####
##########################################

# define a central Gaussian in 2D on the image array (with numba to make it faster)
@conditional_njit
def Gauss_2D(f, sigma, inc_rad):
    r = ((x / np.cos(inc_rad))**2. + y**2.)**0.5
    imagemap = f * np.exp(-r**2. / (2. * sigma**2.))
    return imagemap


# define a Gaussian ring in 2D on the image array (with numba to make it faster)
@conditional_njit
def ring_2D(f, R, sigma, inc_rad):
    r = ((x / np.cos(inc_rad))**2. + y**2.)**0.5
    imagemap = f * np.exp(-((r - R)**2.) / (2. * sigma**2.))
    return imagemap


# Define a 2D 'Gaussian' arc on the image array (with numba to make it faster)
'''
2D arc defined as a Gaussian 2D ring with an exponential cutoff on the azimuthal coordinate

phi0 = angle in deg of the central point of the arc
defined with same convention of the PA, 0=up, 90=left, 180=down, 270=right

sigma_phi = angle in deg of the azimuthal extension of the arc,
or better, angle after which there is the exponential cutoff
'''
@conditional_njit
def arc_2D(f, R, sigma, phi_rad, sigma_phi_rad, inc_rad):
    r = ((x / np.cos(inc_rad))**2. + y**2.)**0.5
    phi = np.arctan2(y, x / np.cos(inc_rad)) - phi_rad
    phi = (phi + np.pi) % (2 * np.pi) - np.pi  # Normalize angle to [-π, π]
    imagemap = f * np.exp(-((r - R)**2.) / (2. * sigma**2.)) * np.exp(-phi**2. / (2. * sigma_phi_rad**2.))
    return imagemap



# Define the image model as combination of rings and arcs (with numba to make it faster)

# 1 ring + 1 arc
@conditional_njit
def Image2D_1ring_1arc(f1, r1, sigma1, f_arc1, r_arc1, sigmar_arc1, phi_arc1, sigmaphi_arc1, inc_rad):
    model_image = ring_2D(f1, r1, sigma1, inc_rad) + arc_2D(f_arc1, r_arc1, sigmar_arc1, phi_arc1, sigmaphi_arc1, inc_rad)
    return model_image

# 1 Central Gaussian + 2 rings + 1 arc
@conditional_njit
def Image2D_1Gauss_2rings_1arc(f0, sigma0, f1, r1, sigma1, f2, r2, sigma2, f_arc1, r_arc1, sigmar_arc1, phi_arc1, sigmaphi_arc1, inc_rad):
    model_image = (
        Gauss_2D(f0, sigma0, inc_rad)
        + ring_2D(f1, r1, sigma1, inc_rad)
        + ring_2D(f2, r2, sigma2, inc_rad)
        + arc_2D(f_arc1, r_arc1, sigmar_arc1, phi_arc1, sigmaphi_arc1, inc_rad)
    )
    return model_image

# 2 rings + 2 arcs
@conditional_njit
def Image2D_2rings_2arcs(f1, r1, sigma1, f2, r2, sigma2, f_arc1, r_arc1, sigmar_arc1, phi_arc1, sigmaphi_arc1, f_arc2, r_arc2, sigmar_arc2, phi_arc2, sigmaphi_arc2, inc_rad):
    model_image = (
        ring_2D(f1, r1, sigma1, inc_rad)
        + ring_2D(f2, r2, sigma2, inc_rad)
        + arc_2D(f_arc1, r_arc1, sigmar_arc1, phi_arc1, sigmaphi_arc1, inc_rad)
        + arc_2D(f_arc2, r_arc2, sigmar_arc2, phi_arc2, sigmaphi_arc2, inc_rad)
    )
    return model_image

# 3 rings + 1 arcs
@njit(fastmath=True, parallel=True)
def Image2D_3rings_1arc(f1, r1, sigma1, f2, r2, sigma2, f3, r3, sigma3, f_arc1, r_arc1, sigmar_arc1, phi_arc1, sigmaphi_arc1, inc_rad):
    model_image = (
        ring_2D(f1, r1, sigma1, inc_rad)
        + ring_2D(f2, r2, sigma2, inc_rad)
        + ring_2D(f3, r3, sigma3, inc_rad)
        + arc_2D(f_arc1, r_arc1, sigmar_arc1, phi_arc1, sigmaphi_arc1, inc_rad)
    )
    return model_image


##############################
#####   MODEL REGISTRY   #####
##############################

model_registry = {
    "1ring_1arc": {
        "function": Image2D_1ring_1arc,
        "parameters": ["f1", "r1", "sigma1", "f_arc1", "r_arc1", "sigmar_arc1", "phi_arc1", "sigmaphi_arc1"],
        "labels": ["$f_1$",  r"$r_1$", r"$\sigma_1$", "$f_{\mathrm{Arc}1}$",  r"$r_{\mathrm{Arc}1}$", r"$\sigma_{\mathrm{Arc}1}$", r"$\phi_{\mathrm{Arc}1}$", r"$\sigma_{\phi\mathrm{Arc}1}$",    "inc", "PA", r"$\Delta RA$", r"$\Delta Dec$"]
    },
    "1Gauss_2rings_1arc": {
        "function": Image2D_1Gauss_2rings_1arc,
        "parameters": ["f0", "sigma0", "f1", "r1", "sigma1", "f2", "r2", "sigma2", "f_arc1", "r_arc1", "sigmar_arc1", "phi_arc1", "sigmaphi_arc1"],
        "labels": ["$f_0$", r"$\sigma_0$", "$f_1$",  r"$r_1$", r"$\sigma_1$", "$f_2$",  r"$r_2$", r"$\sigma_2$", "$f_{\mathrm{Arc}1}$",  r"$r_{\mathrm{Arc}1}$", r"$\sigma_{\mathrm{Arc}1}$", r"$\phi_{\mathrm{Arc}1}$", r"$\sigma_{\phi\mathrm{Arc}1}$",    "inc", "PA", r"$\Delta RA$", r"$\Delta Dec$"]
    },
    "2rings_2arcs": {
        "function": Image2D_2rings_2arcs,
        "parameters": ["f1", "r1", "sigma1", "f2", "r2", "sigma2", "f_arc1", "r_arc1", "sigmar_arc1", "phi_arc1", "sigmaphi_arc1", "f_arc2", "r_arc2", "sigmar_arc2", "phi_arc2", "sigmaphi_arc2"],
        "labels": ["$f_1$",  r"$r_1$", r"$\sigma_1$", "$f_2$",  r"$r_2$", r"$\sigma_2$", "$f_{\mathrm{Arc}1}$",  r"$r_{\mathrm{Arc}1}$", r"$\sigma_{\mathrm{Arc}1}$", r"$\phi_{\mathrm{Arc}1}$", r"$\sigma_{\phi\mathrm{Arc}1}$",  "$f_{\mathrm{Arc}2}$",  r"$r_{\mathrm{Arc}2}$", r"$\sigma_{\mathrm{Arc}2}$", r"$\phi_{\mathrm{Arc}2}$", r"$\sigma_{\phi\mathrm{Arc}2}$",  "inc", "PA", r"$\Delta RA$", r"$\Delta Dec$"]
    },
    "3rings_1arc": {
        "function": Image2D_3rings_1arc,
        "parameters": ["f1", "r1", "sigma1", "f2", "r2", "sigma2", "f3", "r3", "sigma3", "f_arc1", "r_arc1", "sigmar_arc1", "phi_arc1", "sigmaphi_arc1"],
        "labels": ["$f_1$",  r"$r_1$", r"$\sigma_1$", "$f_2$",  r"$r_2$", r"$\sigma_2$", "$f_3$",  r"$r_3$", r"$\sigma_3$", "$f_{\mathrm{Arc}1}$",  r"$r_{\mathrm{Arc}1}$", r"$\sigma_{\mathrm{Arc}1}$", r"$\phi_{\mathrm{Arc}1}$", r"$\sigma_{\phi\mathrm{Arc}1}$", "inc", "PA", r"$\Delta RA$", r"$\Delta Dec$"]
    }
}

