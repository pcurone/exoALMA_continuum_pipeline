##########################################################################
import numpy as np
##########################################################################

##########################################
#####   DEFINE THE MODEL FUNCTIONS   #####
##########################################

def PointSource(f0, nR):
    """ Dirac delta function for a Point source """
    delta = np.zeros(nR)
    delta[0] = 1
    return f0 * delta

def CentralGaussian(f0, sigma, Rmin, dR, nR):
    """ Central Gaussian brightness profile """
    # radial grid
    R = np.linspace(Rmin, Rmin + dR*nR, nR, endpoint=False)
    return f0 * np.exp(-0.5*(R/sigma)**2)

def CentraltaperedGaussian_centralGaussian(f0, sigma0, gamma, f1, sigma1, Rmin, dR, nR):
    """ Central Gaussian with exponential taper + central Gaussian brightness profile """
    # radial grid
    R = np.linspace(Rmin, Rmin + dR*nR, nR, endpoint=False)
    return f0 * (R/sigma0)**(-gamma) * np.exp(-0.5*(R/sigma0)**2) + f1 * np.exp(-0.5*(R/sigma1)**2)

def SingleRingProfile(f1, r1, sigma1, Rmin, dR, nR):
    """ Single Gaussian ring brightness profile """
    # radial grid
    R = np.linspace(Rmin, Rmin + dR*nR, nR, endpoint=False)
    return f1 * np.exp(-(R-r1)**2./(2.*sigma1**2.)) 

def CentralGauss_SingleRingProfile(f0, sigma0, f1, r1, sigma1, Rmin, dR, nR):
    """ Central Gaussian + single Gaussian ring brightness profile """
    # radial grid
    R = np.linspace(Rmin, Rmin + dR*nR, nR, endpoint=False)
    return f0 * np.exp(-(R)**2./(2.*sigma0**2.)) + f1 * np.exp(-(R-r1)**2./(2.*sigma1**2.)) 

def DoubleRingProfile(f1, r1, sigma1, f2, r2, sigma2, Rmin, dR, nR):
    """ Double Gaussian ring brightness profile. """
    # radial grid
    R = np.linspace(Rmin, Rmin + dR*nR, nR, endpoint=False)
    return f1 * np.exp(-(R-r1)**2./(2.*sigma1**2.)) + f2 * np.exp(-(R-r2)**2./(2.*sigma2**2.))

def ThreeRingProfile(f1, r1, sigma1, f2, r2, sigma2, f3, r3, sigma3, Rmin, dR, nR):
    """ Three Gaussian rings brightness profile """
    # radial grid
    R = np.linspace(Rmin, Rmin + dR*nR, nR, endpoint=False)
    return f1 * np.exp(-(R-r1)**2./(2.*sigma1**2.)) + f2 * np.exp(-(R-r2)**2./(2.*sigma2**2.)) + f3 * np.exp(-(R-r3)**2./(2.*sigma3**2.))

def CentralGauss_ThreeRingProfile(f0, sigma0, f1, r1, sigma1, f2, r2, sigma2, f3, r3, sigma3, Rmin, dR, nR):
    """ Three Gaussian rings + central Gaussian brightness profile """
    # radial grid
    R = np.linspace(Rmin, Rmin + dR*nR, nR, endpoint=False)
    return f0 * np.exp(-(R)**2./(2.*sigma0**2.)) + f1 * np.exp(-(R-r1)**2./(2.*sigma1**2.)) + f2 * np.exp(-(R-r2)**2./(2.*sigma2**2.)) + f3 * np.exp(-(R-r3)**2./(2.*sigma3**2.))

def FourRingProfile(f1, r1, sigma1, f2, r2, sigma2, f3, r3, sigma3, f4, r4, sigma4, Rmin, dR, nR):
    """ Four Gaussian rings brightness profile """
    # radial grid
    R = np.linspace(Rmin, Rmin + dR*nR, nR, endpoint=False)
    return f1 * np.exp(-(R-r1)**2./(2.*sigma1**2.)) + f2 * np.exp(-(R-r2)**2./(2.*sigma2**2.)) + f3 * np.exp(-(R-r3)**2./(2.*sigma3**2.)) + f4 * np.exp(-(R-r4)**2./(2.*sigma4**2.))

def FourRingProfile_PointSource(f0, f1, r1, sigma1, f2, r2, sigma2, f3, r3, sigma3, f4, r4, sigma4, Rmin, dR, nR):
    """ Four Gaussian rings + central point source brightness profile """
    # radial grid
    delta = np.zeros(nR)
    delta[0] = 1
    R = np.linspace(Rmin, Rmin + dR*nR, nR, endpoint=False)
    return f0 * delta + f1 * np.exp(-(R-r1)**2./(2.*sigma1**2.)) + f2 * np.exp(-(R-r2)**2./(2.*sigma2**2.)) + f3 * np.exp(-(R-r3)**2./(2.*sigma3**2.)) + f4 * np.exp(-(R-r4)**2./(2.*sigma4**2.))

##############################
#####   MODEL REGISTRY   #####
##############################

model_registry = {
    "PointSource": {
        "function": PointSource,
        "parameters": ["f0", "nR"],
        "labels": ["$f_0$", "inc", "PA", r"$\Delta RA$", r"$\Delta Dec$"]
    },
    "CentralGaussian": {
        "function": CentralGaussian,
        "parameters": ["f0", "sigma", "Rmin", "dR", "nR"],
        "labels": ["$f_0$", r"$\sigma$", "inc", "PA", r"$\Delta RA$", r"$\Delta Dec$"]
    },
    "CentraltaperedGaussian_centralGaussian": {
        "function": CentraltaperedGaussian_centralGaussian,
        "parameters": ["f0", "sigma0", "gamma", "f1", "sigma1", "Rmin", "dR", "nR"],
        "labels": ["$f_0$", r"$\sigma_0$", r"$\gamma$", "$f_1$", r"$\sigma_1$", "inc", "PA", r"$\Delta RA$", r"$\Delta Dec$"]
    },
    "SingleRingProfile": {
        "function": SingleRingProfile,
        "parameters": ["f1", "r1", "sigma1", "Rmin", "dR", "nR"],
        "labels": ["$f_1$", r"$r_1$", r"$\sigma_1$", "inc", "PA", r"$\Delta RA$", r"$\Delta Dec$"]
    },
    "CentralGauss_SingleRingProfile": {
        "function": CentralGauss_SingleRingProfile,
        "parameters": ["f0", "sigma0", "f1", "r1", "sigma1", "Rmin", "dR", "nR"],
        "labels": ["$f_0$", r"$\sigma_0$", "$f_1$", r"$r_1$", r"$\sigma_1$", "inc", "PA", r"$\Delta RA$", r"$\Delta Dec$"]
    },
    "DoubleRingProfile": {
        "function": DoubleRingProfile,
        "parameters": ["f1", "r1", "sigma1", "f2", "r2", "sigma2", "Rmin", "dR", "nR"],
        "labels": ["$f_1$", r"$r_1$", r"$\sigma_1$", "$f_2$", r"$r_2$", r"$\sigma_2$", "inc", "PA", r"$\Delta RA$", r"$\Delta Dec$"]
    },
    "ThreeRingProfile": {
        "function": ThreeRingProfile,
        "parameters": ["f1", "r1", "sigma1", "f2", "r2", "sigma2", "f3", "r3", "sigma3", "Rmin", "dR", "nR"],
        "labels": [
            "$f_1$", r"$r_1$", r"$\sigma_1$", "$f_2$", r"$r_2$", r"$\sigma_2$", 
            "$f_3$", r"$r_3$", r"$\sigma_3$", "inc", "PA", r"$\Delta RA$", r"$\Delta Dec$"
        ]
    },
    "CentralGauss_ThreeRingProfile": {
        "function": CentralGauss_ThreeRingProfile,
        "parameters": [
            "f0", "sigma0", "f1", "r1", "sigma1", "f2", "r2", "sigma2", "f3", "r3", "sigma3", "Rmin", "dR", "nR"
        ],
        "labels": [
            "$f_0$", r"$\sigma_0$", "$f_1$", r"$r_1$", r"$\sigma_1$", "$f_2$", r"$r_2$", 
            r"$\sigma_2$", "$f_3$", r"$r_3$", r"$\sigma_3$", "inc", "PA", r"$\Delta RA$", r"$\Delta Dec$"
        ]
    },
    "FourRingProfile": {
        "function": FourRingProfile,
        "parameters": [
            "f1", "r1", "sigma1", "f2", "r2", "sigma2", "f3", "r3", "sigma3", "f4", "r4", "sigma4", "Rmin", "dR", "nR"
        ],
        "labels": [
            "$f_1$", r"$r_1$", r"$\sigma_1$", "$f_2$", r"$r_2$", r"$\sigma_2$", 
            "$f_3$", r"$r_3$", r"$\sigma_3$", "$f_4$", r"$r_4$", r"$\sigma_4$", "inc", "PA", r"$\Delta RA$", r"$\Delta Dec$"
        ]
    },
    "FourRingProfile_PointSource": {
        "function": FourRingProfile_PointSource,
        "parameters": [
            "f0", "f1", "r1", "sigma1", "f2", "r2", "sigma2", "f3", "r3", "sigma3", "f4", "r4", "sigma4", "Rmin", "dR", "nR"
        ],
        "labels": [
            "$f_0$", "$f_1$", r"$r_1$", r"$\sigma_1$", "$f_2$", r"$r_2$", r"$\sigma_2$", 
            "$f_3$", r"$r_3$", r"$\sigma_3$", "$f_4$", r"$r_4$", r"$\sigma_4$", "inc", "PA", r"$\Delta RA$", r"$\Delta Dec$"
        ]
    }
}

