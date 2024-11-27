import numpy as np
import scipy.constants as sc

# bmin and bmaj in arcsec
# nu in Hz

def Jybeam_to_Jysr(data,bmin,bmaj):
    # [Jy/beam] to [Jy/sr] conversion
    omega = np.radians(bmin / 3600.)
    omega *= np.radians(bmaj / 3600.)
    beam_area = np.pi * omega / 4. / np.log(2.)
    return data / beam_area

def Jysr_to_Tb(data, nu):
    # [Jy/sr] to [K] conversion using the full Planck law.
    Tb = sc.h * nu / (sc.k * np.log(1+(2 * sc.h * nu**3) / (1e-26*data * sc.c**2)))
    return Tb

def Jysr_to_Tb_err(data, err_data, nu):
    # [Jy/sr] to [K] conversion using the full Planck law.
    Tb = sc.h * nu / (sc.k * np.log(1+(2 * sc.h * nu**3) / (1e-26*data * sc.c**2)))
    dTb =  (2 * sc.h**2 * nu**4) / (sc.k * (1e-26*data) * ((1e-26*data) * sc.c**2 + 2 * sc.h * nu**3) * (np.log(1 + (2 * sc.h * nu**3)/((1e-26*data) * sc.c**2)))**2) * (1e-26*err_data)
    # For some reason it gives inf if not using the parenthesis around 1e-26*data 
    return Tb, dTb    

def Jysr_to_Tb_RJ(data, nu):
    # [Jy/sr] to [K] conversion using the Rayleigh-Jeans approximation.
    Tb_RJ = sc.c**2 * 1e-26 * data / (2 * sc.k * nu**2)
    return Tb_RJ

def Jysr_to_Tb_RJ_err(data, err_data, nu):
    # [Jy/sr] to [K] conversion using the Rayleigh-Jeans approximation.
    Tb_RJ = sc.c**2 * 1e-26 * data / (2 * sc.k * nu**2)
    dTb_RJ = sc.c**2 * 1e-26 * err_data / (2 * sc.k * nu**2)
    return Tb_RJ, dTb_RJ