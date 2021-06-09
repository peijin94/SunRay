# tool set
# f : freq [Hz]
# h : height [R_s]

import sunRay.plasmaFreq as pf 
import sunRay.densityModel as dm  
from scipy.optimize import fsolve 
import torch
import numpy as np
PI = torch.acos(torch.Tensor([-1]))

def R_to_freq(R,f_ratio,ne_r = dm.parkerfit):
    """
    Wave frequency from R
    """
    return pf.omega_pe_r_np(ne_r,torch.Tensor([R])) /2/PI  *f_ratio

def freq_to_R(f,f_ratio,ne_r = dm.parkerfit):
    """
    Starting height for wave frequency f
    """
    f_pe = f/f_ratio
    func  = lambda R : f_pe - (pf.omega_pe_r_np(ne_r,torch.Tensor([R])) /2/PI).numpy()[0]
    R_solution = fsolve(func, 2) # solve the R
    return R_solution # [R_s]

def rect_to_sphere(rx,ry,rz):
    r=0
    theta=0
    phi=0
    return(r,theta,phi)