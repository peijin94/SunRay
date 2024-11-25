# The density models
import numpy as np
import torch
import math

@torch.enable_grad()
def saito77(r):
    return 1.36e6 * r**(-2.14) + 1.68e8 * r**(-6.13)

@torch.enable_grad()
def leblanc98(r):
    return 3.3e5* r**(-2.)+ 4.1e6 * r**(-4.)+8.0e7* r**(-6.)

@torch.enable_grad()
def parkerfit(r):
    h0=144.0/6.96e5
    h1=20.0/960.
    nc=3e11*torch.exp(-(r-1.0e0)/h1)
    return  4.8e9/r**14. + 3e8/r**6.+1.39e6/r**2.3+nc

@torch.enable_grad()
def dndr_leblanc98(r):
    return -2.*3.3e5* r**(-3.) -4.*4.1e6 * r**(-5.) -6.*8.0e7* r**(-7.)

@torch.enable_grad()
def newkirk(r):
    return 4.2e4*10. **(4.32/r)

@torch.enable_grad()
def dens3dcoronalloop(r_vec, baseNe=parkerfit,dens_ud_ratio = 8,dens_grad_width = 0.10,
                    r_shift = 1.2, r_size = 0.3):
    # r_vec: 3xN
    xx = r_vec[0,:]
    yy = r_vec[1,:]
    zz = r_vec[2,:]
    rr = torch.sqrt(xx**2+yy**2+zz**2)
    rr_shifted = torch.sqrt((xx-r_shift)**2+yy**2+(zz)**2)
    dens = baseNe(rr)
    dens =  dens*(1+(dens_ud_ratio-1)*((1-torch.tanh((rr_shifted-r_size)/dens_grad_width))/2))
    return dens

@torch.enable_grad()
def f_Ne(N_e): 
    # in Hz
    return 8.93e3 * (N_e)**(0.5)

@torch.enable_grad()
def Ne_f(f):
    # in cm-3
    return (f/8.93e3)**2.0

