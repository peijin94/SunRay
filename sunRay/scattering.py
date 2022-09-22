import torch
import numpy as np
from sunRay import densityModel as dm 
from sunRay.parameters import dev_u,const_c
import math



@torch.enable_grad()
def nuScatteringFixed(r,Omega,epsilon=1,dens = dm.leblanc98,dev_u=dev_u):
# basic scattering model
    eps = epsilon
    PI = torch.acos(-torch.ones(1,device=dev_u))
    h_i = 684*1e5/torch.sqrt(dens(r));
    # inner tubulence scale in cm
    q_av = 4/(torch.sqrt(PI)*h_i)
    # average q assuming Gaussian spectrum
    w_pe = dm.f_Ne(dens(r)) * 2 * PI
    
    eps2_L=2e3/(r**0.7)*((r-1.)/r)**2.7
    eps2_L=eps2_L*(eps**2)/6.96e10
    nu_s=PI/8*eps2_L*w_pe**4*const_c/Omega/(Omega**2-w_pe**2)**1.5
    #; this is the fixed eps2/L value  by (good_rinit-1)^0.9*2/good_rinit^(0.82*2./3.+0.333)
    #; from the results of MC simulations
    #return,nu_s
    #; units of rad^2/s
    # nu_s = nuScatterKrupar(r,Omega,epsilon,dens,dev_u=dev_u)
    return nu_s

@torch.enable_grad()
def nuScattering(r,Omega,epsilon,dens = dm.leblanc98,dev_u=dev_u):
# basic scattering model
    PI = torch.acos(-torch.ones(1,device=dev_u))
    h_i = 684*1e5/torch.sqrt(dens(r));
    # inner tubulence scale in cm
    q_av = 4/(torch.sqrt(PI)*h_i)
    # average q assuming Gaussian spectrum
    w_pe = dm.f_Ne(dens(r)) * 2 * PI

    nu_s = nuScatterKrupar(r,Omega,epsilon,dens,dev_u=dev_u)
    return nu_s

@torch.enable_grad()
def nuScatterKrupar(r,Omega,epsilon,dens = dm.leblanc98,dev_u=dev_u):
# scattering power as per Krupar paper
    PI = torch.acos(-torch.ones(1,device=dev_u))
    l_i = 1.0e5* r
    l_0 = 0.23e0* 6.9e10*r**0.82
    # outer turbulence scale
    w_pe = dm.f_Ne(dens(r))* 2 * PI

    qeqs2=8.0 * epsilon**2 /(l_i**(1./3.)*l_0**(2./3.))
    nu_s = (PI*epsilon**2 / 
        (l_i**(1./3.)*l_0**(2./3.))*w_pe**4 
        *const_c/Omega/(Omega**2 -w_pe**2)**1.5)
    return nu_s

@torch.enable_grad()
def nuScatterKrupar2(r,Omega,epsilon,dens=dm.leblanc98,dev_u=dev_u):
# scattering power as per Krupar paper with 
# different outter turbulence scale
    PI = torch.acos(-torch.ones(1,device=dev_u))
    l_i = 684 * 1e5 / torch.sqrt(dens(r))
    l_0 = 0.23*6.9e10*(r-1)
    w_pe = dm.f_Ne(dens(r))* 2 * PI
    nu_s = (PI*epsilon**2 / 
        (l_i**(1./3.)*l_0**(2./3.))*w_pe**4 
        *const_c/Omega/(Omega**2 -w_pe**2)**1.5)
    return nu_s

@torch.enable_grad()
def nuScatterChen(r,Omega,epsilon,dens=dm.leblanc98,dev_u=dev_u):
# scattering power as per the Chen paper
    PI = torch.acos(-torch.ones(1,device=dev_u))
    h_i = 684*1e5/torch.sqrt(dens(r));
    # inner tubulence scale in cm
    q_av = 4/(torch.sqrt(PI)*h_i)
    # average q assuming Gaussian spectrum
    w_pe = dm.f_Ne(dens(r))* 2 * PI

    nu_s  = (2 * PI /8 * epsilon**2 
        *q_av*w_pe**4*const_c/Omega/
        (Omega**2-w_pe**2)**1.5)
    return nu_s

@torch.enable_grad()
def nuScatterSpangler(r,Omega,epsilon,dens=dm.leblanc98,dev_u=dev_u):
# scattering power as per Splangler paper
    PI = torch.acos(-torch.ones(1,device=dev_u))
    h_i = 684*1e5/torch.sqrt(dens(r));
    # inner tubulence scale in cm
    w_pe = dm.f_Ne(dens(r))* 2 * PI
    
    m2cm = 2.15e13
    dn2_n2 = (1.8e10/m2cm)*(10/r)**3.7/(dens(r)**2)

    q_av =12* PI *(2*PI/h_i)**(1./3.)

    nu_s = (2*PI / 8 *dn2_n2 *q_av*w_pe**4
        *const_c/Omega/(Omega^2-w_pe^2))**1.5
    return nu_s

