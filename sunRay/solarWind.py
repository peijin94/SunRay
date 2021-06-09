import torch
import numpy as np
from sunRay import densityModel as dm 
from sunRay.parameters import dev_u,const_c
pi=3.141592653589793238462643383279502884197

def BfieldR(R):
    '''
    input: R [solar radius]
    output B [Gauss]
    From 2 to 15 solar radii, Pätzold et al. (1987)
    obtained from Faraday rotation the global magnetic field variation of
    B  =(6/R^3 + 1.18/R^2)G
    '''
    B_high=6./R^3+1.18/R^2
    #For higher corona we take the function from
    #https://solarscience.msfc.nasa.gov/papers/garyga/PlasmaBeta.pdf
    #Equation 2 B(r)=
    R_s=6.996e10
    #solar radius
    R1=0.5*1e8/R_s
    R2=75.*1e8/R_s
    R3=696.*1e8/R_s
    B_low=2500./(1.+(R-1.)/R1)^3+50./(1.+(R-1.)/R2)^3+1./(1.+(R-1.)/R3)^3
    return B_high+B_low


def ParkerBxyzEarth(r,theta,fi):
    B0=5e-9 * 1e4 # Gauss at 1AU
    r0=215.0
    vsw=4.2e7 # solar wind speed 1AU
    #vsw=1e9 ; 10000 km/s to check radial
    Omega=2*pi/(27.0*24*3600) # solar rotation rate

    #parker spiral in spherical coordinates
    Br = B0*(r0/r)**2
    Btheta =0.
    Bfi=-B0*(r0/r)*(Omega*r0*6.95e10/vsw)*torch.sin(theta)
    # converting to cartesian coordinates
    BxE=Br*torch.sin(theta)*torch.cos(fi) - Bfi*torch.sin(fi) 
    ByE=Br*torch.sin(theta)*torch.sin(fi) + Bfi*torch.cos(fi)
    BzE=Br*torch.cos(theta)
    return BxE,ByE,BzE
    