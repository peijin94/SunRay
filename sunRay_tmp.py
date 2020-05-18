import numpy as np
import matplotlib.pyplot as plt
from sunRay import densityModel as dm
from sunRay import scattering as scat 
import torch

# initialize
photon_N = 20
start_r = 1.5; # in solar radii
start_theta = 0.1; # in rad
start_phi  = 0; # in rad

f_ratio  = 1.05
density_r = dm.leblanc98 # use leblanc for this calculation 

@torch.enable_grad()
def omega_pe_r(r):
    # plasma frequency density relationship
    return 8.93e3* (density_r(r))**(0.5) * 2 * np.pi

freq0 = f_ratio * omega_pe_r(start_r)/(2*np.pi)

print('----------------------------------')
print('Frequency : '+str(freq0/1e6))

rxx = start_r * np.sin(start_theta) * np.cos(start_phi) * np.ones(photon_N)
ryy = start_r * np.sin(start_theta) * np.sin(start_phi) * np.ones(photon_N)
rzz = start_r * np.cos(start_theta) * np.ones(photon_N)
rr = start_r * np.ones(photon_N)



omega0 = freq0*(2*np.pi)
kc0 = np.sqrt(omega0**2. - omega_pe_r(rr)**2.)

k_mu0  = np.random.uniform(low=0 ,high=1,size=photon_N) # k_z > 0
k_phi0 = np.random.uniform(low=0 ,high= 2*np.pi, size=photon_N) # phi in all dir

kxx_k = kc0 * np.sqrt(1-k_mu0**2.) * np.cos(k_phi0)
kyy_k = kc0 * np.sqrt(1-k_mu0**2.) * np.sin(k_phi0)
kzz_k = kc0 * k_mu0

steps_N  = 100;


r_vec = torch.tensor(np.array([rxx,ryy,rzz]), requires_grad=True)
k_vec = torch.tensor(np.array([kxx_k,kyy_k,kzz_k]))
rr = torch.sqrt(torch.sum(r_vec.pow(2),axis=0))
omega_pe_xyz = omega_pe_r(rr).repeat(3,1) # to be size:3*N 
omega_pe_xyz.backward((omega_pe_xyz*0+1)) # for the gradient of omega
domega_pe_dxyz = r_vec.grad.data

dr_dt = 0

print(domega_pe_dxyz)

steps_N  = 100;

for idx_step in np.arange(steps_N):
    RR = [rxx,ryy,rzz]

