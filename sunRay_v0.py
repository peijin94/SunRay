import numpy as np
import matplotlib.pyplot as plt
from sunRay import plasmaFreq as pfreq
from sunRay import densityModel as dm
from sunRay import scattering as scat 
from sunRay.parameters import dev_u
import torch

# initialize
steps_N  = 3000;       # number of the step
collect_N = 40;       # number of recorded step
t_param = 20.0;       # parameter of t step length
# larger t_parm corresponding to smaller dt

photon_N = 2         # number of photon
start_r = 1.5;        # in solar radii
start_theta = 0.1;    # in rad
start_phi  = 0;       # in rad

R_S = 6.96e10         # the radius of the sun 
c   = 2.998e10        # speed of light
c_r = c/R_S           # [t]

f_ratio  = 1.05       # f/f_pe
ne_r = dm.leblanc98   # use leblanc for this calculation 
epsilon = 0.1         # fluctuation scale
anis = 0.1            # the anisotropic parameter
asym = 1.0            # asymetric scale

Te = 86.0             # eV temperature in eV

nu_e = 2.91e-6*ne_r(start_r)*20./Te**1.5

# frequency of the wave
freq0 = f_ratio * pfreq.omega_pe_r(ne_r,start_r)/(2*np.pi)
print('----------------------------------')
print('Frequency : '+str(freq0/1e6)[0:6]+'MHz')
print('Compute with : '+str(dev_u))
print('----------------------------------')

PI = torch.acos(torch.Tensor([-1])).to(dev_u)
freq0 = torch.Tensor([freq0]).to(dev_u)

# position of the photons
rxx = start_r * np.sin(start_theta) * np.cos(start_phi) * np.ones(photon_N)
ryy = start_r * np.sin(start_theta) * np.sin(start_phi) * np.ones(photon_N)
rzz = start_r * np.cos(start_theta) * np.ones(photon_N)
rr = start_r * torch.ones(photon_N).to(dev_u)

omega0 = freq0*(2*PI)
nu_s0 = scat.nuScattering(rr,omega0,epsilon,ne_r)

# wave-vector of the photons
kc0 = torch.sqrt(omega0**2. - pfreq.omega_pe_r(ne_r,rr)**2.)
k_mu0  = torch.Tensor(np.random.uniform(low=0 ,high=1,size=photon_N)).to(dev_u) # k_z > 0
k_phi0 = torch.Tensor(np.random.uniform(low=0 ,high= 2*np.pi, size=photon_N)).to(dev_u) # phi in all dir

kxx_k = kc0 * torch.sqrt(1-k_mu0**2.) * torch.cos(k_phi0)
kyy_k = kc0 * torch.sqrt(1-k_mu0**2.) * torch.sin(k_phi0)
kzz_k = kc0 * k_mu0

r_vec = torch.Tensor(np.array([rxx,ryy,rzz])).to(dev_u)
k_vec = torch.stack((kxx_k,kyy_k,kzz_k),0).to(dev_u)
kc = torch.sqrt(torch.sum(k_vec.pow(2),axis=0))

# Detach from the previous compute graph 
# before record steps for diff
domega_pe_dxyz = pfreq.domega_dxyz(ne_r,r_vec.detach())

Exp_size = 1.25*30./(freq0/1e6)
dt0 = 0.01*Exp_size/c_r
tau = torch.zeros(rr.shape).to(dev_u)

# collect the variables of the simulation
collectPoints = np.round(np.linspace(0,steps_N-1,collect_N))
r_vec_collect = torch.zeros(collect_N,3,photon_N).to(dev_u)
k_vec_collect = torch.zeros(collect_N,3,photon_N).to(dev_u)
t_collect = torch.zeros(collect_N).to(dev_u)
idx_collect  =  0
t_current = 0

# the big loop
for idx_step in np.arange(steps_N):
    
    # dispersion relation reform
    omega = torch.sqrt(pfreq.omega_pe_r(ne_r,rr)**2 + kc**2)

    nu_s = scat.nuScattering(rr,omega,epsilon,ne_r)
    nu_s = nu_s*(nu_s<nu_s0)+nu_s0*(nu_s>=nu_s0) # use the smaller nu_s

    domega_pe_dxyz = pfreq.domega_dxyz(ne_r,r_vec.detach())
    domega_pe_dr = torch.sqrt(torch.sum(domega_pe_dxyz.pow(2),axis=0))
    with torch.no_grad():
        # component of r and k vector at current step
        rx_cur,ry_cur,rz_cur = r_vec[0,:],r_vec[1,:],r_vec[2,:]
        kx_cur,ky_cur,kz_cur = k_vec[0,:],k_vec[1,:],k_vec[2,:]
        rr_cur = torch.sqrt(torch.sum(k_vec.pow(2),axis=0))
        kc_cur = torch.sqrt(torch.sum(k_vec.pow(2),axis=0))

        # dynamic time step
        dt_ref = torch.min(torch.abs(kc/ (domega_pe_dr*c_r)/t_param)) # t step
        dt_dr  = torch.min(rr/omega0*kc)/t_param
        dt = torch.Tensor([np.min([1.0/torch.max(nu_s),dt_ref,dt_dr,dt0])]).to(dev_u)

    
        g0 = torch.sqrt(nu_s*kc**2)

        # position step vec
        dr_vec = c_r/omega.repeat(3,1)*k_vec

        # random vec for wave scattering  # [3*N] normal distribution
        W_vec = torch.randn(r_vec.shape).to(dev_u) * torch.sqrt(dt) 
        Wx,Wy,Wz = W_vec[0,:],W_vec[1,:],W_vec[2,:]

        # photon position in spherical coordinates
        # (rx,ry,rz) is the direction of anisotropic tubulence
        fi = torch.atan(ry_cur/rx_cur)
        costheta = rz_cur/rr_cur
        sintheta = torch.sqrt(1-costheta**2)

        # rotate the k vec into the r-z coordinate
        kcx = - kx_cur*torch.sin(fi) + ky_cur*torch.cos(fi) 
        kcy = (- kx_cur*costheta*torch.cos(fi) 
            - ky_cur*costheta*torch.sin(fi) + kz_cur*sintheta) 
        kcz = (  kx_cur*sintheta*torch.cos(fi) 
            + ky_cur*sintheta*torch.sin(fi) + kz_cur*costheta)

        kw     =  Wx*kcx+Wy*kcy+Wz*kcz*anis
        Akc    = torch.sqrt(kcx*kcx+kcy*kcy+kcz*kcz*anis**2)
        z_asym = (asym*(kcz > 0.0) + (2.0-asym)*(kcz < 0.))*(kc_cur/Akc)**2
    
        A_perp = (nu_s*z_asym* kc_cur /Akc**3 *
            (-(1+anis**2)*Akc**2+3*anis**2 *(anis**2-1)*kcz**2) *anis)
        A_par  = (nu_s*z_asym* kc_cur /Akc**3 *
            ((-3*anis**4+anis**2)*Akc**2+3*anis**4 * (anis**2-1)*kcz**2)*anis)
        A_g0   = g0*torch.sqrt(z_asym*anis)
    
        kcx=kcx + A_perp*kcx*dt + A_g0*(Wx-kcx*kw/Akc**2)
        kcy=kcy + A_perp*kcy*dt + A_g0*(Wy-kcy*kw/Akc**2)
        kcz=kcz + A_par *kcz*dt + A_g0*(Wz-kcz*kw*anis/Akc**2)*anis


        # rotate back to normal coordinate
        kx_cur = (-kcx*torch.sin(fi) 
            -kcy*costheta*torch.cos(fi) +kcz*sintheta*torch.cos(fi) )
        ky_cur = ( kcx*torch.cos(fi) 
            -kcy*costheta*torch.sin(fi) +kcz*sintheta*torch.sin(fi) )
        kz_cur =  kcy*sintheta+kcz*costheta


        r_vec = torch.stack((rx_cur,ry_cur,rz_cur),0)
        k_vec = torch.stack((kx_cur,ky_cur,kz_cur),0)

        # re-normalize  # to keep |k_vec| stable
        kc_norm = torch.sqrt(kx_cur**2+ky_cur**2+kz_cur**2)
        k_vec = k_vec * kc_cur.repeat(3,1)/ kc_norm.repeat(3,1)

        # k step forward  # refraction
        dk_xyz_dt = ((pfreq.omega_pe_r(ne_r,rr_cur)/omega).repeat(3,1) 
                * domega_pe_dxyz * c_r * r_vec/rr_cur.repeat(3,1)) 
        k_vec = k_vec - dk_xyz_dt * dt

        # r step forward
        r_vec = r_vec + dr_vec * dt

        # re-normalize  # to keep omega stable
        kc_refresh = (torch.sqrt(omega**2-pfreq.omega_pe_r(ne_r,rr_cur)**2)
                /torch.sqrt(kx_cur**2+ky_cur**2+kz_cur**2))
        k_vec = k_vec * kc_refresh

        nu_e = (2.91e-6*ne_r(rr_cur)*20./Te**1.5
            *pfreq.omega_pe_r(ne_r, rr_cur)**2/omega**2)

        tau = tau + nu_e*dt
        
    t_current = t_current + dt
    if idx_step in collectPoints:
        t_collect[idx_collect] = t_current
        r_vec_collect[idx_collect,:,:] = r_vec
        k_vec_collect[idx_collect,:,:] = k_vec
        idx_collect = idx_collect +1

t_collect_local = t_collect.cpu().data.numpy()
r_vec_collect_local  = r_vec_collect.cpu().data.numpy()

#plt.figure(1)
#plt.plot(t_collect_local, r_vec_collect_local[:,0,0])
#plt.figure(2)
#plt.plot( r_vec_collect_local[:,0,0], r_vec_collect_local[:,1,0])

plt.show()

print(t_collect_local)