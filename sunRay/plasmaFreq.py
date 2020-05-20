# relationship of plasma and freqeuncy
import numpy as np
import torch
from sunRay.parameters import dev_u,const_c


PI = torch.acos(torch.Tensor([-1])).to(dev_u)

@torch.enable_grad()
def omega_pe_r(ne_r,r):
    # plasma frequency density relationship
    return 8.93e3* (ne_r(r))**(0.5) * 2 * PI

@torch.enable_grad()
def domega_dxyz(ne_r,r_vec):
    # differential of omegape
    r_vec.requires_grad_(True)
    rr = torch.sqrt(torch.sum(r_vec.pow(2),axis=0))
    omega_pe_xyz = omega_pe_r(ne_r,rr)#.repeat(3,1) # to be size:3*N 
    omega_pe_xyz.backward(torch.ones(omega_pe_xyz.shape).to(dev_u)) # for the gradient of omega
    diff_vec = r_vec.grad.data
    return diff_vec.detach()


def dNe_dxyz(ne_r,r_vec):
    # differential of omegape
    r_vec.requires_grad_(True)
    rr = torch.sqrt(torch.sum(r_vec.pow(2),axis=0))
    Ne_arr = ne_r(rr)#.repeat(3,1) # to be size:3*N 
    Ne_arr.backward(torch.ones(Ne_arr.shape).to(dev_u)) # for the gradient of omega
    diff_vec = r_vec.grad.data
    return diff_vec