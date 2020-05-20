import matplotlib.pyplot as plt
import numpy as np
from sunRay import plasmaFreq as pfreq
from sunRay import densityModel as dm
from sunRay import scattering as scat 
from sunRay.parameters import dev_u # use GPU if available

import torch

def showParameters(Ne_r,omega,epsilon):
    rr = torch.linspace(3,100,300)
    ne = Ne_r(rr)
    fig = plt.figure(1)
    ax = plt.gca()
    ax.plot(rr,ne)
    ax.set_xlabel('Heliocentric distance [Rs]')
    ax.set_ylabel('Electron Density [cm-3}]')
    ax.set_yscale('log')

    nu_r = scat.nuScattering(rr,omega,epsilon)
    fig = plt.figure(2)
    ax1 = plt.gca()
    ax1.plot(rr,nu_r)
    ax1.set_xlabel('Heliocentric distance [Rs]')
    ax1.set_ylabel('Scattering nu [cm-3}]')
    ax1.set_yscale('log')

def showResultR(r_vec_cur):
    fig2,axs = plt.subplots(1,3)
    fig2.set_size_inches(18, 6.5)
    for num in range(np.min(np.array([300,r_vec_cur.shape[2]]))-1):
        axs[0].plot(r_vec_cur[:,2,num],r_vec_cur[:,1,num])
        axs[1].plot(r_vec_cur[:,2,num],r_vec_cur[:,0,num])


    axs[0].set_xlabel('Z (Rs)')
    axs[0].set_ylabel('Y (Rs)')
    axs[0].set_aspect('equal')

    axs[1].set_xlabel('Z (Rs)')
    axs[1].set_ylabel('Y (Rs)')
    axs[1].set_aspect('equal')
    
    axs[2].plot(r_vec_cur[-1,0,:],r_vec_cur[-1,1,:],'k.',markersize=0.5)
    axs[2].plot(np.sin(np.linspace(0,np.pi*2,300)),
        np.cos(np.linspace(0,np.pi*2,300)),'r')
    axs[2].set_xlabel('X (Rs)')
    axs[2].set_ylabel('Y (Rs)')
    axs[2].set_aspect('equal')
    #axs[4].plot()


