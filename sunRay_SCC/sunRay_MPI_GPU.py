# for multiple node cluster
# based on MPI

# TODO: change input, to GPU compatible 

from mpi4py import MPI 
import random
import numpy as np

import sys
sys.path.insert(1, '/gpfs/home/ess/pjzhang/sunray/')

from sunRay import plasmaFreq as pfreq
from sunRay import densityModel as dm
from sunRay import scattering as scat 
from sunRay import showPlot as SP
from sunRay.parameters import c,c_r,R_S  # physics parameters
import torch
import time
from tqdm import tqdm # for processing bar
import sunRay.SunRayRunAnisScat as anisRay
import sunRay.statisticalRays as raystat
import sunRay.tool as st


print(torch.__version__)
print(torch.cuda.is_available())


def sunrayMPIGPU(jobs,f_ratio,r0,data_dir='./'):

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    photon_N=4000000
    for parset_cur in jobs:
        eps_cur,alpha_cur = parset_cur
        anisRay.runRays(steps_N  = -1 , collect_N = 800, t_param = 20.0, 
            photon_N = photon_N,start_r = r0,f_ratio  = f_ratio,
            start_theta = -0/180.0*np.pi, start_phi  = 0/180.0*np.pi,
            epsilon = eps_cur, anis = alpha_cur,
            asym = 1.0, Te = 86.0, Scat_include = True, Show_param = True,
            Show_result_k = False, Show_result_r = False,  verb_out = True,
            sphere_gen = False, num_thread =4, early_cut= True ,
            ignore_down=False,Absorb_include=True,dev_u = torch.device('cuda:'+str(rank%2)),
            save_npz = True, data_dir=data_dir,save_level=1,dk_record=False)
        

if __name__=="__main__": 
    
    # parameter space to explore
    
    jobs = [(0.1,0.15), (0.1,0.8),
            (0.35,0.8),(0.35,0.15)]
    
    f_ratio=1.005
    r0=st.freq_to_R(35e6,f_ratio)[0]
    
    sunrayMPIGPU(jobs, f_ratio=f_ratio, r0=r0,
              data_dir='/gpfs/home/ess/pjzhang/sunray/datatmp/GPU/')

# fundamental 1.75 r=1.1
# f=35.045076 MHz