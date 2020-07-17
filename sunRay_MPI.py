# for multiple node cluster
# based on MPI

from mpi4py import MPI 
import random
import numpy as np

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


def sunrayMPI(arr_eps,arr_alpha,dev_u,photon_N = 200000,
        data_dir='./datatmp/funda',save_npz=True, start_r = 1.75, 
                start_theta = 1.e-6/180.0*np.pi, start_phi  = 1.e-6/180.0*np.pi, f_ratio  = 1.1,  
                asym = 1.0, Te = 86.0,  Scat_include = True, Show_param = True,
                Show_result_k = False, Show_result_r = False,  verb_out = False,
                sphere_gen = False, num_thread =2 , early_cut= True, save_level=1):

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    size_eps,size_alpha =  arr_eps.shape[0],arr_alpha.shape[0]
    numjobs=size_eps*size_alpha

    job_content = [] # the collection of parameters [eps,anis]
    for eps_cur in arr_eps:
        for alpha_cur in arr_alpha:
            job_content.append((eps_cur,alpha_cur))

    # arrange the works and jobs
    if rank==0:
        # this is head worker
        # jobs are arranged by this worker

        job_all_idx =list(range(numjobs))  
        random.shuffle(job_all_idx)
        # shuffle the job index to make all workers equal

    else:
        job_all_idx=None
    
    job_all_idx = comm.bcast(job_all_idx,root=0)
    
    njob_per_worker = int(numjobs/size) 
    # the number of jobs should be a multiple of the NumProcess[MPI]
    
    this_worker_job = [job_all_idx[x] for x in range(rank*njob_per_worker, (rank+1)*njob_per_worker)]
    
    # map the index to parameterset [eps,anis]
    work_content = [job_content[x] for x in this_worker_job ]

    for a_piece_of_work in work_content:
        print('==============')
        anisRay.runRays(steps_N  = -1 , collect_N = 360, t_param = 20.0, 
                photon_N = photon_N, start_r = 1.75, start_theta = 1.e-6/180.0*np.pi,    
                start_phi  = 1.e-6/180.0*np.pi, f_ratio  = 1.1, #ne_r = dm.parkerfit,    
                epsilon = a_piece_of_work[0], anis = a_piece_of_work[1],  asym = 1.0, Te = 86.0, 
                Scat_include = True, Show_param = True,
                Show_result_k = False, Show_result_r = False,  verb_out = False,
                sphere_gen = False, num_thread =2 , early_cut= True,
                dev_u= dev_u,save_npz=save_npz,data_dir=data_dir,save_level=1)

        print("[Info] : parameter",[np.round(x,5) for x in  a_piece_of_work])
        

if __name__=="__main__": 
    
    # parameter space to explore
    arr_eps   = np.linspace(0.03,0.45,80)    
    arr_alpha = np.linspace(0.05,0.99,80)
    
    dev_u = torch.device('cpu') 
    sunrayMPI(arr_eps,arr_alpha,dev_u=dev_u,photon_N = 250000
            ,data_dir='/gpfs/home/ess/pjzhang/sunray/datatmp/funda/')

# fundamental 1.75 r=1.1
# f=35.045076 MHz