# updated 2020-06-29
# The script to do the ray tracing


import numpy as np
from sunRay import plasmaFreq as pfreq
from sunRay import densityModel as dm
from sunRay import scattering as scat 
from sunRay import showPlot as SP
from sunRay.parameters import c,c_r,R_S  # physics parameters
from sunRay.parameters import dev_u  # computation device
import torch
import time
from tqdm import tqdm # for processing bar
import sunRay.SunRayRunAnisScat as anisRay

# torch thread dosen't work
# njit dosen't work

import multiprocessing as mp

arr_eps   = np.linspace(0.03,0.5,30)
arr_alpha = np.linspace(0.05,0.95,30)

def run_par(eps_input, alpha_input):

    (steps_N  ,  collect_N,  photon_N, start_r,  start_theta, 
            start_phi,  f_ratio, epsilon ,  anis, asym,  omega0, freq0, 
            t_collect, tau, r_vec_collect_local,  k_vec_collect_local,  tau_collect_local
            ) = anisRay.runRays(steps_N  = -1 , collect_N = 180, t_param = 20.0, 
                photon_N = 20000, start_r = 1.75, start_theta = 0./180.0*np.pi,    
                start_phi  = 0./180.0*np.pi, f_ratio  = 1.1, #ne_r = dm.parkerfit,    
                epsilon = eps_input, anis = alpha_input, asym = 1.0, Te = 86.0, 
                Scat_include = True, Show_param = True,
                Show_result_k = False, Show_result_r = False,  verb_out = False,
                sphere_gen = False, num_thread =1 )

    return 0

def run_parset(arr_eps,arr_alpha, num_process=16):
    
    pool = mp.Pool(processes=num_process)
    
    for eps_cur in arr_eps: 
        args_input = [(eps_cur,alpha_cur) for alpha_cur in arr_alpha ]
        #result = pool.starmap(run_par, args_input)

    return 0

if __name__ =="__main__":

    res = run_parset(arr_eps, arr_alpha)
    # use ray for parallel