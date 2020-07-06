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


import sunRay.statisticalRays as raystat

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
                photon_N = 8000, start_r = 1.75, start_theta = 1.e-6/180.0*np.pi,    
                start_phi  = 1.e-6/180.0*np.pi, f_ratio  = 1.1, #ne_r = dm.parkerfit,    
                epsilon = eps_input, anis = alpha_input,  asym = 1.0, Te = 86.0, 
                Scat_include = True, Show_param = True,
                Show_result_k = False, Show_result_r = False,  verb_out = False,
                sphere_gen = False, num_thread =1 )


    (x_im_stat,y_im_stat,t_reach_1au_stat,weights_stat,t_free_stat
        ) = raystat.collectXYtatR(photon_N,r_vec_collect_local,
        k_vec_collect_local,t_collect,tau_collect_local,omega0)

    (xc,yc,sx,sy,err_xc,err_yc,err_sx,err_sy) = raystat.centroidXYFWHM(
        x_im_stat,y_im_stat,weights_stat)

    (t_bin_center,flux_all,xc_all,yc_all,sx_all,sy_all,err_xc_all,err_yc_all,
        err_sx_all,err_sy_all) = raystat.variationXYFWHM(x_im_stat,y_im_stat,
        t_reach_1au_stat,weights_stat,num_t_bins=100)

    FWHM_range = raystat.FWHM(t_bin_center,flux_all)
    duration_cur  =  FWHM_range[1]-FWHM_range[0]

    # TODO : collecting multiple variable

    print(duration_cur)

    return (duration_cur,sx,sy)

def run_parset(arr_eps,arr_alpha, num_process=20):

    res_arr_tFWHM = np.zeros([arr_eps.shape[0],arr_alpha.shape[0]])
    res_arr_sizex = np.zeros([arr_eps.shape[0],arr_alpha.shape[0]])
    res_arr_sizey = np.zeros([arr_eps.shape[0],arr_alpha.shape[0]])
    
    pool = mp.Pool(processes=num_process)
    
    idx_eps = 0
    for eps_cur in arr_eps:
        # parallel calc 
        args_input = [(eps_cur,alpha_cur) for alpha_cur in arr_alpha ]
        results = pool.starmap(run_par, args_input)
        
        # collect the results one by one
        idx_alpha=0
        for res_tuple in results:
            
            res_arr_tFWHM[idx_eps,idx_alpha] = res_tuple[0]
            res_arr_sizex[idx_eps,idx_alpha] = res_tuple[1]
            res_arr_sizey[idx_eps,idx_alpha] = res_tuple[2]

            idx_alpha = idx_alpha + 1

        idx_eps = idx_eps+1

    return (res_arr_tFWHM,res_arr_sizex,res_arr_sizey)

if __name__ =="__main__":

    res = run_parset(arr_eps, arr_alpha)

    print(res)
    # use ray for parallel