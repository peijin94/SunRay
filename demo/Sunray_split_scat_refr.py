
import torch
print(torch.__version__)
print(torch.cuda.is_available())
import matplotlib.pyplot as plt
import sunRay.SunRayRunAnisScat as anisRay
import numpy as np

import sunRay.SunRayRunAnisScat as anisRay
import sunRay.statisticalRays as raystat
photon_N=500000

#2.10428659  2.0
#1.75  1.1

#[eps0.102]_[alpha0.15743]
#[eps0.354]_[alpha0.15743]
#[eps0.102]_[alpha0.802]
#[eps0.354]_[alpha0.802]

for pat in ['funda','harmo']:

    if pat=='funda':
        r0=1.75
        fr=1.1

    if pat=='harmo':
        r0=2.10428659
        fr=2.0

    idx=0
    for parset_cur in [(0.102,0.15743),(0.354,0.15743),(0.102,0.802),(0.354,0.802)]:
        eps_cur,alpha_cur = parset_cur
        (steps_N, collect_N,  photon_N, start_r,  start_theta,
                start_phi,  f_ratio, epsilon ,  anis, asym,  omega0, freq0,
                t_collect, tau, r_vec_collect_local,  k_vec_collect_local,
                tau_collect_local,dk_refr_collect, dk_scat_collect
                ) = anisRay.runRays(steps_N  = -1 , collect_N = 150, t_param = 20.0,
                photon_N = photon_N,start_r = r0,f_ratio  = fr,
                start_theta = -0/180.0*np.pi, start_phi  = 0/180.0*np.pi,
                epsilon = eps_cur, anis = alpha_cur,
                asym = 1.0, Te = 86.0, Scat_include = True, Show_param = True,
                Show_result_k = False, Show_result_r = False,  verb_out = True,
                sphere_gen = False, num_thread =2, early_cut= True ,
                ignore_down=True,Absorb_include=True,dev_u = torch.device('cuda:'+str(3-idx)),
                save_npz = True, data_dir='../tmpRUNvec/'+pat+'/',save_level=1)

        idx=idx+1
        # run in serial
