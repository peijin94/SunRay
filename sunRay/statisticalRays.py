#---------------------------------------------
#  to calculate the statistical result of the rays
#  by : Peijin
#  date : 2020-6-9
#---------------------------------------------

import numpy as np 
import torch
from sunRay.parameters import c_r


def collectXYt1AU(photon_N,r_vec_collect_local,k_vec_collect_local,t_collect,tau):

    find_small_1e3 = lambda arr:  np.sort(arr)[int(photon_N*1e-3)]

    # collect the photons
    r_vec_end = r_vec_collect_local[-1,:,:].reshape(3,-1)
    k_vec_end = k_vec_collect_local[-1,:,:].reshape(3,-1)
    rr_end = np.sqrt(np.sum(r_vec_end**2,axis=0))
    kk_end = np.sqrt(np.sum(k_vec_end**2,axis=0))

    # most of the photons passed this range
    r_get = find_small_1e3(rr_end)
    kx_end,ky_end,kz_end = k_vec_end[0,:],k_vec_end[1,:],k_vec_end[2,:]

    idx_for_stat = np.where( (rr_end>(r_get-0.1)) & 
                         (kz_end/kk_end>0.9) & 
                         (kz_end/kk_end<1.0) )


    x_im_stat = np.zeros(idx_for_stat[0].shape)
    y_im_stat = np.zeros(idx_for_stat[0].shape)
    t_reach_stat = np.zeros(idx_for_stat[0].shape)
    tau_stat = np.zeros(idx_for_stat[0].shape)


    idx_tmp = 0
    for idx_cur in idx_for_stat[0]:
        # for all rays do the collect:
        r_vec_tmp = r_vec_collect_local[:,:,idx_cur]
        rr_tmp = np.sqrt(np.sum(r_vec_tmp**2,axis=1))
        idx_r_reach = np.argmin(np.abs(rr_tmp-r_get))
        # linear estimation of the [t,r,k] at r_get
        t_reach_tmp = (t_collect[idx_r_reach-1] + 
            (t_collect[idx_r_reach]-t_collect[idx_r_reach-1]) *
            (r_get-rr_tmp[idx_r_reach-1]) /
            (rr_tmp[idx_r_reach]-rr_tmp[idx_r_reach-1]) )

        r_vec_reach_tmp = (r_vec_collect_local[idx_r_reach-1,:,idx_cur] + 
            (r_vec_collect_local[idx_r_reach,:,idx_cur]-
            r_vec_collect_local[idx_r_reach-1,:,idx_cur]) *
            (r_get-rr_tmp[idx_r_reach-1]) /
            (rr_tmp[idx_r_reach]-rr_tmp[idx_r_reach-1]) )

        k_vec_reach_tmp = (k_vec_collect_local[idx_r_reach-1,:,idx_cur] + 
            (k_vec_collect_local[idx_r_reach,:,idx_cur]-
            k_vec_collect_local[idx_r_reach-1,:,idx_cur]) *
            (r_get-rr_tmp[idx_r_reach-1]) /
            (rr_tmp[idx_r_reach]-rr_tmp[idx_r_reach-1]) )

        t_reach_stat[idx_tmp] = t_reach_tmp

        kk_tmp = np.sqrt(np.sum(k_vec_reach_tmp**2))
        kx_tmp = k_vec_reach_tmp[0]
        ky_tmp = k_vec_reach_tmp[1]
        kz_tmp = k_vec_reach_tmp[2]

        # use Delta R as free path integral
        r_free_tmp_a = np.sqrt(np.sum((r_vec_reach_tmp - r_vec_collect_local[0,:,idx_cur])**2))
        # use t*c as free path integral
        r_free_tmp_b = t_reach_tmp*c_r

        x_im_stat[idx_tmp] = r_vec_reach_tmp[0] - r_free_tmp_a*kx_tmp/kk_tmp
        y_im_stat[idx_tmp] = r_vec_reach_tmp[1] - r_free_tmp_a*ky_tmp/kk_tmp

        
        tau_stat[idx_tmp] = tau[idx_cur]

        idx_tmp = idx_tmp+1    

    weights_stat = np.exp(-tau_stat)

    return (x_im_stat,y_im_stat,t_reach_stat,weights_stat)


