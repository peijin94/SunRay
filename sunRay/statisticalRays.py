#---------------------------------------------
#  to calculate the statistical result of the rays
#  by : Peijin
#  date : 2020-6-9
#---------------------------------------------

import numpy as np 
import torch
from sunRay.parameters import c_r
from sunRay import plasmaFreq as pfreq
from sunRay import densityModel as dm
from scipy import integrate
from scipy.optimize import curve_fit


def collectXYt1AU(photon_N,r_vec_collect_local,k_vec_collect_local,t_collect,tau,omega0):

# this collection method deprecated

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

    t_delay = integrate.quad(lambda x: (1/(c_r*np.sqrt(1.0-pfreq.omega_pe_r_np(dm.leblanc98,x)/omega0)) ) ,r_get,215 )[0]
    t_reach_1au_stat = t_reach_stat + t_delay

    return (x_im_stat,y_im_stat,t_reach_1au_stat,weights_stat)


def collectXYtatR(photon_N,r_vec_collect_local,k_vec_collect_local,
            t_collect,tau_collect,omega0):
    
    find_small_1e3 = lambda arr:  np.sort(arr)[int(photon_N*1e-3)]

    # collect the photons
    r_vec_end = r_vec_collect_local[-1,:,:].reshape(3,-1)
    k_vec_end = k_vec_collect_local[-1,:,:].reshape(3,-1)
    rr_end = np.sqrt(np.sum(r_vec_end**2,axis=0))
    kk_end = np.sqrt(np.sum(k_vec_end**2,axis=0))

    r_vec_start = r_vec_collect_local[0,:,:].reshape(3,-1)
    rr_start = np.sqrt(np.sum(r_vec_start**2,axis=0))   

    # most of the photons passed this range
    r_get = np.min([find_small_1e3(rr_end),205])
    kx_end,ky_end,kz_end = k_vec_end[0,:],k_vec_end[1,:],k_vec_end[2,:]
    rx_end,ry_end,rz_end = k_vec_end[0,:],k_vec_end[1,:],k_vec_end[2,:]

    rr_end = np.nan_to_num(rr_end)
    idx_available = np.where(rr_end>(r_get-0.1))

    x_im_stat_avail = np.zeros(idx_available[0].shape)
    y_im_stat_avail = np.zeros(idx_available[0].shape)
    t_reach_stat_avail = np.zeros(idx_available[0].shape)
    tau_stat_avail = np.zeros(idx_available[0].shape)
    t_free_stat_avail = np.zeros(idx_available[0].shape)
    tau_stat_avail = np.zeros(idx_available[0].shape)

    kk_stat_avail = np.zeros(idx_available[0].shape)
    kz_stat_avail = np.zeros(idx_available[0].shape)
    
    idx_tmp = 0
    for idx_cur in idx_available[0]:
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

        # linear estimation of tau
        tau_tmp = (tau_collect[idx_r_reach-1,idx_cur] + 
            (tau_collect[idx_r_reach,idx_cur]-
            tau_collect[idx_r_reach-1,idx_cur]) *
            (r_get-rr_tmp[idx_r_reach-1]) /
            (rr_tmp[idx_r_reach]-rr_tmp[idx_r_reach-1]) )  
 


        kk_tmp = np.sqrt(np.sum(k_vec_reach_tmp**2))
        kx_tmp = k_vec_reach_tmp[0]
        ky_tmp = k_vec_reach_tmp[1]
        kz_tmp = k_vec_reach_tmp[2]

        # use Delta R as free path integral
        r_free_tmp_a = np.sqrt(np.sum((r_vec_reach_tmp - r_vec_collect_local[0,:,idx_cur])**2))
        
        # use t*c as free path integral
        r_free_tmp_b = t_reach_tmp*c_r

        t_reach_stat_avail[idx_tmp] = t_reach_tmp- r_free_tmp_a/c_r
        t_free_stat_avail[idx_tmp] = r_free_tmp_a/c_r

        x_im_stat_avail[idx_tmp] = r_vec_reach_tmp[0] - r_free_tmp_a*kx_tmp/kk_tmp
        y_im_stat_avail[idx_tmp] = r_vec_reach_tmp[1] - r_free_tmp_a*ky_tmp/kk_tmp

        tau_stat_avail[idx_tmp] = tau_tmp

        kk_stat_avail[idx_tmp] = kk_tmp
        kz_stat_avail[idx_tmp] = kz_tmp

        idx_tmp = idx_tmp+1  

    idx_for_stat = np.where((kz_stat_avail/kk_stat_avail<1.00) & 
                            (kz_stat_avail/kk_stat_avail>0.90))


    x_im_stat = np.zeros(idx_for_stat[0].shape)
    y_im_stat = np.zeros(idx_for_stat[0].shape)
    t_reach_stat = np.zeros(idx_for_stat[0].shape)
    tau_stat = np.zeros(idx_for_stat[0].shape)
    t_free_stat = np.zeros(idx_for_stat[0].shape)

    idx_tmp = 0
    for idx_cur in idx_for_stat[0]:
        
        x_im_stat[idx_tmp] = x_im_stat_avail[idx_cur]
        y_im_stat[idx_tmp] = y_im_stat_avail[idx_cur]
        t_reach_stat[idx_tmp] = t_reach_stat_avail[idx_cur]
        tau_stat[idx_tmp] = tau_stat_avail[idx_cur]
        t_free_stat[idx_tmp] = t_free_stat_avail[idx_cur]

        idx_tmp = idx_tmp+1  

    weights_stat = np.exp(-tau_stat)

    #t_delay = integrate.quad(lambda x: (1/(c_r*np.sqrt(1.0-pfreq.omega_pe_r_np(dm.leblanc98,x)/omega0)) ) ,r_get,215 )[0]
    
    t_delay = integrate.quad(lambda x: (1/(c_r*np.sqrt(1.0-pfreq.omega_pe_r_np(dm.leblanc98,x)/omega0)) ),
                 np.min(rr_start),r_get,full_output=1 )[0]

    #print(t_delay)
    #print(np.mean(t_free_stat))
    #print(r_get+1)
    #print(np.mean(t_reach_stat))

    t_reach_1au_stat = t_reach_stat

    return (x_im_stat,y_im_stat,t_reach_1au_stat,weights_stat,t_free_stat)




def centroidXYFWHM(x,y,weights_data=1):
    """
        calculate the weighted centroid and FWHM from scattered points
        x: X position
        y: Y position
        weights_data: default is 1 for all points
    """
    xc = (np.mean(x*weights_data) / 
                    np.mean(weights_data))
    yc = (np.mean(y*weights_data) / 
                    np.mean(weights_data))
    
    sx=np.sqrt(np.mean(weights_data*(x-xc)**2)/
                np.mean(weights_data))*2.355
    sy=np.sqrt(np.mean(weights_data*(y-yc)**2)/
                np.mean(weights_data))*2.355

    err_xc = sx/(np.sqrt(np.prod(x.shape))+1e-8)/2.355
    err_yc = sy/(np.sqrt(np.prod(y.shape))+1e-8)/2.355

    err_sx = sx*np.sqrt(2)/(np.sqrt(np.prod(x.shape))+1e-8)
    err_sy = sy*np.sqrt(2)/(np.sqrt(np.prod(y.shape))+1e-8)
    
    return (xc,yc,sx,sy,err_xc,err_yc,err_sx,err_sy)

def variationXYFWHM(x_data,y_data,t_data,weights_data,t_step = 0.05,num_t_bins=-1):
    """
        The variation of the XY positions with a [t_step] cadence
        
        input vars
            [-] : num_t_bins : -1 if according to the [t_step]
                    positive integer to override t_step
    """

    x_im_stat = x_data
    y_im_stat = y_data
    

    t_reach_1au_stat = t_data
    weights_stat = weights_data

    lower_t_lim = np.sort(t_reach_1au_stat)[int(t_reach_1au_stat.shape[0]*1e-3)]-0.5
    upper_t_lim = np.sort(t_reach_1au_stat)[int(t_reach_1au_stat.shape[0]*(1-1e-3))]+0.5

    if num_t_bins<0:
        num_t_bins = int((upper_t_lim-lower_t_lim)/t_step)

    t_bins = np.linspace(lower_t_lim,upper_t_lim,num_t_bins)
    t_bin_center = (t_bins[0:-1]+t_bins[1:])/2


    flux_all = np.zeros(t_bin_center.shape)
    xc_all = np.zeros(t_bin_center.shape)
    yc_all = np.zeros(t_bin_center.shape)
    sx_all = np.zeros(t_bin_center.shape)
    sy_all = np.zeros(t_bin_center.shape)
    err_xc_all = np.zeros(t_bin_center.shape)
    err_yc_all = np.zeros(t_bin_center.shape)
    err_sx_all = np.zeros(t_bin_center.shape)
    err_sy_all = np.zeros(t_bin_center.shape)

    idx_cur = 0
    for idx_t_bin in np.arange(len(t_bin_center)):

        idx_in_t_range = np.where((t_reach_1au_stat>t_bins[idx_t_bin]) 
                                & (t_reach_1au_stat<t_bins[idx_t_bin+1]))
        #print(str(t_bins[idx_t_bin])+" [-] "+str((idx_in_t_range[0].shape)))

        if True:#(idx_in_t_range[0].shape[0])>2:

            x_im_in_t_range = x_im_stat[idx_in_t_range]
            y_im_in_t_range = y_im_stat[idx_in_t_range]
            weights_in_t_range = weights_stat[idx_in_t_range]
            #print(weights_in_t_range)

            # collect the variation of xc yc sx sy
            ( xc_all[idx_cur],yc_all[idx_cur],sx_all[idx_cur],sy_all[idx_cur],
                err_xc_all[idx_cur],err_yc_all[idx_cur],
                err_sx_all[idx_cur],err_sy_all[idx_cur]
                ) = centroidXYFWHM(x_im_in_t_range,y_im_in_t_range,weights_in_t_range)
            flux_all[idx_cur] = np.sum(weights_in_t_range*np.ones(x_im_in_t_range.shape))
            #flux_all[idx_cur] = np.sum(1.0*np.ones(x_im_in_t_range.shape))
        

        idx_cur = idx_cur + 1

    return (t_bin_center,flux_all,xc_all,yc_all,
        sx_all,sy_all,err_xc_all,err_yc_all,err_sx_all,err_sy_all)


def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def FWHM(x, y):
    """
    Determine the FWHM position [x] of a distribution [y]
    """
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]


def fit_biGaussian(x,y):
    """
    Derive the best fit curve for the flux-time distribution
    """
    popt, pcov = curve_fit(biGaussian,x,y,p0=(2.,1.,1.,2.))
    return popt


def biGaussian(x,x0,sig1,sig2,A):
    # combine 2 gaussian:
    return A*np.exp(-0.5*((x-x0)/
        (sig1*(x<x0)+sig2*(x>=x0)))**2)
    