# a script to collect the level 0 npz data
# then reduct to level 3 data

import numpy as np 
from functools import partial
import sunRay.statisticalRays as raystat
import multiprocessing as mp


def run_reduction_lv0(arr_eps, arr_alpha,data_dir='./datatmp/'):
    """
    Collect information from a bunch of level0 files
    """
    res_arr_tFWHM = np.zeros([arr_eps.shape[0],arr_alpha.shape[0]])
    res_arr_sizex = np.zeros([arr_eps.shape[0],arr_alpha.shape[0]])
    res_arr_sizey = np.zeros([arr_eps.shape[0],arr_alpha.shape[0]])
    
    idx_eps = 0
    for eps_cur in arr_eps:
        # collect the results one by one
        idx_alpha=0
        for alpha_cur in arr_alpha:
            
            print('Proc : '+str(idx_eps)+', '+str(idx_alpha)+ '   file:'+
                data_dir+'RUN_[eps'+str(np.round(eps_cur,3)) +
                ']_[alpha'+str(np.round(alpha_cur,3))+'].npz')
            
            dataset_cur = np.load(data_dir+'RUN_[eps'+str(np.round(eps_cur,3)) +
            ']_[alpha'+str(np.round(alpha_cur,3))+'].npz')

            #extract the variables from .npz file
            steps_N  =dataset_cur.f.steps_N
            collect_N =dataset_cur.f.collect_N
            photon_N =dataset_cur.f.photon_N
            start_r =dataset_cur.f.start_r
            start_theta =dataset_cur.f.start_theta
            start_phi  =dataset_cur.f.start_phi
            f_ratio  =dataset_cur.f.f_ratio
            epsilon =dataset_cur.f.epsilon 
            anis =dataset_cur.f.anis
            asym =dataset_cur.f.asym
            omega0 = dataset_cur.f.omega0
            freq0 = dataset_cur.f.freq0
            t_collect=dataset_cur.f.t_collect
            tau=dataset_cur.f.tau
            r_vec_collect_local=dataset_cur.f.r_vec_collect_local
            k_vec_collect_local=dataset_cur.f.k_vec_collect_local
            tau_collect_local = dataset_cur.f.tau_collect_local

            
            (duration_cur,sx,sy) = raystat.reduct_lv3(photon_N,
            r_vec_collect_local,k_vec_collect_local,
            t_collect,tau_collect_local,omega0,num_t_bins=85)
            
            res_arr_tFWHM[idx_eps,idx_alpha] = duration_cur
            res_arr_sizex[idx_eps,idx_alpha] = sx
            res_arr_sizey[idx_eps,idx_alpha] = sy

            idx_alpha = idx_alpha + 1

        idx_eps = idx_eps+1

    return (res_arr_tFWHM,res_arr_sizex,res_arr_sizey)


def reduct_single_lv1(eps_cur,alpha_cur,data_dir='../funda/',ang_rot=0):
    
            print('Proc : '+'   file:'+
                data_dir+'RUN_[eps'+str(np.round(eps_cur,5)) +
                ']_[alpha'+str(np.round(alpha_cur,5))+'].lv1.npz')
            
            data_set = np.load(data_dir+'RUN_[eps'+str(np.round(eps_cur,5)) +
                ']_[alpha'+str(np.round(alpha_cur,5))+'].lv1.npz')

            #extract the variables from .npz file
            anis  = data_set.f.anis
            asym  = data_set.f.asym
            collect_N  = data_set.f.collect_N
            epsilon  = data_set.f.epsilon
            f_ratio  = data_set.f.f_ratio
            freq0  = data_set.f.freq0
            k_vec_0  = data_set.f.k_vec_0
            k_vec_stat_avail  = data_set.f.k_vec_stat_avail
            omega0  = data_set.f.omega0
            photon_N  = data_set.f.photon_N
            r_vec_0  = data_set.f.r_vec_0
            r_vec_stat_avail  = data_set.f.r_vec_stat_avail
            start_phi  = data_set.f.start_phi
            start_r  = data_set.f.start_r
            start_theta  = data_set.f.start_theta
            steps_N  = data_set.f.steps_N
            t_reach_stat_avail  = data_set.f.t_reach_stat_avail
            tau_stat_avail  = data_set.f.tau_stat_avail

            r_vec_rot,k_vec_rot = raystat.rotateCoordKX(
                r_vec_stat_avail,k_vec_stat_avail,-ang_rot*np.pi/180)
            r_vec0_rot,k_vec0_rot = raystat.rotateCoordKX(
                r_vec_0,k_vec_0,-ang_rot*np.pi/180)

            (x_im_stat,y_im_stat,t_reach_1au_stat,weights_stat,t_free_stat,idx_for_stat
                )=raystat.ImgXYtEstimate(r_vec_rot,k_vec_rot,t_reach_stat_avail,
                        tau_stat_avail,r_vec0_rot, k_vec0_rot,num_t_bins=150)
            
            flux_this = np.sum(weights_stat)/photon_N

            x_0,y_0=np.mean(r_vec0_rot,axis=1)[0:2]
            
            (xc,yc,sx,sy,err_xc,err_yc,err_sx,err_sy) = raystat.centroidXYFWHM(
                x_im_stat,y_im_stat,weights_stat)

            
            try:
                (t_bin_center,flux_all,xc_all,yc_all,sx_all,
                    sy_all,err_xc_all,err_yc_all,
                    err_sx_all,err_sy_all
                    ) = raystat.variationXYFWHM(x_im_stat,y_im_stat,
                    t_reach_1au_stat,weights_stat,num_t_bins=160)
                
                (FWHM_ab,
                pfit_xc_a,pfit_xc_b,pfit_yc_a,pfit_yc_b,
                pfit_sx_a,pfit_sx_b,pfit_sy_a,pfit_sy_b,offset_xa,
                offset_xb,offset_ya,offset_yb,
                pfit_xc_fwhm,pfit_yc_fwhm,
                pfit_sx_fwhm,pfit_sy_fwhm,offset_x_fwhm,offset_y_fwhm,
                       sx_a,sx_b,sy_a,sy_b)=raystat.OffsetSpeedPhase(
                    t_bin_center,flux_all,
                    xc_all,yc_all,sx_all,sy_all,
                    err_xc_all,err_yc_all,err_sx_all,err_sy_all,
                    x0_all=x_0,y0_all=y_0,offset=True)
            except:
                FWHM_ab=[np.nan]*3
                offset_xa,offset_xb,offset_ya,offset_yb=[np.nan]*4
                (pfit_xc_a,pfit_xc_b,pfit_yc_a,pfit_yc_b,
                pfit_sx_a,pfit_sx_b,pfit_sy_a,pfit_sy_b,
                  pfit_xc_fwhm,pfit_yc_fwhm,
                  pfit_sx_fwhm,pfit_sy_fwhm)=[[np.nan,np.nan]]*12
                

            #try:
            #    fit_res = raystat.fit_biGaussian(t_bin_center,flux_all)
            #    fitted_flux = raystat.biGaussian(t_bin_center,*fit_res)
            #    FWHM_range = raystat.DecayExpTime(t_bin_center,fitted_flux)
            #except:
            #    try:
            #        FWHM_range = raystat.DecayExpTime(t_bin_center,flux_all)
            #    except:
            #        FWHM_range = [0,0]
            #        print('[Warning] FWHM not true')

            try:
                    FWHM_range = raystat.DecayExpTime(t_bin_center,flux_all)
            except:
                    FWHM_range = [np.nan,np.nan]
                    print('[Warning] FWHM not true')

            duration_cur  =  FWHM_range[1]-FWHM_range[0]
            
            return (duration_cur,sx,sy,xc-x_0,yc-y_0,pfit_xc_fwhm[0],pfit_yc_fwhm[0],
                    pfit_sx_fwhm[0],pfit_sy_fwhm[0],offset_xa,offset_xb,offset_ya,offset_yb,
                    np.float32(FWHM_ab[1]-FWHM_ab[0]), np.float32(FWHM_ab[2]-FWHM_ab[1]),
                    np.float32(flux_this))


def run_reduction_lv1(arr_eps,arr_alpha,data_dir,ang_rot=0):
    """
    reduct level 1 data
    """
    res_arr_tFWHM = np.zeros((arr_eps.shape[0],arr_alpha.shape[0]))
    res_arr_sizex = np.zeros((arr_eps.shape[0],arr_alpha.shape[0]))
    res_arr_sizey = np.zeros((arr_eps.shape[0],arr_alpha.shape[0]))
    
    idx_eps = 0
    for eps_cur in arr_eps:
        # collect the results one by one
        idx_alpha=0
        for alpha_cur in arr_alpha:
            reduct_single_lv1(eps_cur,alpha_cur,data_dir=data_dir)

            idx_alpha = idx_alpha + 1

        idx_eps = idx_eps+1

    return (res_arr_tFWHM,res_arr_sizex,res_arr_sizey)

def run_reduction_lv1_parallel(arr_eps,arr_alpha,data_dir,
                               num_process=56,ang_rot=0):
    """
    reduct level 1 data
    """
    def run_single_proc_cur(eps_var,alpha_var):
        return reduct_single_lv1(eps_var,alpha_var,data_dir)
    
    res_arr_tFWHM = np.zeros((arr_eps.shape[0],arr_alpha.shape[0]))
    res_arr_sizex = np.zeros((arr_eps.shape[0],arr_alpha.shape[0]))
    res_arr_sizey = np.zeros((arr_eps.shape[0],arr_alpha.shape[0]))
    res_arr_offset_x = np.zeros((arr_eps.shape[0],arr_alpha.shape[0]))
    res_arr_offset_y = np.zeros((arr_eps.shape[0],arr_alpha.shape[0]))
    res_arr_vx = np.zeros((arr_eps.shape[0],arr_alpha.shape[0]))
    res_arr_vy = np.zeros((arr_eps.shape[0],arr_alpha.shape[0]))
    res_arr_ERx = np.zeros((arr_eps.shape[0],arr_alpha.shape[0]))
    res_arr_ERy = np.zeros((arr_eps.shape[0],arr_alpha.shape[0]))
    res_arr_offset_xa = np.zeros((arr_eps.shape[0],arr_alpha.shape[0]))
    res_arr_offset_xb = np.zeros((arr_eps.shape[0],arr_alpha.shape[0]))
    res_arr_offset_ya = np.zeros((arr_eps.shape[0],arr_alpha.shape[0]))
    res_arr_offset_yb = np.zeros((arr_eps.shape[0],arr_alpha.shape[0]))
    res_arr_tFWHM_a = np.zeros((arr_eps.shape[0],arr_alpha.shape[0]))
    res_arr_tFWHM_b = np.zeros((arr_eps.shape[0],arr_alpha.shape[0]))
    res_arr_flux = np.zeros((arr_eps.shape[0],arr_alpha.shape[0]))
    
    pool = mp.Pool(processes=num_process)
    
    idx_eps = 0
    for eps_cur in arr_eps:
        # parallel calc 
        args_input = [(eps_cur,alpha_cur) for alpha_cur in arr_alpha ]
        results = pool.starmap( partial(reduct_single_lv1,
                        data_dir=data_dir,ang_rot=ang_rot) , args_input)
        # collect the results one by one
        idx_alpha=0
        for res_tuple in results:
            
            res_arr_tFWHM[idx_eps,idx_alpha] = res_tuple[0]
            res_arr_sizex[idx_eps,idx_alpha] = res_tuple[1]
            res_arr_sizey[idx_eps,idx_alpha] = res_tuple[2]
            res_arr_offset_x[idx_eps,idx_alpha] = res_tuple[3]
            res_arr_offset_y[idx_eps,idx_alpha] = res_tuple[4]
            res_arr_vx[idx_eps,idx_alpha] = res_tuple[5]
            res_arr_vy[idx_eps,idx_alpha] = res_tuple[6]
            res_arr_ERx[idx_eps,idx_alpha] = res_tuple[7]
            res_arr_ERy[idx_eps,idx_alpha] = res_tuple[8]
            res_arr_offset_xa[idx_eps,idx_alpha] = res_tuple[9]
            res_arr_offset_xb[idx_eps,idx_alpha] = res_tuple[10]
            res_arr_offset_ya[idx_eps,idx_alpha] = res_tuple[11]
            res_arr_offset_yb[idx_eps,idx_alpha] = res_tuple[12]
            res_arr_tFWHM_a[idx_eps,idx_alpha] = res_tuple[13]
            res_arr_tFWHM_b[idx_eps,idx_alpha] = res_tuple[14]
            res_arr_flux[idx_eps,idx_alpha] = res_tuple[15]

            idx_alpha = idx_alpha + 1

        idx_eps = idx_eps+1
    pool.close()
    pool.join()


    return (res_arr_tFWHM,res_arr_sizex,res_arr_sizey,
            res_arr_offset_x,res_arr_offset_y,res_arr_vx,res_arr_vy,
            res_arr_ERx,res_arr_ERy,res_arr_offset_xa,res_arr_offset_xb,
           res_arr_offset_ya,res_arr_offset_yb,
           res_arr_tFWHM_a,res_arr_tFWHM_b,res_arr_flux)

if __name__ =="__main__":
    
    #arr_eps   = np.linspace(0.03,0.5,20)    
    #arr_alpha = np.linspace(0.05,0.95,20)
    
    #run2
    #arr_eps   = np.linspace(0.03,0.45,80)    
    #arr_alpha = np.linspace(0.05,0.99,80)

    arr_eps   = np.linspace(0.03,0.45,36)    
    arr_alpha = np.linspace(0.05,0.99,36)
    angrot_cur = 0
    pat = 'funda'
    res = run_reduction_lv1_parallel(arr_eps, arr_alpha,'../RUN3/'+pat+'/',ang_rot=angrot_cur)
    np.savez('parsetRUN3.'+pat+'.rot'+str(angrot_cur)+'.v1.npz',res)
    pat = 'harmo'
    res = run_reduction_lv1_parallel(arr_eps, arr_alpha,'../RUN3/'+pat+'/',ang_rot=angrot_cur)
    np.savez('parsetRUN3.'+pat+'.rot'+str(angrot_cur)+'.v1.npz',res)
    
    print(res)
    # use ray for parallel