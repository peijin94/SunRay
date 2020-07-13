# a script to collect the level 0 npz data
# then reduct to level 3 data

import numpy as np 
import sunRay.statisticalRays as raystat

def run_reduction(arr_eps, arr_alpha,data_dir='./datatmp/'):
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

    
    pass

if __name__ =="__main__":
    
    arr_eps   = np.linspace(0.03,0.5,20)    
    arr_alpha = np.linspace(0.05,0.95,20)

    res = run_reduction(arr_eps, arr_alpha)

    np.savez('parset.npz',res)
    print(res)
    # use ray for parallel