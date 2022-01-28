import numpy as np

# for processing of lv1 data
def sunray_dataload(fname):
    #data_set  = np.load('../RUN3/funda/RUN_[eps0.27]_[alpha0.72143].lv1.npz')
    data_set  = np.load(fname)
    # collect the data
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
    
    return (anis ,asym ,collect_N ,epsilon ,f_ratio ,freq0 ,
            k_vec_0 ,k_vec_stat_avail ,omega0 ,photon_N ,r_vec_0 ,
            r_vec_stat_avail ,start_phi ,start_r ,start_theta ,steps_N ,
            t_reach_stat_avail ,tau_stat_avail )