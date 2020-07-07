# collect the saved npz data into an array

import numpy as np 
import sunRay.statisticalRays as raystat

def run_reduction(arr_eps, arr_alpha,data_dir='./datatmp/'):
    res_arr_tFWHM = np.zeros([arr_eps.shape[0],arr_alpha.shape[0]])
    res_arr_sizex = np.zeros([arr_eps.shape[0],arr_alpha.shape[0]])
    res_arr_sizey = np.zeros([arr_eps.shape[0],arr_alpha.shape[0]])
    
    idx_eps = 0
    for eps_cur in arr_eps[0:3]:
        # collect the results one by one
        idx_alpha=0
        for alpha_cur in arr_alpha:
            dataset_cur = np.load(data_dir+'RUN_[eps'+str(np.round(eps_cur,3)) +
            ']_[alpha'+str(np.round(alpha_cur,3))+'].npz')
            #res_arr_tFWHM[idx_eps,idx_alpha] = res_tuple[0]
            #res_arr_sizex[idx_eps,idx_alpha] = res_tuple[1]
            #res_arr_sizey[idx_eps,idx_alpha] = res_tuple[2]

            idx_alpha = idx_alpha + 1

        idx_eps = idx_eps+1
        print('Proc : '+str(idx_eps)+' of '+str(arr_eps.shape[0]))

    return (res_arr_tFWHM,res_arr_sizex,res_arr_sizey)

    
    pass

if __name__ =="__main__":
    
    arr_eps   = np.linspace(0.03,0.5,20)    
    arr_alpha = np.linspace(0.05,0.95,20)

    res = run_reduction(arr_eps, arr_alpha)

    #np.savez('parset.npz',res)
    print(res)
    # use ray for parallel