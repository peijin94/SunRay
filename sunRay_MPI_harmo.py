import sunRay_MPI
import torch
import numpy as np

if __name__=="__main__": 
    
    # parameter space to explore
    arr_eps   = np.linspace(0.03,0.45,80)    
    arr_alpha = np.linspace(0.05,0.99,80)
    
    dev_u = torch.device('cpu') 
    sunRay_MPI.sunrayMPI(arr_eps,arr_alpha,dev_u=dev_u,photon_N = 250000,
            start_r = 2.10428659,f_ratio  = 2,
            data_dir='/gpfs/home/ess/pjzhang/sunray/datatmp/harmo/')

# fundamental 1.75 r=1.1
# f=35.045076 MHz