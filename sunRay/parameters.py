import torch
const_c = 2.998e10
# use CUDA if there is Nvidia GPU

dev_u = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#dev_u = torch.device('cpu') # force CPU

R_S = 6.96e10         # the radius of the sun 
c   = 2.998e10        # speed of light
c_r = c/R_S           # [t]