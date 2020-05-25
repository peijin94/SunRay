import torch

const_c = 2.998e10
# use CUDA if there is Nvidia GPU

dev_u = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#dev_u = torch.device('cpu') # force CPU
#dev_u = torch.device('cpu')