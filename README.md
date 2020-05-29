# Python version of scattering simulation

Note: this is done with:  
    PyTorch==1.5.0  
    python>=3.8.3

The algorithm of anisotropic scattering is reproduced from the code mentioned in [Kontar et. al. ApJ 2019]

The major computational part is done with the help of PyTorch, which is known as an excellent machine learning framework, but also a powerful tool for numerical computation, the computaion can be done with either CPU or GPU. Also the best feature is the **autodiff**, with carried out all the differential works in the computaion.

## Variables

### Variable list

| Variable Name | Info |
|---------------|------|
| ```dev_u```  | Device decalimer, 'cuda' for GPU calculation,'cpu' for cpu|
| ```step_N``` | The number of step, complete simulation case with N>5000 |
| ```collect_N```| The steps to be collected |
| ```photon_N```| The number of simulated photons, GPU: 500k, CPU: 30k |

### Important variable size

| Variable Name | ```var.shape```  |
|---------------|-----------------|
| ```r_vec```   | ```[3,photon_N]``` |
| ```k_vec```   | ```[3,photon_N]``` |
| ```rr_cur```  | ```[photon_N]``` |
| ```tau```     | ```[photon_N]``` |



### Benchmark

 ```bash
 pprofile --exclude-syspath sunRay_v0 > a_gpu.txt
 ```

GPU tends to be faster when the number of photon ```photon_N``` >100k.
