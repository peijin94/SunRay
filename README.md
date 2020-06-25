# Python version of scattering simulation

The algorithm of anisotropic scattering is reproduced from the code mentioned in [Kontar et. al. ApJ 2019]

The major computational part is done with the help of PyTorch, which is known as an excellent machine learning framework, but also a powerful tool for numerical computation, the computaion can be done with either CPU or GPU. Also the best feature is the **autodiff**, which carried out most of the differential works in the computaion.

## Dependency

Note: this is done with:  
    PyTorch==1.5.0  
    python>=3.8.3

## Variables

### Variable list

| Variable Name | Info |
|---------------|------|
| ```dev_u```  | Device decalimer, 'cuda' for GPU calculation,'cpu' for cpu|
| ```step_N``` | The number of step, complete simulation <br> set as '-1' to let the program decide how many steps to run |
| ```collect_N```| The steps to be collected (for debug and dev)|
| ```photon_N```| The number of simulated photons, <br> (recommend: GPU: 500000, CPU: 10000) |

when ```photon_N``` is smaller than 10000, CPU is faster than GPU due to the memory exchange

### Important variable size

| Variable Name | ```var.shape```  |
|---------------|-----------------|
| ```r_vec```   | ```[3,photon_N]``` |
| ```k_vec```   | ```[3,photon_N]``` |
| ```r_vec_collect```   | ```[t_N,3,photon_N]``` |
| ```rr_cur```  | ```[photon_N]``` |
| ```tau```     | ```[photon_N]``` |




## Benchmark

 ```bash
 pprofile --exclude-syspath sunRay_v0 > a_gpu.txt
 ```

GPU tends to be faster when the number of photon ```photon_N``` >100k.
