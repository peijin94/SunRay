# Python version of scattering simulation

Note: this is done with:  
    PyTorch==1.5.0  
    python 3.8.3

## check with office computer the version

### Variables

A list of varibles

| Variable Name | Info |
|---------------|------|
| ```dev_u```  | Device decalimer, 'cuda' for GPU calculation,'cpu' for cpu|
| ```step_N``` | The number of step, complete simulation case with N>5000 |
| ```collect_N```| The steps to be collected |
| ```photon_N```| The number of simulated photons, GPU: 500k, CPU: 30k |

### Benchmark

 ```bash
 pprofile --exclude-syspath sunRay_v0 > a_gpu.txt
 ```

GPU tends to be faster when the number of photon ```photon_N``` >100k.
