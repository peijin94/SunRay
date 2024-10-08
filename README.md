# Python version of scattering simulation

The algorithm of anisotropic scattering is reproduced from the code mentioned in [Kontar et. al. ApJ 2019]

The major computational part is done with the help of PyTorch, which is known as an excellent machine learning framework, but also a powerful tool for numerical computation, the computaion can be done with either CPU or GPU. Also the best feature is the **autodiff**, which carried out most of the differential works in the computaion.

## installation

```
git clone https://github.com/peijin94/SunRay.git
cd SunRay
python -m pip install .
```

## Dependency

Note: this is done with:  
    PyTorch==1.5.0  
    python>=3.8.3
    mpi4py (for parallelize)

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

For multiple GPU, try bash ```parallel``` or  python ```multiprocessing``` to parallel:

```bash
cat params.input | parallel -j4 'CUDA_VISIBLE_DEVICES=$(({%} - 1)) python {} &> {#}.out'
```

### Simulation output

* Level 0 (lv0) x,y,z kx,ky,kz, t at sampled time points. (huge in size but contains every thing, all information of all checkpoints, just for test cases, too large)

|Variable Name|
|---|
|steps_N             | 
|collect_N           |
|photon_N            |
|start_r             |
|start_theta         |
|start_phi           |
|f_ratio             |
|epsilon             |
|anis                |
|asym                |
|omega0              |
|freq0               |
|t_collect           |
|tau                 |
|r_vec_collect_local |
|k_vec_collect_local |
|tau_collect_local   |


* Level 1 (lv1) x,y,z kx,ky,kz at arrival shell (single time point, **recommended form**)

|Variable Name|
|---|
|steps_N  |
 |collect_N |
 |photon_N |
 |start_r |
 |start_theta |
 |start_phi  |
 |f_ratio  |
 |epsilon |
 |anis |
 |asym |
 |omega0|
 |freq0|
 |r_vec_stat_avail|
 |k_vec_stat_avail|
 |t_reach_stat_avail|
|tau_stat_avail|
|r_vec_0|
 |k_vec_0|
 |dk_refr_avail|
 |dk_scat_avail|
 |dkx_refr_avail|
|dky_refr_avail|
|dkz_refr_avail|
|dkx_scat_avail|
|dky_scat_avail|
|dkz_scat_avail|


* Level 2 (lv2) im_x, im_y, tretraced and observed x,y position estimated (TBD)
* Level 3 (lv3) duration and source size (TBD)

## Benchmark

 ```bash
 pprofile --exclude-syspath sunRay_v0 > a_gpu.txt
 ```

GPU tends to be faster when the number of photon ```photon_N``` >100k.
