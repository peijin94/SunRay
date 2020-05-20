# Python version of scattering simulation

Note: this is done with:  
    PyTorch==1.5.0  
    python 3.8.3

## check with office computer the version

### Benchmark

 ```bash
 pprofile --exclude-syspath sunRay_v0 > a_gpu.txt
 ```

GPU tends to be faster when the number of particles >100k.
