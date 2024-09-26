#!/bin/sh
# script for MPI job submission
#SBATCH -J rayCalc_GPU
#SBATCH -o job-%j.log
#SBATCH -e job-%j.err
#SBATCH -N 4 --ntasks-per-node=2 -p GPU-V100 --gres=gpu:v100:2
echo Time is `date`
echo Directory is $PWD
echo This job runs on the following nodes:
echo $SLURM_JOB_NODELIST
echo This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu cores.

source /home/ess/pjzhang/conda_start.sh
conda activate torch15 # activate the enviroment with torch 1.5

#module load mpich/3.2/intel/2016.3.210
MPIRUN=mpiexec #MPICH
MPIOPT="-iface ib0" #MPICH3 # use infiniband for communication
$MPIRUN $MPIOPT -n 8 python /home/ess/pjzhang/sunray/sunRay_SCC/sunRay_MPI_GPU.py

echo End at `date`  # for the measure of the running time of the 
