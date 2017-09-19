#!/bin/bash

#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --time=120
#SBATCH --job-name=simulation

#node_cores=24
#nodes=1
#node_proc=1
#node_thread=$(( node_cores / node_proc ))
#procs=$(( nodes * node_proc))

now=date #'date +%Y%m%d-%H:%M:%S'
echo "job datestamp = ${now}"
echo "Starting at" 
date
#source ~/source_desi_updated.sh

nspec="$1"
mode="$2"
echo "Nspec", ${nspec}
echo "Mode", ${mode}
seed=12345

echo "No of arguments", $#
if [[ $# -eq 1 ]]
then
    mode="nominal"
else
    mode=$mode
fi

echo "This mode is", ${mode}
out="slurm_"${mode}${nspec}".out"
brickname=${mode}${nspec}
logfile="sim_"${brickname}".log"
paramfile="param_"${mode}".txt"
expfile='../data/exposure.fits'

echo "This param file is", ${paramfile}
if [[ $mode == "nominal" ]]
then
    echo "Simulating mode:", $mode 
    srun -o ${out} run_simulation --expfile ${expfile} --brickname ${brickname} --nspec ${nspec} --seed ${seed} --paramfile ${paramfile} --breakbricks >> ${logfile} 2>&1
else
    echo "Simulation mode:", $mode
    srun -o ${out} run_simulation --expfile ${expfile} --brickname ${brickname} --${mode} --galsim --nspec ${nspec} --seed ${seed} --paramfile ${paramfile} --breakbricks >> ${logfile} 2>&1
fi

echo "Finished simulation for", ${brickname}
date

