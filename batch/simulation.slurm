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
#source ~/source_desi_elg2017b.sh

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

logdirectory='/scratch1/scratchdirs/govinda/projects/elg2017b/logs/'$mode
out=$logdirectory"/slurm_"${mode}${nspec}".out"
brickname=${mode}${nspec}
logfile=$logdirectory"/sim_"${brickname}".log"
paramfile=$logdirectory"/param_"${mode}".txt"
expfile='./exposure.fits'

echo "This param file is", ${paramfile}
if [[ $mode == "nominal" ]]
then
    echo "Simulating mode:", $mode 
    srun -o ${out} run_simulation --expfile ${expfile} --brickname ${brickname} --nspec ${nspec} --seed ${seed} --breakbricks >> ${logfile} 2>&1
else
    echo "Simulation mode:", $mode
    srun -o ${out} run_simulation --expfile ${expfile} --brickname ${brickname} --${mode} --nspec ${nspec} --seed ${seed} --breakbricks >> ${logfile} 2>&1
fi

echo "Finished simulation for", ${brickname}
date

