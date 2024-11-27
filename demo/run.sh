#!/bin/bash
# dsub -A root.project.P24Z10200N0985 -R 'cpu=50;gpu=0;mem=160000' -eo %J.%I.err.log -oo %J.%I.out.log -s run.sh

source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
# module load libs/openblas/0.3.18_kgcc9.3.1
module load compilers/gcc/12.2.0
module load compilers/cuda/11.8.0
module load libs/cudnn/8.2.1_cuda11.3
module load libs/nccl/2.17.1-1_cuda11.0
source activate /home/share/huadjyin/home/baiyong01/.conda/envs/py10

JOB_ID=${BATCH_JOB_ID}
NODEFILE=${JOB_ID}.nodefile
# touch ${NODEFILE}
touch $NODEFILE
HOST=`hostname`
flock -x ${NODEFILE} -c "echo ${HOST} >> ${NODEFILE}"
MASTER_IP=`head -n 1 ${NODEFILE}`
echo $MASTER_IP
rm $NODEFILE

python 00_func_test.py