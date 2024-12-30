#!/bin/bash
#DSUB -n job_genomix
#DSUB -A root.project.P24Z31300N0038_tmp
#DSUB -eo tmp/%J.%I.err.log
#DSUB -oo tmp/%J.%I.out.log

# 
# (C)Copyright 2023-2024: Yong Bai, baiyong at genomics.cn
#
source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
# module load libs/openblas/0.3.18_kgcc9.3.1
module load compilers/gcc/12.2.0
module load compilers/cuda/11.8.0
module load libs/cudnn/8.2.1_cuda11.3
module load libs/nccl/2.17.1-1_cuda11.0
source activate /home/share/huadjyin/home/baiyong01/.conda/envs/py10

export TMPDIR='/tmp'
export TRITON_CACHE_DIR='/tmp/.triton/cache'

###Set Start Path
JOB_PATH="/home/share/huadjyin/home/baiyong01/projects/genomix/training"

## Create nodefile
JOB_ID=${BATCH_JOB_ID}
NODEFILE=${JOB_PATH}/tmp/${JOB_ID}.nodefile
# touch ${NODEFILE}
touch $NODEFILE
#cat $CCS_ALLOC_FILE | grep ^cyclone001-agent | awk '{print $1,"slots="$2}' > ${JOB_PATH}/tmp/${JOB_ID}.nodefile
# cat $CCS_ALLOC_FILE | grep ^cyclone001-agent | awk '{print $1}' > ${NODEFILE}
cat ${CCS_ALLOC_FILE} > ${JOB_PATH}/tmp/CCS_ALLOC_FILE

HOST=`hostname`
flock -x ${NODEFILE} -c "echo ${HOST} >> ${NODEFILE}"
MASTER_IP=`head -n 1 ${NODEFILE}`
echo $MASTER_IP

torchrun --nproc_per_node 1 01_debug_test.py 

# dsub -R 'cpu=20;gpu=1;mem=100000' -s run_one_node.sh

# nohup torchrun --nproc_per_node 1 run_bioseqmamba_train.py > train.log &
# if use one GPU, set os.environ["CUDA_VISIBLE_DEVICES"] = "0" in front of .py script.