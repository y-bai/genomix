#!/bin/bash
#DSUB -n job_genomix
#DSUB -A root.project.P24Z31300N0038_tmp
#DSUB -R 'cpu=48;gpu=4;mem=100000'
#DSUB -N 3
#DSUB -eo tmp/%J.%I.err.log
#DSUB -oo tmp/%J.%I.out.log

## Set scripts
RANK_SCRIPT="run_multi_node_script.sh"

###Set Start Path
JOB_PATH="/home/share/huadjyin/home/baiyong01/projects/genomix/training"

## Set NNODES
NNODES=3

## Create nodefile
JOB_ID=${BATCH_JOB_ID}
NODEFILE=${JOB_PATH}/tmp/${JOB_ID}.nodefile
# touch ${NODEFILE}
touch $NODEFILE
#cat $CCS_ALLOC_FILE | grep ^cyclone001-agent | awk '{print $1,"slots="$2}' > ${JOB_PATH}/tmp/${JOB_ID}.nodefile
# cat $CCS_ALLOC_FILE | grep ^cyclone001-agent | awk '{print $1}' > ${NODEFILE}
cat ${CCS_ALLOC_FILE} > tmp/CCS_ALLOC_FILE

mkdir -p /tmp/.triton/cache; chmod 777 /tmp/.triton/cache

cd ${JOB_PATH};/usr/bin/bash ${RANK_SCRIPT} ${NNODES} ${NODEFILE}


# dsub -s run_multi_node_launch.sh