#!/bin/bash

for SLIDE_AL in random; do
    for PATCH_AL in random; do
        DATE=$(date +"%Y%m%d_%H%M%S")
        ROOT_DIR=/media/temp/DATA21/Project_FAST/CODE/FALST/log/CAMELYON_TextBranchLearnable_202503/Slide_${SLIDE_AL}_Patch_${PATCH_AL}_${DATE}
        echo ${ROOT_DIR}
        mkdir -p ${ROOT_DIR}/4_InstanceShot
        mkdir -p ${ROOT_DIR}/16_InstanceShot
        mkdir -p ${ROOT_DIR}/64_InstanceShot
        bash /media/temp/DATA21/Project_FAST/CODE/FALST/exp_CAMELYON_TextBranchLearnable_active_fast.sh ${SLIDE_AL} ${PATCH_AL} ${ROOT_DIR}
    done
done