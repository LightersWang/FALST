#!/bin/bash

# for SLIDE_AL in random; do
#     for PATCH_AL in falst; do
#         DATE=$(date +"%Y%m%d_%H%M%S")
#         ROOT_DIR=/media/temp/DATA21/Project_FAST/CODE/FALST/log/CAMELYON_TextBranchLearnable_20250318_all_pos/Slide_${SLIDE_AL}_Patch_${PATCH_AL}_gmm100_${DATE}
#         echo ${ROOT_DIR}
#         mkdir -p ${ROOT_DIR}/4_InstanceShot
#         mkdir -p ${ROOT_DIR}/16_InstanceShot
#         bash /media/temp/DATA21/Project_FAST/CODE/FALST/exp_CAMELYON_TextBranchLearnable_active_fast_all_pos_gmm100.sh ${SLIDE_AL} ${PATCH_AL} ${ROOT_DIR}
#     done
# done

for SLIDE_AL in random; do
    for PATCH_AL in falst; do
        DATE=$(date +"%Y%m%d_%H%M%S")
        ROOT_DIR=/media/temp/DATA21/Project_FAST/CODE/FALST/log/CAMELYON_TextBranchLearnable_20250318_all_pos/Slide_${SLIDE_AL}_Patch_${PATCH_AL}_pos_only_${DATE}
        echo ${ROOT_DIR}
        mkdir -p ${ROOT_DIR}/4_InstanceShot
        mkdir -p ${ROOT_DIR}/16_InstanceShot
        bash /media/temp/DATA21/Project_FAST/CODE/FALST/exp_CAMELYON_TextBranchLearnable_active_fast_all_pos_only_pos.sh ${SLIDE_AL} ${PATCH_AL} ${ROOT_DIR}
    done
done

for SLIDE_AL in random; do
    for PATCH_AL in falst; do
        DATE=$(date +"%Y%m%d_%H%M%S")
        ROOT_DIR=/media/temp/DATA21/Project_FAST/CODE/FALST/log/CAMELYON_TextBranchLearnable_20250318_all_pos/Slide_${SLIDE_AL}_Patch_${PATCH_AL}_neg_only_${DATE}
        echo ${ROOT_DIR}
        mkdir -p ${ROOT_DIR}/4_InstanceShot
        mkdir -p ${ROOT_DIR}/16_InstanceShot
        bash /media/temp/DATA21/Project_FAST/CODE/FALST/exp_CAMELYON_TextBranchLearnable_active_fast_all_pos_only_neg.sh ${SLIDE_AL} ${PATCH_AL} ${ROOT_DIR}
    done
done
