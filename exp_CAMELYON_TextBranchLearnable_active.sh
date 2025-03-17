#!/bin/bash

# for SLIDE_AL in random selected; do
#     for PATCH_AL in random; do
#         DATE=$(date +"%Y%m%d_%H%M%S")
#         ROOT_DIR=/media/temp/DATA21/Project_FAST/CODE/FALST/log/CAMELYON_TextBranchLearnable_202503_all_pos/Slide_${SLIDE_AL}_Patch_${PATCH_AL}_${DATE}
#         echo ${ROOT_DIR}
#         mkdir -p ${ROOT_DIR}/4_InstanceShot
#         mkdir -p ${ROOT_DIR}/16_InstanceShot
#         mkdir -p ${ROOT_DIR}/64_InstanceShot
#         bash /media/temp/DATA21/Project_FAST/CODE/FALST/exp_CAMELYON_TextBranchLearnable_active_fast_all_pos.sh ${SLIDE_AL} ${PATCH_AL} ${ROOT_DIR}
#     done
# done

# for SLIDE_AL in random selected; do
#     for PATCH_AL in falst; do
#         DATE=$(date +"%Y%m%d_%H%M%S")
#         ROOT_DIR=/media/temp/DATA21/Project_FAST/CODE/FALST/log/CAMELYON_TextBranchLearnable_202503_all_pos/Slide_${SLIDE_AL}_Patch_${PATCH_AL}_${DATE}
#         echo ${ROOT_DIR}
#         mkdir -p ${ROOT_DIR}/4_InstanceShot
#         mkdir -p ${ROOT_DIR}/16_InstanceShot
#         mkdir -p ${ROOT_DIR}/64_InstanceShot
#         bash /media/temp/DATA21/Project_FAST/CODE/FALST/exp_CAMELYON_TextBranchLearnable_active_fast_all_pos.sh ${SLIDE_AL} ${PATCH_AL} ${ROOT_DIR}
#     done
# done

for SLIDE_AL in random; do
    for PATCH_AL in oracle; do
        DATE=$(date +"%Y%m%d_%H%M%S")
        ROOT_DIR=/media/temp/DATA21/Project_FAST/CODE/FALST/log/CAMELYON_TextBranchLearnable_202503_all_pos/Slide_${SLIDE_AL}_Patch_${PATCH_AL}_${DATE}
        echo ${ROOT_DIR}
        mkdir -p ${ROOT_DIR}/4_InstanceShot
        mkdir -p ${ROOT_DIR}/16_InstanceShot
        mkdir -p ${ROOT_DIR}/64_InstanceShot
        bash /media/temp/DATA21/Project_FAST/CODE/FALST/exp_CAMELYON_TextBranchLearnable_active_fast_all_pos.sh ${SLIDE_AL} ${PATCH_AL} ${ROOT_DIR}
    done
done

### ablation

# for SLIDE_AL in random; do
#     for PATCH_AL in falst; do
#         DATE=$(date +"%Y%m%d_%H%M%S")
#         ROOT_DIR=/media/temp/DATA21/Project_FAST/CODE/FALST/log/CAMELYON_TextBranchLearnable_202503/Slide_${SLIDE_AL}_Patch_${PATCH_AL}_ablation2_${DATE}
#         echo ${ROOT_DIR}
#         mkdir -p ${ROOT_DIR}/4_InstanceShot
#         mkdir -p ${ROOT_DIR}/16_InstanceShot
#         bash /media/temp/DATA21/Project_FAST/CODE/FALST/exp_CAMELYON_TextBranchLearnable_active_fast_2.sh ${SLIDE_AL} ${PATCH_AL} ${ROOT_DIR}
#     done
# done

# for SLIDE_AL in random; do
#     for PATCH_AL in falst; do
#         DATE=$(date +"%Y%m%d_%H%M%S")
#         ROOT_DIR=/media/temp/DATA21/Project_FAST/CODE/FALST/log/CAMELYON_TextBranchLearnable_202503/Slide_${SLIDE_AL}_Patch_${PATCH_AL}_ablation3_${DATE}
#         echo ${ROOT_DIR}
#         mkdir -p ${ROOT_DIR}/4_InstanceShot
#         mkdir -p ${ROOT_DIR}/16_InstanceShot
#         bash /media/temp/DATA21/Project_FAST/CODE/FALST/exp_CAMELYON_TextBranchLearnable_active_fast_3.sh ${SLIDE_AL} ${PATCH_AL} ${ROOT_DIR}
#     done
# done



# for SLIDE_AL in random; do
#     for PATCH_AL in falst; do
#         DATE=$(date +"%Y%m%d_%H%M%S")
#         ROOT_DIR=/media/temp/DATA21/Project_FAST/CODE/FALST/log/CAMELYON_TextBranchLearnable_202503/Slide_${SLIDE_AL}_Patch_${PATCH_AL}_ablation4_${DATE}
#         echo ${ROOT_DIR}
#         mkdir -p ${ROOT_DIR}/4_InstanceShot
#         mkdir -p ${ROOT_DIR}/16_InstanceShot
#         bash /media/temp/DATA21/Project_FAST/CODE/FALST/exp_CAMELYON_TextBranchLearnable_active_fast_4.sh ${SLIDE_AL} ${PATCH_AL} ${ROOT_DIR}
#     done
# done