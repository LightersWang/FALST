#!/bin/bash

# # Ours
# for SLIDE_AL in random; do
#     for PATCH_AL in falst; do
#         DATE=$(date +"%Y%m%d_%H%M%S")
#         ROOT_DIR=/media/temp/DATA21/Project_FAST/CODE/FALST/log/CAMELYON_TextBranchLearnable_all_pos_final/Slide_${SLIDE_AL}_Patch_${PATCH_AL}_gmm50_${DATE}
#         echo ${ROOT_DIR}
#         mkdir -p ${ROOT_DIR}/4_InstanceShot
#         mkdir -p ${ROOT_DIR}/16_InstanceShot
#         bash /media/temp/DATA21/Project_FAST/CODE/FALST/exp_CAMELYON_TextBranchLearnable_active_fast_all_pos_gmm50.sh ${SLIDE_AL} ${PATCH_AL} ${ROOT_DIR}
#     done
# done

# # GMM20
# for SLIDE_AL in random; do
#     for PATCH_AL in falst; do
#         DATE=$(date +"%Y%m%d_%H%M%S")
#         ROOT_DIR=/media/temp/DATA21/Project_FAST/CODE/FALST/log/CAMELYON_TextBranchLearnable_all_pos_final/Slide_${SLIDE_AL}_Patch_${PATCH_AL}_gmm20_${DATE}
#         echo ${ROOT_DIR}
#         mkdir -p ${ROOT_DIR}/4_InstanceShot
#         mkdir -p ${ROOT_DIR}/16_InstanceShot
#         bash /media/temp/DATA21/Project_FAST/CODE/FALST/exp_CAMELYON_TextBranchLearnable_active_fast_all_pos_gmm20.sh ${SLIDE_AL} ${PATCH_AL} ${ROOT_DIR}
#     done
# done

# # GMM100
# for SLIDE_AL in random; do
#     for PATCH_AL in falst; do
#         DATE=$(date +"%Y%m%d_%H%M%S")
#         ROOT_DIR=/media/temp/DATA21/Project_FAST/CODE/FALST/log/CAMELYON_TextBranchLearnable_all_pos_final/Slide_${SLIDE_AL}_Patch_${PATCH_AL}_gmm100_${DATE}
#         echo ${ROOT_DIR}
#         mkdir -p ${ROOT_DIR}/4_InstanceShot
#         mkdir -p ${ROOT_DIR}/16_InstanceShot
#         bash /media/temp/DATA21/Project_FAST/CODE/FALST/exp_CAMELYON_TextBranchLearnable_active_fast_all_pos_gmm100.sh ${SLIDE_AL} ${PATCH_AL} ${ROOT_DIR}
#     done
# done

# KDE
for SLIDE_AL in random; do
    for PATCH_AL in falst; do
        DATE=$(date +"%Y%m%d_%H%M%S")
        ROOT_DIR=/media/temp/DATA21/Project_FAST/CODE/FALST/log/CAMELYON_TextBranchLearnable_all_pos_final/Slide_${SLIDE_AL}_Patch_${PATCH_AL}_kde_${DATE}
        echo ${ROOT_DIR}
        mkdir -p ${ROOT_DIR}/4_InstanceShot
        mkdir -p ${ROOT_DIR}/16_InstanceShot
        bash /media/temp/DATA21/Project_FAST/CODE/FALST/exp_CAMELYON_TextBranchLearnable_active_fast_all_pos_kde.sh ${SLIDE_AL} ${PATCH_AL} ${ROOT_DIR}
    done
done

# Neg Only
for SLIDE_AL in random; do
    for PATCH_AL in falst; do
        DATE=$(date +"%Y%m%d_%H%M%S")
        ROOT_DIR=/media/temp/DATA21/Project_FAST/CODE/FALST/log/CAMELYON_TextBranchLearnable_all_pos_final/Slide_${SLIDE_AL}_Patch_${PATCH_AL}_only_neg_${DATE}
        echo ${ROOT_DIR}
        mkdir -p ${ROOT_DIR}/4_InstanceShot
        mkdir -p ${ROOT_DIR}/16_InstanceShot
        bash /media/temp/DATA21/Project_FAST/CODE/FALST/exp_CAMELYON_TextBranchLearnable_active_fast_all_pos_only_neg.sh ${SLIDE_AL} ${PATCH_AL} ${ROOT_DIR}
    done
done

# Pos Only
for SLIDE_AL in random; do
    for PATCH_AL in falst; do
        DATE=$(date +"%Y%m%d_%H%M%S")
        ROOT_DIR=/media/temp/DATA21/Project_FAST/CODE/FALST/log/CAMELYON_TextBranchLearnable_all_pos_final/Slide_${SLIDE_AL}_Patch_${PATCH_AL}_only_pos_${DATE}
        echo ${ROOT_DIR}
        mkdir -p ${ROOT_DIR}/4_InstanceShot
        mkdir -p ${ROOT_DIR}/16_InstanceShot
        bash /media/temp/DATA21/Project_FAST/CODE/FALST/exp_CAMELYON_TextBranchLearnable_active_fast_all_pos_only_pos.sh ${SLIDE_AL} ${PATCH_AL} ${ROOT_DIR}
    done
done

