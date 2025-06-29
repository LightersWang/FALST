#!/bin/bash
SLIDE_AL=$1
PATCH_AL=$2
ROOT_DIR=$3

echo 'All Positive'

# Instance Shot = 16
python CAMELYON_TextBranchLearnable_active_all_pos.py --slide_active_method ${SLIDE_AL} --patch_active_method ${PATCH_AL}  --seed 42 --num_bag_shot 16 --num_instance_shot 16 --downsample_neg_instances -2000  --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ${ROOT_DIR}/16_InstanceShot/Seed42_16_16.txt
wait

python CAMELYON_TextBranchLearnable_active_all_pos.py --slide_active_method ${SLIDE_AL} --patch_active_method ${PATCH_AL}  --seed 43 --num_bag_shot 16 --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ${ROOT_DIR}/16_InstanceShot/Seed43_16_16.txt
wait

python CAMELYON_TextBranchLearnable_active_all_pos.py --slide_active_method ${SLIDE_AL} --patch_active_method ${PATCH_AL}  --seed 44 --num_bag_shot 4  --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ${ROOT_DIR}/16_InstanceShot/Seed44_4_16.txt
python CAMELYON_TextBranchLearnable_active_all_pos.py --slide_active_method ${SLIDE_AL} --patch_active_method ${PATCH_AL}  --seed 44 --num_bag_shot 8  --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ${ROOT_DIR}/16_InstanceShot/Seed44_8_16.txt
python CAMELYON_TextBranchLearnable_active_all_pos.py --slide_active_method ${SLIDE_AL} --patch_active_method ${PATCH_AL}  --seed 44 --num_bag_shot 16 --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ${ROOT_DIR}/16_InstanceShot/Seed44_16_16.txt
wait
