# CUDA_VISIBLE_DEVICES=0 \
# python /home/se3_tracknet/predict.py \
#   --train_data_path /media/bowen/e25c9489-2f57-42dd-b076-021c59369fec/DATASET/train_example/bleach_cleanser/train_data_blender_DR \
#   --ckpt_dir /home/bowen/debug/bleach_cleanser/model_best_val.pth.tar \
#   --mean_std_path /home/bowen/debug/bleach_cleanser \
#   --class_id 12 \

CUDA_VISIBLE_DEVICES=0 \
/home/marcusmartin/miniconda3/envs/iros20/bin/python /home/marcusmartin/repos/iros20-6d-pose-tracking/predict.py \
  --train_data_path /media/marcusmartin/e25c9489-2f57-42dd-b076-021c59369fec/DATASET/train_example/bleach_cleanser/train_data_blender_DR \
  --ckpt_dir /home/marcusmartin/debug/bleach_cleanser/model_best_val.pth.tar \
  --mean_std_path /home/marcusmartin/debug/bleach_cleanser \
  --class_id 12 \
