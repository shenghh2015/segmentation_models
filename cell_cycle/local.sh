# compton
python train_model.py --net_type AtUnet --backbone efficientnetb2 --pre_train True --batch_size 4 --dim 512 --epoch 2400 --lr 5e-4 --dataset cycle_736x752 --gpu 6 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4 --docker False
python train_model.py --net_type AtUnet --backbone efficientnetb3 --pre_train True --batch_size 4 --dim 512 --epoch 2400 --lr 5e-4 --dataset cycle_736x752 --gpu 1 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4 --docker False
python train_model.py --net_type AtUnet --backbone efficientnetb4 --pre_train True --batch_size 2 --dim 512 --epoch 2400 --lr 5e-4 --dataset cycle_736x752 --gpu 2 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4 --docker False
python train_model.py --net_type AtUnet --backbone efficientnetb5 --pre_train True --batch_size 2 --dim 512 --epoch 2400 --lr 5e-4 --dataset cycle_736x752 --gpu 3 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4 --docker False
python train_model.py --net_type AtUnet --backbone efficientnetb6 --pre_train True --batch_size 1 --dim 512 --epoch 2400 --lr 5e-4 --dataset cycle_736x752 --gpu 4 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4 --docker False
python train_model.py --net_type AtUnet --backbone efficientnetb7 --pre_train True --batch_size 1 --dim 512 --epoch 2400 --lr 5e-4 --dataset cycle_736x752 --gpu 5 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4 --docker False

# predator
# python train_binary_model.py --net_type AtUnet --backbone efficientnetb0 --pre_train True --batch_size 6 --dim 512 --epoch 2400 --lr 5e-4 --dataset cycle_736x752 --gpu 0 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4 --docker True
python train_binary_model.py --net_type Unet --backbone efficientnetb0 --pre_train True --batch_size 6 --dim 512 --epoch 2400 --lr 5e-4 --dataset cycle_736x752 --gpu 1 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4 --docker True
# Jan 16, 2021
python train_viability.py --net_type AtUnet --backbone efficientnetb0 --pre_train True --batch_size 6 --dim 512 --epoch 2400 --lr 5e-4 --dataset viability_832x832 --gpu 0 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4 --docker True
python train_viability.py --net_type AtUnet --backbone efficientnetb1 --pre_train True --batch_size 6 --dim 512 --epoch 2400 --lr 5e-4 --dataset viability_832x832 --gpu 1 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4 --docker True

# Jan 16, 2021
python train_viability.py --net_type AtUnet --backbone efficientnetb2 --pre_train True --batch_size 4 --dim 512 --epoch 2400 --lr 5e-4 --dataset viability_832x832 --gpu 3 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4 --docker False
python train_viability.py --net_type AtUnet --backbone efficientnetb3 --pre_train True --batch_size 4 --dim 512 --epoch 2400 --lr 5e-4 --dataset viability_832x832 --gpu 4 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4 --docker False
python train_viability.py --net_type AtUnet --backbone efficientnetb4 --pre_train True --batch_size 2 --dim 512 --epoch 2400 --lr 5e-4 --dataset viability_832x832 --gpu 2 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4 --docker False
# python train_viability.py --net_type AtUnet --backbone efficientnetb5 --pre_train True --batch_size 2 --dim 512 --epoch 2400 --lr 5e-4 --dataset viability_832x832 --gpu 3 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4 --docker False

# Jan 19, 2021
# predator
python train_viability.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 4 --dim 512 --epoch 2400 --lr 5e-4 --dataset viability2_832x832 --gpu 0 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4 --docker True
python train_viability.py --net_type AtUnet --backbone efficientnetb3 --pre_train True --batch_size 4 --dim 512 --epoch 2400 --lr 5e-4 --dataset viability2_832x832 --gpu 1 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4 --docker True