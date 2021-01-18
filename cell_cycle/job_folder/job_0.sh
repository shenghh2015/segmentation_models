python train_viability.py --net_type AtUnet --backbone efficientnetb0 --pre_train True --batch_size 4 --dim 800 --epoch 2400 --lr 5e-4 --dataset viability_832x832 --gpu 0 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4 

