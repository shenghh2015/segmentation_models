python single_train.py --net_type ResUnet --backbone efficientnetb0 --pre_train True --batch_size 14 --dim 512 --epoch 150 --lr 5e-4 --dataset live_dead --train 900 --gpu 0 --loss focal+dice --filters 256 --upsample upsampling

