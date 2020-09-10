python single_train.py --net_type ResUnet --backbone efficientnetb3 --pre_train True --batch_size 14 --dim 512 --epoch 150 --lr 5e-4 --dataset live_dead --train 900 --gpu 3 --loss focal+dice --filters 256 --upsample upsampling

