## June 18
# python live_dead.py --net_type FPN --backbone efficientnetb3 --pre_train True --batch_size 8 --dim 320 --epoch 600 --lr 5e-4 --gpu 0

# python live_dead.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 5 --dim 512 --epoch 200 --lr 5e-4 --train 900 --bk_weight 0.7 --gpu 0
python live_dead.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 5 --dim 512 --epoch 300 --lr 1e-4 --one_dataset True --train 900 --gpu 0