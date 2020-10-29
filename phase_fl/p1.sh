# python phase_fl_train.py --net_type Unet --backbone efficientnetb0 --dataset bead_dataset --epoch 200 --dim 512 --loss mae --batch_size 6 --lr 5e-4 --pre_train True --gpu 1
# python phase_fl_train.py --net_type Unet --backbone efficientnetb1 --dataset neuron_x2 --epoch 200 --dim 512 --loss mse --batch_size 6 --rot 20 --lr 5e-4 --pre_train True --gpu 1
python phase_fl_train3.py --net_type Unet --backbone efficientnetb1 --dataset neuron_x2 --epoch 100 --dim 512 --loss mse --batch_size 6 --rot 20 --lr 5e-4 --pre_train True --gpu 1 --ch_in 3 --ch_out 1
python phase_fl_train3.py --net_type Unet --backbone efficientnetb1 --dataset neuron_x2 --epoch 100 --dim 512 --loss mse --batch_size 6 --rot 20 --lr 5e-4 --pre_train True --gpu 1 --ch_in 1 --ch_out 1