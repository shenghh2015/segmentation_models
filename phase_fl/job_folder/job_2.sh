python phase_fl1_fl2_train.py --net_type Unet --backbone efficientnetb7 --dataset neuron_wbx1 --subset train --epoch 150 --dim 512 --loss mse --batch_size 4 --rot 20 --lr 1e-5 --pre_train True --gpu 2 --fl_ch fl1

