python phase_fl1_fl2_train.py --net_type Unet --backbone efficientnetb7 --dataset neuron_wbx1 --subset train --epoch 150 --dim 512 --loss mse --batch_size 4 --rot 20 --lr 1e-4 --pre_train True --gpu 0 --fl_ch fl1 --ch_in 1 --ch_out 1

