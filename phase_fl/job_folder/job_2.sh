python phase_fl1_fl2_train2.py --net_type Unet --backbone efficientnetb0 --dataset neuron_trn_tst --subset train --epoch 100 --dim 512 --loss mse --batch_size 3 --rot 50 --lr 1e-4 --pre_train True --gpu 2 --fl_ch fl1 --ch_in 3 --ch_out 3

