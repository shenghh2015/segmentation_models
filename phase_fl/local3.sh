## compton
# python parallel_phase_fl1_fl2_train.py --net_type Unet --backbone efficientnetb7 --dataset neuron_trn_tst --subset train --epoch 100 --dim 320 --loss mse --batch_size 1 --rot 50 --lr 1e-4 --pre_train True --gpu 1,2,3,4 --fl_ch fl12 --ch_in 3 --ch_out 3 --docker False
python phase_fl1_fl2_train2.py --net_type Unet --backbone efficientnetb3 --dataset neuron_trn_tst --subset train --epoch 100 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 1e-4 --pre_train True --gpu 1 --fl_ch fl1 --ch_in 3 --ch_out 3