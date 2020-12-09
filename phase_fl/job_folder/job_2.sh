python train_exp.py --net_type Unet --backbone efficientnetb7 --dataset neuron_trn_tst_v3 --subset train --epoch 800 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 5e-4 --pre_train True --gpu 2 --fl_ch fl1 --ch_in 3 --ch_out 1 --best_select True --scale 1. --docker True --extra False

