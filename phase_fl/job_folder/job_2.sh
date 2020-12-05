python train_neuron.py --net_type Unet --backbone efficientnetb6 --dataset spheroids_v5 --subset train --epoch 800 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 1e-4 --pre_train True --gpu 4 --fl_ch fl1 --ch_in 3 --ch_out 3 --best_select True --scale 100.

