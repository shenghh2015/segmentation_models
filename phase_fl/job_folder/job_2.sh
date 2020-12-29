python train_neuron.py --net_type AtUnet --backbone efficientnetb6 --dataset spheroids_v5 --subset train --epoch 1200 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 1e-4 --pre_train True --gpu 2 --fl_ch fl1 --ch_in 3 --ch_out 1 --best_select True --scale 10.0 --extra False

