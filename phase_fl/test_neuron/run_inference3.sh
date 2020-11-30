# Cor-FL1_FL2-net-Unet-bone-efficientnetb7-pre-True-epoch-800-batch-2-lr-0.0001-dim-1024-train-None-rot-50.0-set-neuron_trn_tst_v2-subset-train-loss-mse-act-relu-scale-100-decay-0.8-delta-10-chi-3-cho-3-chf-fl2-bselect-True-Scr
# Cor-FL1_FL2-net-Unet-bone-efficientnetb7-pre-True-epoch-800-batch-2-lr-0.0001-dim-1024-train-None-rot-50.0-set-neuron_trn_tst_v2-subset-train-loss-mse-act-relu-scale-100-decay-0.8-delta-10-chi-3-cho-3-chf-fl1-bselect-True-Scr
# Cor-FL1_FL2-net-Unet-bone-efficientnetb7-pre-True-epoch-800-batch-4-lr-0.0001-dim-512-train-None-rot-50.0-set-neuron_trn_tst_v2-subset-train-loss-mse-act-relu-scale-100-decay-0.8-delta-10-chi-3-cho-3-chf-fl2-bselect-True
# Cor-FL1_FL2-net-Unet-bone-efficientnetb7-pre-True-epoch-800-batch-4-lr-0.0001-dim-512-train-None-rot-50.0-set-neuron_trn_tst_v2-subset-train-loss-mse-act-relu-scale-100-decay-0.8-delta-10-chi-3-cho-3-chf-fl1-bselect-True
# Cor-FL1_FL2-net-Unet-bone-efficientnetb7-pre-True-epoch-800-batch-4-lr-0.0005-dim-512-train-None-rot-50.0-set-neuron_trn_tst_v2-subset-train-loss-mse-act-relu-scale-100-decay-0.8-delta-10-chi-3-cho-3-chf-fl2-bselect-True
# Cor-FL1_FL2-net-Unet-bone-efficientnetb7-pre-True-epoch-800-batch-4-lr-0.0005-dim-512-train-None-rot-50.0-set-neuron_trn_tst_v2-subset-train-loss-mse-act-relu-scale-100-decay-0.8-delta-10-chi-3-cho-3-chf-fl1-bselect-True
# Cor-FL1_FL2-net-Unet-bone-efficientnetb7-pre-True-epoch-800-batch-4-lr-5e-05-dim-512-train-None-rot-50.0-set-neuron_trn_tst_v2-subset-train-loss-mse-act-relu-scale-100-decay-0.8-delta-10-chi-3-cho-3-chf-fl1-bselect-True-Scr
# Cor-FL1_FL2-net-Unet-bone-efficientnetb6-pre-True-epoch-800-batch-4-lr-0.0001-dim-512-train-None-rot-50.0-set-neuron_trn_tst_v2-subset-train-loss-mse-act-relu-scale-100-decay-0.8-delta-10-chi-3-cho-3-chf-fl1-bselect-True-Scr
# Cor-FL1_FL2-net-Unet-bone-efficientnetb3-pre-True-epoch-400-batch-4-lr-0.0001-dim-512-train-None-rot-50.0-set-neuron_trn_tst_v2-subset-train-loss-mse-act-relu-scale-100-decay-0.8-delta-10-chi-3-cho-3-chf-fl12-bselect-True-Scr

# python predict_neuron2.py --gpu 0 --model_index 30 --save False --epoch 52 --train True
# python predict_neuron2.py --gpu 0 --model_index 29 --save False --epoch 56 --train True
# python predict_neuron2.py --model_file model_list2.txt --gpu 2 --model_index 0 --save False --epoch 18
# python predict_neuron2.py --model_file model_list2.txt --gpu 2 --model_index 1 --save False --epoch 17
# python predict_neuron2.py --model_file model_list2.txt --gpu 2 --model_index 2 --save False --epoch 56
# python predict_neuron2.py --model_file model_list2.txt --gpu 2 --model_index 3 --save False --epoch 47
# python predict_neuron2.py --model_file model_list2.txt --gpu 2 --model_index 4 --save False --epoch 37
# python predict_neuron2.py --model_file model_list2.txt --gpu 2 --model_index 6 --save False --epoch 63
# python predict_neuron2.py --model_file model_list2.txt --gpu 2 --model_index 7 --save False --epoch 64
# python predict_neuron2.py --model_file model_list2.txt --gpu 2 --model_index 8 --save False --epoch 24

## check results
# python predict_neuron2.py --model_file model_list2.txt --gpu 2 --model_index 4 --save False --epoch 107
python predict_neuron2.py --model_file model_list2.txt --gpu 1 --model_index 0 --save False --epoch 89
# python predict_neuron2.py --model_file model_list2.txt --gpu 2 --model_index 7 --save False --epoch 135
# python predict_neuron2.py --model_file model_list2.txt --gpu 2 --model_index 1 --save False --epoch 30
# python predict_neuron2.py --model_file model_list2.txt --gpu 2 --model_index 6 --save False --epoch 134

## generate results
# python predict_neuron2.py --model_file model_list2.txt --gpu 2 --model_index 6 --save True --train True --epoch 134
# python predict_neuron2.py --model_file model_list2.txt --gpu 2 --model_index 0 --save True --train True --epoch 23
# python predict_neuron2.py --model_file model_list2.txt --gpu 2 --model_index 2 --save True --train True --epoch 71
# python predict_neuron2.py --model_file model_list2.txt --gpu 2 --model_index 6 --save True --train True --epoch 76