# python predict_for_report.py --model_file neuron_list.txt --gpu 0 --model_index 0 --save True --train True --epoch 203
# python predict_for_report.py --model_file neuron_list.txt --gpu 1 --model_index 1 --save True --train True --epoch 118
python predict_exp_neuron.py --model_file neuron_list.txt --gpu 2 --model_index 0 --save False --train False --epoch 84
python predict_exp_neuron.py --model_file neuron_list.txt --gpu 2 --model_index 1 --save False --train False --epoch 42