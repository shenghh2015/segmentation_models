# parser.add_argument("--model_index", type=int, default = 0)
# parser.add_argument("--gpu", type=str, default = '0')
# parser.add_argument("--model_file", type=str, default = 'model_list.txt')
# parser.add_argument("--epoch", type=int, default = -1)
# parser.add_argument("--save", type=str2bool, default = False)
# parser.add_argument("--train", type=str2bool, default = False)
python predict_neuron3.py --gpu 2 --model_file model_list4.txt --model_index 2 --epoch 116
python predict_neuron3.py --gpu 2 --model_file model_list4.txt --model_index 5 --epoch 299
python predict_neuron3.py --gpu 2 --model_file model_list4.txt --model_index 6 --epoch 107
 