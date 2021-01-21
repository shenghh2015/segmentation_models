cd ~/segmentation_models/cell_cycle/test_life
python eval_cyc2.py --gpu 0 --model_file model_cyc2.txt --model_index 0
python eval_cyc2.py --gpu 0 --model_file model_cyc2.txt --model_index 1
python eval_cyc2.py --gpu 0 --model_file model_cyc2.txt --model_index 2
python eval_cyc2.py --gpu 0 --model_file model_cyc2.txt --model_index 3