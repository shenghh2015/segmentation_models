import os
import cv2
from skimage import io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import segmentation_models_v1 as sm
from segmentation_models_v1 import Unet, Linknet, PSPNet, FPN, AtUnet

from helper_function import plot_history_flu, plot_set_prediction, generate_folder, plot_prediction_zx
from helper_function import precision, recall, f1_score, calculate_psnr, calculate_pearsonr
from helper_function import plot_flu_prediction, plot_psnr_histogram

sm.set_framework('tf.keras')
import glob
from natsort import natsorted
import time

def read_txt(txt_dir):
    lines = []
    with open(txt_dir, 'r+') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

model_root_folder = '/data/models_fl/'

parser = argparse.ArgumentParser()
parser.add_argument("--model_index", type=int, default = 1)
parser.add_argument("--gpu", type=str, default = '0')
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# model_pools = ['phase_fl-net-Unet-bone-efficientnetb0-pre-True-epoch-100-batch-4-lr-0.0005-dim-800-train-None-rot-10.0-set-neuron_x2-subset-train3_249-loss-mse-act-relu-scale-100-decay-0.8-delta-10',
# 			   'phase_fl-net-Unet-bone-efficientnetb1-pre-True-epoch-100-batch-4-lr-0.0005-dim-800-train-None-rot-10.0-set-neuron_x2-subset-train3_249-loss-mse-act-relu-scale-100-decay-0.8-delta-10',
# 			   'phase_fl-net-Unet-bone-efficientnetb2-pre-True-epoch-100-batch-4-lr-0.0005-dim-800-train-None-rot-10.0-set-neuron_x2-subset-train3_249-loss-mse-act-relu-scale-100-decay-0.8-delta-10',
# 			   'phase_fl-net-Unet-bone-efficientnetb3-pre-True-epoch-100-batch-4-lr-0.0005-dim-800-train-None-rot-10.0-set-neuron_x2-subset-train3_249-loss-mse-act-relu-scale-100-decay-0.8-delta-10']

# model_name = model_pools[args.model_index]
# model_name = 'phase_fl-net-Unet-bone-efficientnetb7-pre-True-epoch-100-batch-4-lr-0.0005-dim-512-train-None-rot-20.0-set-neuron_x2-loss-mse-act-relu-scale-100-decay-0.8'

model_pool = read_txt('./model_list.txt')
for model_name in model_pool:
    print(model_name)
model_name = model_pool[args.model_index]
model_folder = model_root_folder+ model_name

## parse model name
splits = model_name.split('-')
dataset = 'bead_dataset'
val_dim = 608

scale = 100
subset = 'train2'
chi, cho = 3,3
separate_mode = 0
for v in range(len(splits)):
	if splits[v] == 'net':
		net_arch = splits[v+1]
	elif splits[v] == 'bone':
		backbone = splits[v+1]
	elif splits[v] == 'scale':
		scale = float(splits[v+1])
	elif splits[v] == 'act':
		act_fun = splits[v+1]
	elif splits[v] == 'scale':
		scale = int(splits[v+1])
	elif splits[v] == 'set':
		dataset = splits[v+1]
	elif splits[v] == 'subset':
		subset = splits[v+1]
	elif splits[v] == 'chi':
		chi = int(splits[v+1])
	elif splits[v] == 'cho':
		cho = int(splits[v+1])
	elif splits[v] == 'sep':
		separate_mode = int(splits[v+1])

DATA_DIR = '/data/datasets/neuron_dataset_x{}'.format(dataset[-1])
if dataset == 'bead_dataset' or dataset == 'bead_dataset_v2':
    val_dim = 608
elif dataset == 'neuron_x2':
    val_dim = 896
    offset = 12
elif dataset == 'neuron_x1':
    val_dim = 1760  # 1744
    offset = 8
    
volume_fns = [fn for fn in os.listdir(DATA_DIR) if 'output' in fn]

train_fns = read_txt(DATA_DIR+'/train_list.txt')+read_txt(DATA_DIR+'/valid_list.txt')
test_fns = read_txt(DATA_DIR+'/test_list.txt')

if separate_mode == 1:
    train_fns = [fn for fn in train_fns if not 'umbeads' in fn]
    test_fns = [fn for fn in test_fns if not 'umbeads' in fn]
elif separate_mode == 2:
    train_fns = [fn for fn in train_fns if 'umbeads' in fn]
    test_fns = [fn for fn in test_fns if 'umbeads' in fn]
## volum formulation
def extract_vol(vol):
	vol_extr = []
	for i in range(vol.shape[0]):
		if i == 0:
			vol_extr.append(vol[i,:,:,0])
			vol_extr.append(vol[i,:,:,1])
		elif i>0 and i< vol.shape[0]:
			vol_extr.append(vol[i,:,:,1])
		else:
			vol_extr.append(vol[i,:,:,1])
			vol_extr.append(vol[i,:,:,2])
	return np.stack(vol_extr)

## metrics
SMOOTH= 1e-6; seed = 0
def calculate_pearsonr(vol1, vol2):
	from scipy import stats
	import numpy as np
	SMOOTH_ = np.random.random(vol1.flatten().shape)*SMOOTH
	flat1 = vol1.flatten(); flat2 = vol2.flatten().flatten()
	flat1 = SMOOTH_ + flat1; flat2 = SMOOTH_ + flat2
	score, _= stats.pearsonr(flat1, flat2)
	return score

def calculate_psnr(vol1, vol2):
	print('value in map: max {}, min {}'.format(vol1.max(),vol2.min()))
	mse = np.mean((vol1-vol2)**2)
	psnr_score = 20*np.log10(255/np.sqrt(mse))
	return psnr_score


# classes for data loading and preprocessing
class Dataset:
    """ Preprocessing transformations.
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. normalization, shape manipulation, etc.)
    """
    def __init__(
            self, 
            images_dir, 
            masks_dir,
            channels = [3,3],
            scale = 1.0,
            nb_data=None,
            augmentation=None, 
            preprocessing=None,
    ):
        id_list = natsorted(os.listdir(images_dir))
        if nb_data ==None:
            self.ids = id_list
        else:
            self.ids = id_list[:int(min(nb_data,len(id_list)))]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        print(len(self.images_fps)); print(len(self.masks_fps))
        self.channels = channels        
        self.scale = scale
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):

        image = io.imread(self.images_fps[i])
        mask  = io.imread(self.masks_fps[i])
        mask = mask/255.*self.scale

        if self.channels == [3, 1]:
            mask = mask[:,:,1:2]
        elif self.channels == [1, 1]:
            image[:,:,0], image[:,:,2]= image[:,:,1], image[:,:,1]
            mask = mask[:,:,1:2]

	    # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

	    # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask
        
    def __len__(self):
        return len(self.ids)
 
class Dataloder(tf.keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return (batch[0], batch[1])
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

import albumentations as A

# define heavy augmentations
def get_training_augmentation(dim, rot = 0):
    train_transform = [
        A.PadIfNeeded(min_height=dim, min_width=dim, always_apply=True, border_mode=0),
        A.RandomCrop(height=dim, width=dim, always_apply=True),
    ]
    return A.Compose(train_transform)

def get_validation_augmentation(dim = 992):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(dim, dim)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform 
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

# network
best_weight = model_folder+'/best_model.h5'
preprocess_input = sm.get_preprocessing(backbone)

#create model
n_classes = cho
activation = 'relu'
net_func = globals()[net_arch]
model = net_func(backbone, classes=n_classes, activation=activation)

#load best weights
model.load_weights(best_weight)

## save model
model.save(model_folder+'/ready_model.h5')

subsets = ['test', 'train']
subset = 'test'

for subset in subsets:
		vol_fns = train_fns if subset == 'train' else test_fns
		psnr_scores = []
		cor_scores = []
		mse_scores = []
		for vol_fn in vol_fns:
				# vol_fn = vol_fns[0]
				if not os.path.exists(os.path.join(DATA_DIR, vol_fn)):
				    continue
				print('{}: {}'.format(subset, vol_fn))
				X_dir = os.path.join(DATA_DIR, vol_fn,'phase')
				Y_dir = os.path.join(DATA_DIR, vol_fn,'fl2')

				test_dataset = Dataset(
					X_dir, 
					Y_dir,
					channels = [chi, cho],
					scale = scale,
					augmentation=get_validation_augmentation(val_dim),
					preprocessing=get_preprocessing(preprocess_input),
				)
        
				print(test_dataset[0][0].shape, test_dataset[0][1].shape)
        
				test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

				# prediction and ground truth
				pr_masks = model.predict(test_dataloader)
				gt_masks = []; images = []
				for i in range(len(test_dataset)):
					image, gt_mask = test_dataset[i];images.append(image); gt_masks.append(gt_mask)
				images = np.stack(images); gt_masks = np.stack(gt_masks)

				# scale to [0,255]
				gt_masks = np.uint8(gt_masks/scale*255); gt_masks = gt_masks.squeeze()
				pr_masks = pr_masks/scale*255; pr_masks = pr_masks.squeeze()
				pr_masks = np.uint8(np.clip(pr_masks, 0, 255))
				images = np.uint8(images*255)

				# extract the target prediction
				# offset = 12
				ph_vol = extract_vol(images)
				if cho == 3:
						gt_vol = extract_vol(gt_masks)
						pr_vol = extract_vol(pr_masks)
				else:
						gt_vol = gt_masks
						pr_vol = pr_masks
				pr_vol = pr_vol[:,offset:-offset,offset:-offset]
				gt_vol = gt_vol[:,offset:-offset,offset:-offset]
				ph_vol = ph_vol[:,offset:-offset,offset:-offset]

				# psnr and pearsonr correlation
				mse_score = np.mean(np.square(pr_vol-gt_vol))
				psnr_score = calculate_psnr(pr_vol, gt_vol)
				cor_score = calculate_pearsonr(pr_vol, gt_vol)
				mse_scores.append(mse_score); psnr_scores.append(psnr_score); cor_scores.append(cor_score)

				# save prediction
				pred_save = True
				if pred_save:
						pr_vol_dir = model_folder+'/pred_vols'
						generate_folder(pr_vol_dir)
						np.save(os.path.join(pr_vol_dir,'Pr_{}.npy'.format(vol_fn)), pr_vol)
						np.save(os.path.join(pr_vol_dir,'GT_{}.npy'.format(vol_fn)), gt_vol)
						print(pr_vol.shape)
				
				print('{}: psnr {:.4f}, cor {:.4f}, mse {:.4f}\n'.format(vol_fn, psnr_score, cor_score, mse_score))

				# save prediction examples
				prediction_dir = model_folder+'/pred_examples'
				generate_folder(prediction_dir)
				plot_fig_file = prediction_dir+'/{}.png'.format(vol_fn)
				z_index = 158; x_index = 250
				plot_prediction_zx(plot_fig_file, ph_vol, gt_vol, pr_vol, z_index, x_index)

		mPSNR, mCor, mMSE = np.mean(psnr_scores), np.mean(cor_scores), np.mean(mse_scores)
		print('Mean metrics: mPSNR {:.4f}, mCor {:.4f}, mMse {:.4f}\n'.format(mPSNR, mCor, mMSE))
		with open(model_folder+'/{}_summary.txt'.format(subset),'w+') as f:
				for i in range(len(psnr_scores)):
						f.write('{}: psnr {:.4f}, cor {:.4f}, mse {:.4f}\n'.format(vol_fns[i], psnr_scores[i], cor_scores[i], mse_scores[i]))
				f.write('Mean metrics: mPSNR {:.4f}, mCor {:.4f}, mMse {:.4f}\n'.format(mPSNR, mCor, mMSE))