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

def str2bool(value):
    return value.lower() == 'true'

model_root_folder = '/data/models_fl/'

parser = argparse.ArgumentParser()
parser.add_argument("--model_index", type=int, default = 0)
parser.add_argument("--gpu", type=str, default = '0')
parser.add_argument("--save", type=str2bool, default = False)
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

model_pool = read_txt('./fl1fl2_model_list.txt')
for model_name in model_pool:
    print(model_name)
model_name = model_pool[args.model_index]
model_folder = model_root_folder+ model_name

## parse model name
splits = model_name.split('-')
dataset = 'neuron_wbx1'
val_dim = 608

scale = 100
subset = 'train2'
chi, cho = 3,3

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
	elif splits[v] == 'chf':
		fl_ch = splits[v+1]

# DATA_DIR = '/data/datasets/neuron_wo_beads_x{}'.format(dataset[-1])
DATA_DIR = '/data/datasets/neuron_stacks'
if dataset == 'neuron_wbx1' or 'neuron_trn_tst':
    val_dim = 1760  # 1744
    offset = 8

volume_fns = [fn for fn in os.listdir(DATA_DIR) if 'output' in fn]

# train_fns = read_txt(DATA_DIR+'/train_sample_list.txt')
# test_fns = read_txt(DATA_DIR+'/test_sample_list.txt')

train_fns = read_txt(DATA_DIR+'/neuron_train_list.txt')
test_fns = read_txt(DATA_DIR+'/neuron_test_list.txt')

## volum formulation
def extract_vol(vol):
	vol_extr = []
	for i in range(vol.shape[0]):
		if i == 0:
			vol_extr.append(vol[i,:,:,0])
			vol_extr.append(vol[i,:,:,1])
		elif i>0 and i< vol.shape[0]-1:
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
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            images_dir, 
            masks_dir1,
            masks_dir2,
            fl_ch = None,
            scale = 1.0,
            channels = [3,3],
            nb_data=None,
            augmentation=None, 
            preprocessing=None,
    ):
        #self.ids = os.listdir(images_dir)
        id_list = natsorted(os.listdir(images_dir))
        if nb_data ==None:
            self.ids = id_list
        else:
            self.ids = id_list[:int(min(nb_data,len(id_list)))]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks1_fps = [os.path.join(masks_dir1, image_id) for image_id in self.ids]
        self.masks2_fps = [os.path.join(masks_dir2, image_id) for image_id in self.ids]
        print('Load files: image {}, fl1: {}, fl2:{}'.format(len(self.images_fps),len(self.masks1_fps),len(self.masks2_fps)))      
        self.scale = scale
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.channels = channels
        self.fl_ch = fl_ch
    
    def __getitem__(self, i):
        
        # load image and fl1 or fl2 or both
        image = io.imread(self.images_fps[i])
        if self.fl_ch == 'fl1':
            mask  = io.imread(self.masks1_fps[i])
            mask = mask/255.*self.scale
        elif self.fl_ch == 'fl2':
            mask  = io.imread(self.masks2_fps[i])
            mask = mask/255.*self.scale        
        elif self.fl_ch == 'fl12':
            mask1  = io.imread(self.masks1_fps[i])
            mask2  = io.imread(self.masks2_fps[i])
            mask = np.stack([mask1[:,:,1], mask2[:,:,1]], axis = -1)
            mask = mask/255.*self.scale
        
        # decide the input and output channels
        if self.channels[0] == 1:
            image[:,:,0], image[:,:,2] = image[:,:,1], image[:,:,1]
        elif self.channels[0] == 2:
            image[:,:,2] = image[:,:,1]		
        
        if self.channels[1] == 1 and not (self.fl_ch=='fl12'):
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
n_classes = cho if not fl_ch == 'fl12' else 2
activation = 'relu'
net_func = globals()[net_arch]
model = net_func(backbone, classes=n_classes, activation=activation)

#load best weights
model.load_weights(best_weight)

## save model
model.save(model_folder+'/ready_model.h5')

subsets = ['test', 'train']
# subset = 'test'
for subset in subsets:
		vol_fns = train_fns if subset == 'train' else test_fns
		psnr_scores = []; psnr2_scores = []
		cor_scores = []; cor2_scores = []
		mse_scores = []; mse2_scores = []
		for vol_fn in vol_fns:
				# vol_fn = vol_fns[0]
				if not os.path.exists(os.path.join(DATA_DIR, vol_fn)):
						continue
				print('{}: {}'.format(subset, vol_fn))
				X_dir = os.path.join(DATA_DIR, vol_fn,'phase')
				Y1_dir = os.path.join(DATA_DIR, vol_fn,'fl1')
				Y2_dir = os.path.join(DATA_DIR, vol_fn,'fl2')

				test_dataset = Dataset(
					X_dir,
					Y1_dir,
					Y2_dir,
					fl_ch = fl_ch,
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

				# obtain the images, GT, and prediction
				ph_vol = extract_vol(images)
				ph_vol = ph_vol[:,offset:-offset,offset:-offset]

				# psnr and pearsonr correlation
				if fl_ch == 'fl1':
						if cho == 3:
								gt_vol = extract_vol(gt_masks)
								pr_vol = extract_vol(pr_masks)
						else:
								gt_vol = gt_masks
								pr_vol = pr_masks
				elif fl_ch == 'fl2':
						if cho == 3:
								gt_vol2 = extract_vol(gt_masks)
								pr_vol2 = extract_vol(pr_masks)
						else:
								gt_vol2 = gt_masks
								pr_vol2 = pr_masks
				elif fl_ch == 'fl12':
						gt_vol = gt_masks[:,:,:,0]
						pr_vol = pr_masks[:,:,:,0]
						gt_vol2 = gt_masks[:,:,:,1]
						pr_vol2 = pr_masks[:,:,:,1]

				if fl_ch == 'fl12' or fl_ch == 'fl1':
						pr_vol = pr_vol[:,offset:-offset,offset:-offset]
						gt_vol = gt_vol[:,offset:-offset,offset:-offset]
						mse_score = np.mean(np.square(pr_vol-gt_vol))
						psnr_score = calculate_psnr(pr_vol, gt_vol)
						cor_score = calculate_pearsonr(pr_vol, gt_vol)
						mse_scores.append(mse_score); psnr_scores.append(psnr_score); cor_scores.append(cor_score)
						print('{}-FL1: psnr {:.4f}, cor {:.4f}, mse {:.4f}\n'.format(vol_fn, psnr_score, cor_score, mse_score))
				if fl_ch == 'fl12' or fl_ch == 'fl2':
						pr_vol2 = pr_vol2[:,offset:-offset,offset:-offset]
						gt_vol2 = gt_vol2[:,offset:-offset,offset:-offset]
						mse_score2 = np.mean(np.square(pr_vol2-gt_vol2))
						psnr_score2 = calculate_psnr(pr_vol2, gt_vol2)
						cor_score2 = calculate_pearsonr(pr_vol2, gt_vol2)
						mse2_scores.append(mse_score2); psnr2_scores.append(psnr_score2); cor2_scores.append(cor_score2)
						print('{}-FL2: psnr {:.4f}, cor {:.4f}, mse {:.4f}\n'.format(vol_fn, psnr_score2, cor_score2, mse_score2))

				# save prediction
				pred_save = args.save
				if pred_save:
						pr_vol_dir = model_folder+'/pred_fl1_fl2'
						generate_folder(pr_vol_dir)
						if fl_ch == 'fl12' or fl_ch == 'fl1':				
								np.save(os.path.join(pr_vol_dir,'Pr1_{}.npy'.format(vol_fn)), pr_vol)
								np.save(os.path.join(pr_vol_dir,'GT1_{}.npy'.format(vol_fn)), gt_vol)
								print('FL1: {}'.format(pr_vol.shape))
						if fl_ch == 'fl12' or fl_ch == 'fl2':
								np.save(os.path.join(pr_vol_dir,'Pr2_{}.npy'.format(vol_fn)), pr_vol2)
								np.save(os.path.join(pr_vol_dir,'GT2_{}.npy'.format(vol_fn)), gt_vol2)
								print('FL2: {}'.format(pr_vol2.shape))

				# save prediction examples
				prediction_dir = model_folder+'/pred_examples'
				generate_folder(prediction_dir)
				plot_fig_file = prediction_dir+'/{}_fl1.png'.format(vol_fn)
				plot_fig_file2 = prediction_dir+'/{}_fl2.png'.format(vol_fn)
				if gt_vol.shape[0]>150:
						z_index = 158; x_index = 250
				else:
						z_index = 60; x_index = 250
				if fl_ch == 'fl12' or fl_ch == 'fl1':			
						plot_prediction_zx(plot_fig_file, ph_vol, gt_vol, pr_vol, z_index, x_index)
				if fl_ch == 'fl12' or fl_ch == 'fl2':
						plot_prediction_zx(plot_fig_file2, ph_vol, gt_vol2, pr_vol2, z_index, x_index)

		if fl_ch == 'fl12' or fl_ch == 'fl1':	
			mPSNR, mCor, mMSE = np.mean(psnr_scores), np.mean(cor_scores), np.mean(mse_scores)
			print('Mean metrics on FL1: mPSNR {:.4f}, mCor {:.4f}, mMse {:.4f}\n'.format(mPSNR, mCor, mMSE))
			with open(model_folder+'/{}_FL1_summary.txt'.format(subset),'w+') as f:
					for i in range(len(psnr_scores)):
							f.write('{} FL1: psnr {:.4f}, cor {:.4f}, mse {:.4f}\n'.format(vol_fns[i], psnr_scores[i], cor_scores[i], mse_scores[i]))
					f.write('Mean metrics on FL1: mPSNR {:.4f}, mCor {:.4f}, mMse {:.4f}\n'.format(mPSNR, mCor, mMSE))
		if fl_ch == 'fl12' or fl_ch == 'fl2':
			mPSNR, mCor, mMSE = np.mean(psnr2_scores), np.mean(cor2_scores), np.mean(mse2_scores)
			print('Mean metrics on FL2: mPSNR {:.4f}, mCor {:.4f}, mMse {:.4f}\n'.format(mPSNR, mCor, mMSE))
			with open(model_folder+'/{}_FL2_summary.txt'.format(subset),'w+') as f:
					for i in range(len(psnr2_scores)):
							f.write('{} FL2: psnr {:.4f}, cor {:.4f}, mse {:.4f}\n'.format(vol_fns[i], psnr2_scores[i], cor2_scores[i], mse2_scores[i]))
					f.write('Mean metrics on FL2: mPSNR {:.4f}, mCor {:.4f}, mMse {:.4f}\n'.format(mPSNR, mCor, mMSE))
