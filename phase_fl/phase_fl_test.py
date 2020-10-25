import os
import cv2
from skimage import io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import segmentation_models_v1 as sm
from segmentation_models_v1 import Unet, Linknet, PSPNet, FPN, AtUnet

from helper_function import plot_history_flu, plot_set_prediction, generate_folder
from helper_function import precision, recall, f1_score, calculate_psnr, calculate_pearsonr
from helper_function import plot_flu_prediction, plot_psnr_histogram

sm.set_framework('tf.keras')
import glob
from natsort import natsorted

model_root_folder = '/data/models_fl/'

parser = argparse.ArgumentParser()
parser.add_argument("--model_index", type=int, default = 0)
parser.add_argument("--gpu", type=str, default = '2')
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
model_pools = ['phase_fl-net-Unet-bone-efficientnetb0-pre-True-epoch-100-batch-14-lr-0.0005-dim-512-train-None-rot-20.0-set-neuron_x2-loss-mae-act-relu-scale-100-decay-0.8',
			   'phase_fl-net-Unet-bone-efficientnetb1-pre-True-epoch-100-batch-14-lr-0.0005-dim-512-train-None-rot-20.0-set-neuron_x2-loss-mae-act-relu-scale-100-decay-0.8',
			   'phase_fl-net-Unet-bone-efficientnetb2-pre-True-epoch-100-batch-14-lr-0.0005-dim-512-train-None-rot-20.0-set-neuron_x2-loss-mae-act-relu-scale-100-decay-0.8',
			   'phase_fl-net-Unet-bone-efficientnetb3-pre-True-epoch-100-batch-14-lr-0.0005-dim-512-train-None-rot-20.0-set-neuron_x2-loss-mae-act-relu-scale-100-decay-0.8',
			   'phase_fl-net-Unet-bone-efficientnetb0-pre-True-epoch-100-batch-14-lr-0.0005-dim-512-train-None-rot-20.0-set-neuron_x2-loss-mse-act-relu-scale-100-decay-0.8',
			   'phase_fl-net-Unet-bone-efficientnetb1-pre-True-epoch-100-batch-14-lr-0.0005-dim-512-train-None-rot-20.0-set-neuron_x2-loss-mse-act-relu-scale-100-decay-0.8',
			   'phase_fl-net-Unet-bone-efficientnetb2-pre-True-epoch-100-batch-14-lr-0.0005-dim-512-train-None-rot-20.0-set-neuron_x2-loss-mse-act-relu-scale-100-decay-0.8',
			   'phase_fl-net-Unet-bone-efficientnetb3-pre-True-epoch-100-batch-14-lr-0.0005-dim-512-train-None-rot-20.0-set-neuron_x2-loss-mse-act-relu-scale-100-decay-0.8']

# model_pools = ['phase_fl-net-AtUnet-bone-efficientnetb0-pre-True-epoch-100-batch-14-lr-0.0005-dim-512-train-None-rot-20.0-set-neuron_x2-loss-mse-act-relu-scale-100-decay-0.8',
# 			   'phase_fl-net-AtUnet-bone-efficientnetb1-pre-True-epoch-100-batch-14-lr-0.0005-dim-512-train-None-rot-20.0-set-neuron_x2-loss-mse-act-relu-scale-100-decay-0.8',
# 			   'phase_fl-net-AtUnet-bone-efficientnetb2-pre-True-epoch-100-batch-14-lr-0.0005-dim-512-train-None-rot-20.0-set-neuron_x2-loss-mse-act-relu-scale-100-decay-0.8',
# 			   'phase_fl-net-AtUnet-bone-efficientnetb3-pre-True-epoch-100-batch-14-lr-0.0005-dim-512-train-None-rot-20.0-set-neuron_x2-loss-mse-act-relu-scale-100-decay-0.8',
# 			   'phase_fl-net-Unet-bone-efficientnetb0-pre-True-epoch-100-batch-6-lr-0.0005-dim-512-train-None-rot-20.0-set-neuron_x2-loss-mse-act-relu-scale-100',
# 			   'phase_fl-net-Unet-bone-efficientnetb2-pre-True-epoch-100-batch-14-lr-0.0005-dim-512-train-None-rot-20.0-set-neuron_x2-loss-mse-act-relu-scale-100-decay-0.8',
# 			   'phase_fl-net-Unet-bone-efficientnetb3-pre-True-epoch-100-batch-14-lr-0.0005-dim-512-train-None-rot-20.0-set-neuron_x2-loss-mse-act-relu-scale-100-decay-0.8',
# 			   'phase_fl-net-Unet-bone-efficientnetb4-pre-True-epoch-100-batch-8-lr-0.0005-dim-512-train-None-rot-20.0-set-neuron_x2-loss-mse-act-relu-scale-100-decay-0.8',
# 			   'phase_fl-net-Unet-bone-efficientnetb5-pre-True-epoch-100-batch-6-lr-0.0005-dim-512-train-None-rot-20.0-set-neuron_x2-loss-mse-act-relu-scale-100-decay-0.8']

#model_name = 'phase_fl-net-Unet-bone-efficientnetb0-pre-True-epoch-100-batch-6-lr-0.0005-dim-320-train-None-rot-0-set-bead_dataset_v2-loss-mse-act-relu-scale-100'
#model_name = 'phase_fl-net-Unet-bone-efficientnetb0-pre-True-epoch-100-batch-6-lr-0.0005-dim-512-train-None-rot-20.0-set-bead_dataset_v2-loss-mse-act-relu-scale-100'
#model_name = 'phase_fl-net-Unet-bone-efficientnetb1-pre-True-epoch-200-batch-6-lr-0.0005-dim-512-train-None-rot-20.0-set-neuron_x2-loss-mse-act-relu-scale-100'
model_name = model_pools[args.model_index]
model_folder = model_root_folder+model_name

## parse model name
splits = model_name.split('-')
dataset = 'bead_dataset'
val_dim = 608

scale = 100
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

DATA_DIR = '/data/datasets/{}'.format(dataset)
if dataset == 'bead_dataset' or dataset == 'bead_dataset_v2':
	x_train_dir = os.path.join(DATA_DIR, 'train_phase')
	y_train_dir = os.path.join(DATA_DIR, 'train_fl')

	x_valid_dir = os.path.join(DATA_DIR, 'test_phase')
	y_valid_dir = os.path.join(DATA_DIR, 'test_fl')

	x_test_dir = x_valid_dir
	y_test_dir = y_valid_dir
	val_dim = 608
else:
	x_train_dir = os.path.join(DATA_DIR, 'train/phase')
	y_train_dir = os.path.join(DATA_DIR, 'train/fl')

	x_valid_dir = os.path.join(DATA_DIR, 'valid/phase')
	y_valid_dir = os.path.join(DATA_DIR, 'valid/fl')

	x_test_dir = os.path.join(DATA_DIR, 'test/phase')
	y_test_dir = os.path.join(DATA_DIR, 'test/fl')
	val_dim = 896

print(y_train_dir)

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
            masks_dir,
            scale = 1.0,
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
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        print(len(self.images_fps)); print(len(self.masks_fps))        
        self.scale = scale
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):

        # read data
#         image = cv2.imread(self.images_fps[i])
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(self.masks_fps[i], cv2.COLOR_BGR2RGB)
        image = io.imread(self.images_fps[i])
        mask  = io.imread(self.masks_fps[i])
        mask = mask/255.*self.scale
        
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

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation(dim, rot = 0):
    train_transform = [

#         A.HorizontalFlip(p=0.5),
# 
#         A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=rot, shift_limit=0.1, p=1, border_mode=0),
# 
        A.PadIfNeeded(min_height=dim, min_width=dim, always_apply=True, border_mode=0),
        A.RandomCrop(height=dim, width=dim, always_apply=True),
# 
#         A.IAAAdditiveGaussianNoise(p=0.2),
#         A.IAAPerspective(p=0.5),
# 
#         A.OneOf(
#             [
#                 A.CLAHE(p=1),
#                 A.RandomBrightness(p=1),
#                 A.RandomGamma(p=1),
#             ],
#             p=0.9,
#         ),
# 
#         A.OneOf(
#             [
#                 A.IAASharpen(p=1),
#                 A.Blur(blur_limit=3, p=1),
#                 A.MotionBlur(blur_limit=3, p=1),
#             ],
#             p=0.9,
#         ),
# 
#         A.OneOf(
#             [
#                 A.RandomContrast(p=1),
#                 A.HueSaturationValue(p=1),
#             ],
#             p=0.9,
#         ),
#         A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation(dim = 992):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(dim, dim)
#         A.PadIfNeeded(384, 480)
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
# CLASSES = ['live', 'inter', 'dead']
n_classes = 3
activation = 'relu'
net_func = globals()[net_arch]
model = net_func(backbone, classes=n_classes, activation=activation)

#load best weights
model.load_weights(best_weight)

## save model
model.save(model_folder+'/ready_model.h5')
# evaluate model
# subsets = ['train', 'val', 'test']
# subsets = ['train', 'test']
# subsets = ['test']
# subset = subsets[2]
subset = 'test'
# for subset in subsets:
# subset
if subset == 'val':
	x_test_dir = x_valid_dir; y_test_dir = y_valid_dir
elif subset == 'train':
	x_test_dir = x_train_dir; y_test_dir = y_train_dir

test_dataset = Dataset(
	x_test_dir, 
	y_test_dir,
	scale = scale,
	augmentation=get_validation_augmentation(val_dim),
	preprocessing=get_preprocessing(preprocess_input),
)

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)
## evaluate the performance
# calculate the pixel-level classification performance
pr_masks = model.predict(test_dataloader)
# scale back to [0,1]
pr_masks = pr_masks/scale
gt_masks = []
for i in range(len(test_dataset)):
	_, gt_mask = test_dataset[i];gt_masks.append(gt_mask)
gt_masks = np.stack(gt_masks)

# save prediction examples
pr_masks = model.predict(test_dataloader)
gt_masks = []; images = []
for i in range(len(test_dataset)):
	image, gt_mask = test_dataset[i];images.append(image); gt_masks.append(gt_mask)
images = np.stack(images); gt_masks = np.stack(gt_masks)

# scale back from args.flu_scale
gt_masks = np.uint8(gt_masks/scale*255)
pr_masks = pr_masks/scale*255
pr_masks = np.uint8(np.clip(pr_masks, 0, 255))

# save prediction examples
plot_fig_file = model_folder+'/pred_examples.png'; nb_images = 8
plot_flu_prediction(plot_fig_file, images, gt_masks, pr_masks, nb_images)
## save prediction results
pred_folder = model_folder+'/pred_fl_only'; generate_folder(pred_folder)
for i in range(gt_masks.shape[0]):
	io.imsave(pred_folder+'/p{}'.format(test_dataset.ids[i]), pr_masks[i,:,:])
# output_dir = model_folder+'/pred_fl'; generate_folder(output_dir)
# plot_set_prediction(output_dir, images, gt_masks, pr_masks)
# calculate PSNR
mPSNR, psnr_scores = calculate_psnr(gt_masks, pr_masks)
print('PSNR: {:.4f}'.format(mPSNR))

# calculate Pearson correlation coefficient
mPear, pear_scores = calculate_pearsonr(gt_masks, pr_masks)
print('Pearsonr:{:.4f}'.format(mPear))

with open(model_folder+'/{}_metric_summary.txt'.format(subset),'w+') as f:
	# save PSNR over fluorescent 1 and fluorescent 2
	f.write('PSNR: {:.4f}\n'.format(mPSNR))
	f.write('Pearsonr:{:.4f}\n'.format(mPear))
