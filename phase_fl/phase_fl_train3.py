import os
import cv2
from skimage import io
import sys
# import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
from natsort import natsorted
# sys.path.append('../')
import segmentation_models_v1 as sm
from segmentation_models_v1 import Unet, Linknet, PSPNet, FPN, AtUnet
sm.set_framework('tf.keras')

from helper_function import plot_history_flu, plot_flu_prediction, plot_set_prediction
from helper_function import precision, recall, f1_score, calculate_psnr, calculate_pearsonr
from sklearn.metrics import confusion_matrix

def str2bool(value):
    return value.lower() == 'true'

def generate_folder(folder_name):
	if not os.path.exists(folder_name):
		os.system('mkdir -p {}'.format(folder_name))

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default = '0')
parser.add_argument("--docker", type=str2bool, default = True)
parser.add_argument("--net_type", type=str, default = 'Unet')  #Unet, Linknet, PSPNet, FPN
parser.add_argument("--backbone", type=str, default = 'efficientnetb0')
parser.add_argument("--dataset", type=str, default = 'bead_dataset')
parser.add_argument("--subset", type=str, default = 'train2')
parser.add_argument("--epoch", type=int, default = 10)
parser.add_argument("--dim", type=int, default = 512)
parser.add_argument("--ch_in", type=int, default = 3)
parser.add_argument("--ch_out", type=int, default = 3)
parser.add_argument("--rot", type=float, default = 0)
parser.add_argument("--scale", type=float, default = 100)
parser.add_argument("--train", type=int, default = None)
parser.add_argument("--act_fun", type=str, default = 'relu')
parser.add_argument("--loss", type=str, default = 'mse')
parser.add_argument("--batch_size", type=int, default = 6)
parser.add_argument("--lr", type=float, default = 5e-4)
parser.add_argument("--decay", type=float, default = 0.8)
parser.add_argument("--delta", type=float, default = 10)
parser.add_argument("--pre_train", type=str2bool, default = True)
args = parser.parse_args()
print(args)

model_name = 'phase_fl-net-{}-bone-{}-pre-{}-epoch-{}-batch-{}-lr-{}-dim-{}-train-{}-rot-{}-set-{}-subset-{}-loss-{}-act-{}-scale-{}-decay-{}-delta-{}-chi-{}-cho-{}'.format(args.net_type, args.backbone, args.pre_train,\
		 args.epoch, args.batch_size, args.lr, args.dim, args.train, args.rot, args.dataset, args.subset, args.loss, args.act_fun, args.scale, args.decay, args.delta, args.ch_in, args.ch_out)
print(model_name)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

DATA_DIR = '/data/datasets/{}'.format(args.dataset) if args.docker else './data/{}'.format(args.dataset)
train_dim = args.dim

if args.dataset == 'bead_dataset':
	x_train_dir = os.path.join(DATA_DIR, 'train_phase')
	y_train_dir = os.path.join(DATA_DIR, 'train_fl')

	x_valid_dir = os.path.join(DATA_DIR, 'test_phase')
	y_valid_dir = os.path.join(DATA_DIR, 'test_fl')

	x_test_dir = x_valid_dir
	y_test_dir = y_valid_dir
	val_dim = 608
else:
	x_train_dir = os.path.join(DATA_DIR, '{}/phase'.format(args.subset))
	y_train_dir = os.path.join(DATA_DIR, '{}/fl'.format(args.subset))

	x_valid_dir = os.path.join(DATA_DIR, 'test/phase')
	y_valid_dir = os.path.join(DATA_DIR, 'test/fl')

	x_test_dir = os.path.join(DATA_DIR, 'test/phase')
	y_test_dir = os.path.join(DATA_DIR, 'test/fl')
	if args.dataset == 'neuron_x1':
		val_dim = 1792
	elif args.dataset == 'neuron_x2':
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
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        print(len(self.images_fps)); print(len(self.masks_fps))        
        self.scale = scale
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.channels = channels
    
    def __getitem__(self, i):
        image = io.imread(self.images_fps[i])
        mask  = io.imread(self.masks_fps[i])
        mask = mask/255.*self.scale     
        if self.channels[0] == 1:
            image[:,:,0], image[:,:,2] = image[:,:,1], image[:,:,1]
        elif self.channels[0] == 2:
            image[:,:,2] = image[:,:,1]		
        
        if self.channels[1] == 1:
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


# BACKBONE = 'efficientnetb3'
BACKBONE = args.backbone
BATCH_SIZE = args.batch_size
LR = args.lr
EPOCHS = args.epoch

preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
n_classes = args.ch_out
activation = '{}'.format(args.act_fun)

#create model
net_func = globals()[args.net_type]

encoder_weights='imagenet' if args.pre_train else None

model = net_func(BACKBONE, encoder_weights=encoder_weights, classes=n_classes, activation=activation)

# define optomizer
optim = tf.keras.optimizers.Adam(LR)

if args.loss == 'mse':
	loss = tf.keras.losses.MSE
elif args.loss == 'mae':
	loss = tf.keras.losses.MAE
elif args.loss == 'huber':
	loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)

metrics = [sm.metrics.PSNR(max_val=args.scale)]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, loss, metrics)

# Dataset for train images
train_dataset = Dataset(
    x_train_dir, 
    y_train_dir,
    channels = [args.ch_in, args.ch_out],
    scale = args.scale,
    nb_data=args.train, 
    augmentation=get_training_augmentation(train_dim, args.rot),
    preprocessing=get_preprocessing(preprocess_input),
)

# Dataset for validation images
valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir,
    scale = args.scale,
    channels = [args.ch_in, args.ch_out],
    augmentation=get_validation_augmentation(val_dim),
    preprocessing=get_preprocessing(preprocess_input),
)

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

print(train_dataloader[0][0].shape)
print(train_dataloader[0][1].shape)
# check shapes for errors
assert train_dataloader[0][0].shape == (BATCH_SIZE, train_dim, train_dim, 3)
assert train_dataloader[0][1].shape == (BATCH_SIZE, train_dim, train_dim, n_classes)

model_folder = '/data/models_fl/{}'.format(model_name) if args.docker else './models/{}'.format(model_name)
generate_folder(model_folder)


# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(model_folder+'/best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
    tf.keras.callbacks.ReduceLROnPlateau(factor=args.decay),
]

# train model
history = model.fit_generator(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
)

# save the training information
plot_history_flu(model_folder+'/train_history.png',history)

# evaluate model
test_dataset = Dataset(
    x_test_dir, 
    y_test_dir,
    channels = [args.ch_in, args.ch_out],
    scale = args.scale,
    augmentation=get_validation_augmentation(val_dim),
    preprocessing=get_preprocessing(preprocess_input),
)

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)
# load best weights
model.load_weights(model_folder+'/best_model.h5')
model.save(model_folder+'/ready_model.h5')
scores = model.evaluate_generator(test_dataloader)
print("Loss: {:.5}".format(scores[0]))
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value))

# calculate the pixel-level classification performance
pr_masks = model.predict(test_dataloader)
gt_masks = []; images = []
for i in range(len(test_dataset)):
    image, gt_mask = test_dataset[i];images.append(image); gt_masks.append(gt_mask)
images = np.stack(images); gt_masks = np.stack(gt_masks)

# scale back from args.flu_scale
gt_masks = np.uint8(gt_masks/args.scale*255)
pr_masks = pr_masks/args.scale*255
pr_masks = np.uint8(np.clip(pr_masks, 0, 255))

# save prediction examples
plot_fig_file = model_folder+'/pred_examples.png'; nb_images = 10
plot_flu_prediction(plot_fig_file, images, gt_masks, pr_masks, nb_images)
# output_dir = model_folder+'/pred_fl'; generate_folder(output_dir)
# plot_set_prediction(output_dir, images, gt_masks, pr_masks)
# calculate PSNR
mPSNR, psnr_scores = calculate_psnr(gt_masks, pr_masks)
print('PSNR: {:.4f}'.format(mPSNR))

# calculate Pearson correlation coefficient
mPear, pear_scores = calculate_pearsonr(gt_masks, pr_masks)
print('Pearsonr:{:.4f}'.format(mPear))

with open(model_folder+'/metric_summary.txt','w+') as f:
    # loss
    f.write("loss {}: {:.5}\n".format(metric.__name__, value))
    # average psnr
    for metric, value in zip(metrics, scores[1:]):
        f.write("mean {}: {:.5}\n".format(metric.__name__, value))
    # save PSNR over fluorescent 1 and fluorescent 2
        f.write('PSNR: {:.4f}\n'.format(mPSNR))
        f.write('Pearsonr:{:.4f}\n'.format(mPear))