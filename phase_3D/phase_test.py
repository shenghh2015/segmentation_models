import os
import cv2
from skimage import io
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
from natsort import natsorted
import random
from model import unet_model_3d

def str2bool(value):
    return value.lower() == 'true'

def generate_folder(folder_name):
	if not os.path.exists(folder_name):
		os.system('mkdir -p {}'.format(folder_name))

def read_txt(txt_dir):
    lines = []
    with open(txt_dir, 'r+') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

def combine_xy(voi_list, xl, yl, zl):
    xy = np.zeros((xl, yl, zl))
    for i in range(7):
        xi, xe = 256*i, min(256*(i+1),xl)
        print(xi, xe)
        if i < 6:
            voi_xi, voi_xe = 0, 256
        else:
            voi_xi, voi_xe = 256-(xl-256*i), 256
        for j in range(7):
            yi, ye = 256*j, min(256*(j+1),yl)
            print(yi, ye)
            if j < 6:
                voi_yi, voi_ye = 0, 256
            else:
                voi_yi, voi_ye = 256-(yl-256*j), 256
            voi = voi_list[i*7+j]
            xy[xi:xe,yi:ye,:] = voi[voi_xi:voi_xe,voi_yi:voi_ye,:]
            print(xy[xi:xe,yi:ye,:].shape, voi[voi_xi:voi_xe,voi_yi:voi_ye,:].shape)
    return xy

def combine_z(voi_list, xl, yl, zl):
    vol = np.zeros((xl, yl, zl))
    for i in range(4):
        zi, ze = 32*i, min(32*(i+1), zl)
        if i < 3:
            voi_zi, voi_ze = 0, 32
        else:
            voi_zi, voi_ze = 32*(i+1)-zl, 32
        voi = voi_list[i]
        vol[:,:,zi:ze] = voi[:,:,voi_zi:voi_ze]
    return vol

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

## load data
dataset = 'spheroids_x1'
DATA_DIR = '/data/datasets/{}'.format(dataset)

## load the training and testing list
train_file = DATA_DIR+'/train_list.txt'
test_file = DATA_DIR+'/test_list.txt'
trn_fns = read_txt(train_file)
tst_fns = read_txt(test_file)
val_fns = tst_fns

## loading volumes
ph_vols = np.stack([np.load(os.path.join(DATA_DIR,'phase',vol_fn+'.npy')) for vol_fn in tst_fns])
fl1_vols = np.stack([np.load(os.path.join(DATA_DIR,'fl1',vol_fn+'.npy')) for vol_fn in tst_fns])            

## network architecture
model = unet_model_3d(input_shape = (1, None, None, None), n_base_filters=16, pool_size=(2, 2, 2), depth = 5)
model_name = '3D_phase'
model_folder = '/data/models_fl/{}'.format(model_name)
best_weight = model_folder+'/best_model.h5'
model.load_weights(best_weight)

## padding
ph_vol = ph_vols[0].transpose([1,2,0])
gt_vol = fl1_vols[0].transpose([1,2,0])
xl, yl, zl = ph_vol.shape
ix, iy, iz = 0, 0, 0
ex, ey, ez = ix+256, iy+256, iz+32
all_voi_list = []
nb_voi = 0
iz, ez = 0, 0
while(iz<zl):
		ez = iz+32
		if ez > zl:
				iz, ez = zl-32, zl
		ix, ex = 0, 0
		while(ix<xl):
				ex = ix+256
				if ex > xl:
						ix, ex = xl-256, xl
				iy, ey = 0, 0
				while(iy<yl):
						ey = iy+256
						if ey > yl:
								iy, ey = yl-256, yl
						ph_voi = ph_vol[ix:ex,iy:ey,iz:ez]
						ph_voi_input = np.expand_dims(np.expand_dims(ph_voi,axis = 0),axis = 0)
						pr_voi = model.predict(ph_voi_input).squeeze()
						print(pr_voi.shape)
						all_voi_list.append(pr_voi)
						nb_voi = nb_voi+1
						print(nb_voi)
						iy = ey
				ix = ex
		iz = ez
		
## combine the prediction
# combine x and y
xy_vol_list = []
xl, yl, zl = 1744, 1744, 32
for i in range(4):
    i_start, i_end = i*49, (i+1)*49
    sub_vol_list = all_voi_list[i_start:i_end]
    xy_vol = combine_xy(sub_vol_list, xl, yl, zl)
    xy_vol_list.append(xy_vol)

# combine z
zzl = ph_vol.shape[-1]
pr_vol = combine_z(xy_vol_list, xl, yl, zzl)

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
	print('value in map: max {}, min {}'.format(vol1.max(),vol1.min()))
	print('value in map: max {}, min {}'.format(vol2.max(),vol2.min()))
	mse = np.mean((vol1-vol2)**2)
	psnr_score = 20*np.log10(255/np.sqrt(mse))
	return psnr_score

psnr_score = calculate_psnr(gt_vol, pr_vol)
cor_score = calculate_pearsonr(gt_vol, pr_vol)

def plot_slices(file_name, ph_vol_n, fl1_vol_n, fl2_vol_n, nb_images=10,  colorbar = True):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    rows, cols, size = nb_images,3,5
    font_size = 28; label_size = 20
    
    fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(nrows=rows,ncols=cols)
    for i in range(nb_images):
        slice_index = nb_images*i
        ph_img = ph_vol_n[slice_index:slice_index+3,:].transpose([1,2,0])
        fl1_img = fl1_vol_n[slice_index:slice_index+3,:].transpose([1,2,0])
        fl2_img = fl2_vol_n[slice_index:slice_index+3,:].transpose([1,2,0])
        cx0 = ax[i,0].imshow(ph_img); cx1 = ax[i,1].imshow(fl1_img); 
        cx2 = ax[i,2].imshow(fl2_img)
        ax[i,0].set_xticks([]);ax[i,0].set_yticks([])
        ax[i,1].set_xticks([]);ax[i,1].set_yticks([])
        ax[i,2].set_xticks([]);ax[i,2].set_yticks([])
        if colorbar:
            cbar = fig.colorbar(cx0, ax = ax[i,0], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
            cbar = fig.colorbar(cx1, ax = ax[i,1], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
            cbar = fig.colorbar(cx2, ax = ax[i,2], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
        if i == 0:
            ax[i,0].set_title('Image',fontsize=font_size); ax[i,1].set_title('FL1',fontsize=font_size); 
            ax[i,2].set_title('FL2',fontsize=font_size)
    fig.tight_layout(pad=-2)
    canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=80)

plot_slices(model_folder+'/test_prediction.png', ph_vol.transpose([2,0,1]), gt_vol.transpose([2,0,1]), pr_vol.transpose([2,0,1]), nb_images=10,  colorbar = True)