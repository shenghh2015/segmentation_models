import numpy as np

def plot_flu_prediction(file_name, images, gt_maps, pr_maps, nb_images, rand_seed = 3, colorbar = True):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	import random
	seed = rand_seed #3
	random.seed(seed)
	font_size = 28; label_size = 20
	indices = random.sample(range(gt_maps.shape[0]),nb_images)
	rows, cols, size = nb_images,4,5
	widths = [0.8, 1, 1, 1]; heights = [1, 1, 1, 1]; gs_kw = dict(width_ratios=widths, height_ratios=heights)
	
	fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(nrows=rows,ncols=cols, gridspec_kw=gs_kw)
	for i in range(len(indices)):
		idx = indices[i]
		image, gt_map, pr_map = images[idx,:].squeeze(), gt_maps[idx,:].squeeze(), pr_maps[idx,:].squeeze()
		image = np.uint8((image-image.min())/(image.max()-image.min())*255)
		err_map = np.abs(gt_map-pr_map)*255; err_map_rgb = np.zeros(image.shape).astype(np.uint8); err_map_rgb[:,:,:-1] = err_map
		gt_map_rgb = np.zeros(image.shape,dtype=np.uint8); # gt_map_rgb[:,:,:-1]=np.uint8((gt_map-gt_map.min())/(gt_map.max()-gt_map.min())*255)
		gt_map_rgb[:,:,:-1]=np.uint8(gt_map*255)
		pr_map_rgb = np.zeros(image.shape,dtype=np.uint8); # pr_map_rgb[:,:,:-1]=np.uint8((pr_map-pr_map.min())/(pr_map.max()-pr_map.min())*255)
		pr_map_rgb[:,:,:-1]=np.uint8(pr_map*255)
		cx0 = ax[i,0].imshow(image); cx1 = ax[i,1].imshow(gt_map_rgb); 
		cx2 = ax[i,2].imshow(pr_map_rgb); cx3 = ax[i,3].imshow(err_map_rgb)
# 		cx0 = ax[i,0].imshow(image[::4,::4,:]); cx1 = ax[i,1].imshow(gt_map_rgb[::4,::4,:]); 
# 		cx2 = ax[i,2].imshow(pr_map_rgb[::4,::4,:]); cx3 = ax[i,3].imshow(err_map_rgb[::4,::4])
		ax[i,0].set_xticks([]);ax[i,0].set_yticks([]);
		#ax[i,0].tick_params(axis = 'x', labelsize = label_size);ax[i,0].tick_params(axis = 'y', labelsize = label_size);
		ax[i,1].set_xticks([]);ax[i,2].set_xticks([]);ax[i,3].set_xticks([])
		ax[i,1].set_yticks([]);ax[i,2].set_yticks([]);ax[i,3].set_yticks([])
		if colorbar:
			#fig.colorbar(cx0, ax = ax[i,0], shrink = 0); 
			cbar = fig.colorbar(cx1, ax = ax[i,1], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
			cbar = fig.colorbar(cx2, ax = ax[i,2], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
			cbar = fig.colorbar(cx3, ax = ax[i,3], shrink = 0.68); cbar.ax.tick_params(labelsize=label_size) 
		if i == 0:
			ax[i,0].set_title('Image',fontsize=font_size); ax[i,1].set_title('Ground Truth',fontsize=font_size); 
			ax[i,2].set_title('Prediction',fontsize=font_size); ax[i,3].set_title('Err Map',fontsize=font_size);
	fig.tight_layout(pad=-2)
	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=80)