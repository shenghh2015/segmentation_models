import numpy as np

def generate_folder(folder):
	import os
	if not os.path.exists(folder):
		os.system('mkdir -p {}'.format(folder))

# plot training and validation loss
def plot_history(file_name, history):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	rows, cols, size = 1,3,5
	fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(rows,cols)
	ax[0].plot(history.history['loss']);ax[0].plot(history.history['val_loss'])
	ax[0].set_ylabel('loss');ax[0].set_xlabel('epochs');ax[0].legend(['train','valid'])
	ax[1].plot(history.history['iou_score']);ax[1].plot(history.history['val_iou_score'])
	ax[1].set_ylabel('iou_score');ax[1].set_xlabel('epochs');ax[1].legend(['train','valid'])
	ax[2].plot(history.history['f1-score']);ax[2].plot(history.history['val_f1-score'])
	ax[2].set_ylabel('f1-score');ax[2].set_xlabel('epochs');ax[2].legend(['train','valid'])
	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=100)

# plot for phase -> fluo validation loss
def plot_history_flu(file_name, history):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	rows, cols, size = 1,2,5
	fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(rows,cols)
	ax[0].plot(history.history['loss']);ax[0].plot(history.history['val_loss'])
	ax[0].set_ylabel('MSE');ax[0].set_xlabel('epochs');ax[0].legend(['train','valid'])
	ax[1].plot(history.history['psnr']);ax[1].plot(history.history['val_psnr'])
	ax[1].set_ylabel('PSNR');ax[1].set_xlabel('epochs');ax[1].legend(['train','valid'])
	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=100)

# calculate the IoU and dice scores
def iou_calculate(y_true, y_pred):
	# one hot encoding of predictions
	num_classes = y_pred.shape[-1]
	y_pred = np.array([np.argmax(y_pred, axis=-1)==i for i in range(num_classes)]).transpose(1,2,3,0)
	print(y_pred.shape)
	axes = (1,2) # W,H axes of each image
	intersection = np.sum(np.logical_and(y_pred, y_true), axis=axes)
	union = np.sum(np.logical_or(y_pred, y_true), axis=axes)
	mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)

	smooth = .00001
	iou_per_image_class = (intersection + smooth) / (union + smooth)
	dice_per_image_class = (2 * intersection + smooth)/(mask_sum + smooth)

	mean_iou_over_images = np.mean(iou_per_image_class, axis = 0)
	mean_iou_over_images_class = np.mean(mean_iou_over_images)
	dice_class = np.mean(dice_per_image_class, axis = 0)
	mean_dice = np.mean(dice_per_image_class)

	return mean_iou_over_images, mean_iou_over_images_class, dice_class, mean_dice

def precision(label, confusion_matrix):
	col = confusion_matrix[:, label]
	return confusion_matrix[label, label] / col.sum()
    
def recall(label, confusion_matrix):
	row = confusion_matrix[label, :]
	return confusion_matrix[label, label] / row.sum()

def f1_score(label, confusion_matrix):
	row = confusion_matrix[label, :]; col = confusion_matrix[:, label]
	prec_score = confusion_matrix[label, label] / col.sum()
	recall_score = confusion_matrix[label, label] / row.sum()
	return 2*prec_score*recall_score/(prec_score + recall_score)

def psnr_score(img1, img2):
	import math
	import numpy as np
	img1 = img1.astype(np.float64)
	img2 = img2.astype(np.float64)
	mse = np.mean((img1 - img2)**2)
	if mse == 0:
		return float('inf')
	return 20 * math.log10(1.0 / math.sqrt(mse))

def calculate_psnr(imgs1, imgs2):
	if len(imgs1.shape) == 4:
		axis_tuple = (1,2,3)
	else:
		axis_tuple = (1,2)
	mse = np.mean((imgs1-imgs2)**2, axis = axis_tuple)
	psnr_scores = 20*np.log10(1.0/np.sqrt(mse))
	return np.mean(psnr_scores), psnr_scores

def calculate_pearsonr(imgs1, imgs2):
	from scipy import stats
	scores = []
	for i in range(imgs1.shape[0]):
		flat1 = imgs1[i,:].flatten(); flat2 = imgs2[i,:].flatten()
		score, _= stats.pearsonr(flat1, flat2)
		scores.append(score)
	return np.mean(scores), scores


