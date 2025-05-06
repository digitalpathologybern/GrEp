import os
import torch
from scipy.io import savemat
import matplotlib.pyplot as plt

def get_device():
	if torch.cuda.is_available():
		device = torch.device("cuda:0")
		print('running on GPU')
	else:
		device = torch.device("cpu")
		print('running on cpu')
	return device


def save_predictions(image_name, ep_centers, predictions, save_path):
	img_dict = {'inst_type' : predictions, 'inst_centroid' : ep_centers}
	savemat(save_path + 'nuclei_mat/'+ image_name +  '.mat', img_dict, oned_as='row')


def save_overlays(image_name, tile, ep_centers, predictions, save_path):
	overlay = tile
	centers_coord = ep_centers
	cell_type = predictions
	plt.figure(figsize = (9,9))
	plt.imshow(overlay)
	for i in range(len(centers_coord)):
		x_1 = int(float(centers_coord[i][0]))
		x_2 = int(float(centers_coord[i][1]))

		if cell_type[i] == 0: # means normal
			color = 'green'
		elif cell_type[i] == 1: # means malignant
			color = 'red'

		plt.scatter(x_1, x_2, s=20, c=color, marker='o')
	plt.axis('off')
	plt.savefig(save_path + 'overlay/'+ image_name + '_GrEp_overlay.png', dpi = 300, bbox_inches='tight')