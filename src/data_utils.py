import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Optional, List
import cv2
from PIL import Image
import scipy.io


class TileData(Dataset):
	
	def __init__(self, 
				 image: np.ndarray, 
				 image_name: str,
				 ep_centers: list, 
				 patch_size: Optional[int] = None, 
				 patches: Optional[List[np.ndarray]] = None,
				 patches_trans: Optional[List[np.ndarray]] = None):
				
		self.image = image
		self.image_name = image_name
		self.ep_centers = ep_centers
		
		if patch_size is None:
			self.patch_size = 128
		else:
			self.patch_size = patch_size
		 
		if patches is None:
			self.patches = []
		else:
			self.patches = patches
			
		if patches_trans is None:
			self.patches_trans = []
		else:
			self.patches_trans = patches_trans
			
		
		#TileData.Dataset.append(self) # Store all TileData entities
		
		self.patches = self.get_cell_patches(self)
		self.patches_trans = self.get_transformed_cell_patches(self)

	
	@staticmethod
	def get_cell_patches(self):
		'''
		Crop nuclei tile around nuclei centroids
		'''
		patches = []
		for i, center in enumerate(self.ep_centers):
			patches.append(get_patch(self.image, self.patch_size, center))
		return patches
	
	
	
	@staticmethod
	def get_transformed_cell_patches(self):
		'''
		Apply the transforms to the cell patches before applying the model
		Parameters
		----------
		cell_patches: list of numpy arrays of the cell patches
		Returns
		-------
		patches: list of numpy arrays of size (size, size, 3) - List of the cell patches cropped from the input tile
		'''
		trans_patches = []
		for i, patch in enumerate(self.patches):
			t_patch_gamma = transforms_gamma(patch)
			trans_patches.append(t_patch_gamma)
		return trans_patches



def load_tiles_data(img_mod, tile_path, tile_name, ep_centers_path, cell_type = True) -> TileData:
	'''
	Form the given path, for each loaded tile a TileData is created
	'''
	init_= cv2.imread(tile_path + tile_name)
	image = cv2.cvtColor(init_, cv2.COLOR_BGR2RGB)
	
	image_name = tile_name.split('.' + img_mod)[-2]
	lab_dict = scipy.io.loadmat(ep_centers_path + image_name + '.mat')
	ep_centers = lab_dict['inst_centroid'].tolist()
	
	tile = TileData(image,
				   image_name,
				   ep_centers)
	
	return tile



def get_patch(init_image, crop_size, cell_center):
	
	m = crop_size/2 # margin
	
	img = init_image.copy()
		
	borne_sup_x = init_image.shape[0] - m
	borne_inf_x = init_image.shape[0] - crop_size
	borne_sup_y = init_image.shape[1] - m
	borne_inf_y = init_image.shape[1] - crop_size
	max_x = init_image.shape[0]
	max_y = init_image.shape[1]
			
	center_1 = int(cell_center[0])
	center_2 = int(cell_center[1])
			
	# define the "box" to cut around the cell
				
	if center_1 < m or center_1 > (max_y-m) or center_2 < m or center_2 > (max_x-m):
					
		if center_1 < m and center_2 < m: # means the nucleus is in the upper left corner -> need to add padding on left and upper side of the crop
			top_left = [0, 0]
			bottom_right = [center_1 + m , center_2 + m]
			top = m-center_2
			left = m-center_1
			right =  0
			bottom = 0
		elif m <= center_1 <= (max_y-m) and center_2 < m: #means the nucleus is in the upper border
			top_left = [center_1 - m, 0]
			bottom_right = [center_1 + m, center_2 + m]
			left = 0
			top = m-center_2
			bottom = 0
			right = 0
		elif center_1 > (max_y-m) and center_2 < m: # means the nucleus is in the upper right corner
			top_left = [center_1 - m, 0]
			bottom_right = [max_y + m, center_2 + m]
			top = m-center_2
			right = m-(max_y-center_1)
			left = 0
			bottom = 0
		elif center_1 < m and m <= center_2 <=(max_x-m): #means the nucleus is in the left border
			top_left = [0, center_2 - m]
			bottom_right = [center_1 + m, center_2 + m]
			top = 0
			left = m-center_1
			right = 0
			bottom = 0
		elif center_1 > (max_y-m) and m <= center_2 <= (max_x-m): #means the nucleus is in the right border 
			top_left = [center_1 - m, center_2 - m]
			bottom_right = [max_y, center_2 + m]
			left = 0
			top = 0
			bottom = 0
			right = m-(max_y-center_1)
		elif center_1 < m and center_2 > (max_x-m): #means the nucleus is in the bottom left corner
			top_left = [0, center_2 - m]
			bottom_right = [center_1 + m, max_x]
			top = 0
			right = 0
			left = m-center_1
			bottom = m-(max_x-center_2)
		elif m <= center_1 <= (max_y-m) and center_2 > (max_x-m): #means the nucleus is in the bottom border 
			top_left = [center_1 - m, center_2 - m]
			bottom_right = [center_1 + m, max_x]
			top = 0
			left = 0
			right = 0
			bottom = m-(max_x-center_2)
		elif center_1 > (max_y-m) and center_2 > (max_x-m): #means the nucleus is in the bottom right corner
			top_left = [center_1 - m, center_2 - m]
			bottom_right = [max_y , max_x]
			left = 0
			top = 0
			bottom = m-(max_x-center_2)
			right = m-(max_y-center_1)

		ptch = img[int(top_left[1]):int(bottom_right[1]), int(top_left[0]):int(bottom_right[0])]
		patch = cv2.copyMakeBorder(ptch, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, None, value = 0)
		if(patch.shape != (crop_size, crop_size, 3)):
			print('Size not correct', ptch.shape, patch.shape) 
					
		rgb_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
						
					
	else:
					
		top_left = [center_1 - m, center_2 - m]
		bottom_right = [center_1 + m, center_2 + m]
		
		patch = img[int(top_left[1]):int(bottom_right[1]), int(top_left[0]):int(bottom_right[0])]
		if(patch.shape != (crop_size, crop_size, 3)):
			print('1. Size not correct', patch.shape)			 
			
		#rgb_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
													

		
	return patch


class MyGammaSquare:
	def __init__(self, gamma1, gamma2, thr, radius):
		self.gamma1 = gamma1
		self.gamma2 = gamma2
		self.thr = thr
		self.radius = radius

	def __call__(self, x):
		PIL_img = Image.fromarray(np.uint8(x)).convert('RGB')
		light_img = transforms.functional.adjust_gamma(PIL_img, self.gamma1, self.thr)
		enhanced_img = transforms.functional.adjust_gamma(PIL_img, self.gamma2, self.thr)
		mask = np.zeros((128, 128, 3), dtype = np.uint8)
		mask = cv2.rectangle(mask, pt1=(32,32), pt2=(96,96), color=(255,255,255), thickness = -1)
		out = np.where(mask == (255, 255, 255), enhanced_img, light_img)
		return out


transforms_gamma =  transforms.Compose([
		transforms.ToPILImage(),
		transforms.RandomApply([MyGammaSquare(0.5, 1.5, 1, 30)], p = 1.0),
		transforms.ToTensor(),
])

