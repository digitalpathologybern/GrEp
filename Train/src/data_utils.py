import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Optional, List
import cv2
from PIL import Image
import scipy.io
import pandas as pd


class TileData(Dataset):
	
	def __init__(self,  
				 emb_cells_lab: list,
				 ep_centers: list, 
				 ep_gt: list):
				
		self.emb_cells_lab = emb_cells_lab
		self.ep_centers = ep_centers
		self.ep_gt = ep_gt
		

def remove_nan(array_tot):

	array_good = []
	for im in range(len(array_tot)):
		x = array_tot[im]
		cleanedList = [i for i in x if str(i) != 'nan']
		array_good.append(cleanedList)

	return array_good


def get_coords(df):

	coords = []
	for im in range(len(df)):
		# to get the coordinates of image im
		test_list = df.values[im].copy()
		res = [i for i in test_list if i is not None] # otherwise trashes the 0 also ! need the None condition
		coords.append(res)

	return coords


def get_coords_embs(df):

	coords = []
	for im in range(len(df)):
		# to get the coordinates of image im
		test_list = df.values[im].copy()
		res = [i[0] for i in test_list if i is not None] # otherwise trashes the 0 also ! need the None condition
		coords.append(res)

	return coords



def load_tiles_data(fold, emb_path, ep_centers_path, ep_gt_path, cell_type = True) -> TileData:
	'''
	From the given path, for each loaded tile a TileData is created
	'''
	
	emb_cells_labels_df = pd.read_pickle(emb_path + str(fold+1) + '/features_128/128px_resnet_gamma_cl_preds_512_all.pkl')
	emb_cells_labels = get_coords(emb_cells_labels_df)
	emb_cells_lab = remove_nan(emb_cells_labels)

	gt_cells_labels_df = pd.read_pickle(ep_gt_path + str(fold+1) + '/gt_labels/gt_all.pkl')
	gt_cells_labels = get_coords(gt_cells_labels_df)
	gt_cells_lab = remove_nan(gt_cells_labels)

	ep_centers = []
	for filepath in sorted(os.listdir(ep_centers_path + str(fold+1) + '/gt_ep_lab/')):
		m = scipy.io.loadmat(ep_centers_path + str(fold+1) + '/gt_ep_lab/' + filepath)
		centers = m['inst_centroid']
		ep_centers.append(centers)
	
	Tiles = []
	for i in range(len(ep_centers)):
		
		tile = TileData(emb_cells_lab[i],
				   ep_centers[i],
				   gt_cells_lab[i])
		Tiles.append(tile)
	
	return Tiles




