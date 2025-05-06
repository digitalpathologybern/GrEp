import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models

from src.GCN import GCN


def load_checkpoint(model, path, model_name):
	"""
	load model checpoint
	"""
	checkpoint = torch.load(path)
	try:
		model.load_state_dict(checkpoint['model_state_dict'])
		print("succesfully loaded " + model_name + " weights")
	except:
		print("Not able to load " + model_name + " weights")
	return model.eval()


def get_model_cell_emb(device, path):
	'''
	defines the resnet18 model that was trained to differentiate between healthy(label = 0) and malignant (label = 1) epithelial cells
	'''
	model = models.resnet18(pretrained = True)
	model.avgpool = nn.AdaptiveAvgPool2d(1)
	model.fc = nn.Linear(in_features=512, out_features=2)
	
	model = load_checkpoint(model, path, "ResNet")
	model.to(device)
	model.eval()
	
	model_emb = nn.Sequential(*list(model.children())[:-1])
	model_emb.to(device)
	model_emb.eval()
	
	return model_emb, model


def get_model_graph(device, path):
	'''
	defines the GCN model that was trained to refine node embeddings for normal versus malignant classification
	'''
	model_graph = GCN(num_of_feat=512, num_layers = 20, hidden=512).to(device)
	model_graph = load_checkpoint(model_graph, path, "GCN")
	model_graph.to(device)
	model_graph.eval()
	
	return model_graph


def get_batch(trans_patches):
	for i in range(len(trans_patches)):
		cell_g = trans_patches[i][None, :, :, :]
		if i == 0:
			batch = cell_g
		else:
			batch = torch.cat((batch, cell_g), 0)
	return batch


def get_resnet_emb(model, model_emb, ep_patches, device):
	'''
	Apply healthy vs malignant epithelial cell classification on the cell patches
	Parameters
	----------
	model: ResNet trained model
	model_emb: ResNet trained to extract node embedding
	ep_patches: List of numpy array (images) of the epthelial cell patches on which the model will be applied
	device: either CPU or GPU
	Returns
	-------
	out_tot: list of list of doubles - cell embeddings from resnet hidden layer
	cl_tot: list of int, result of the cell differentiation - 0: healthy epihelial cell and 1: malignant epithelial cell 
	'''   
	
	b_size = 32
	n_batches = int(np.ceil(len(ep_patches)/b_size))
	out_tot = []
	cl_tot = []

	for b in range(n_batches):
		if b == n_batches -1:
			batch = get_batch(ep_patches[b*b_size:])
		else:
			batch = get_batch(ep_patches[b*b_size:(b+1)*b_size])
		
		pred = model(batch.to(device))
		_, preds = torch.max(pred, 1)
		cl = preds.cpu().detach().numpy()
		cl_tot.extend(cl)
		
		out = model_emb(batch.to(device))
		o_tot = []
		for i, out_ in enumerate(out):
			out_g = out_.cpu().detach().numpy()
			o = []
			for k, out_g_ in enumerate(out_g):
				o.append(out_g_[0][0])
			o_tot.append(o)
		out_tot.extend(o_tot)
				  
	return out_tot, cl_tot