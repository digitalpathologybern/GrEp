import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List
from torch.utils.data import Dataset
from torchvision import transforms
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from scipy.spatial import Delaunay


class GraphData(Dataset):
		
	def __init__(self,  
				 ep_centers: list, 
				 node_gt: list,
				 node_emb_type: int,
				 node_emb: list,
				 edge_threshold = int,
				 edges: Optional[List[tuple]] = None,
				 node_pos: Optional[torch.Tensor] = None):
				
		self.ep_centers = ep_centers
		self.node_gt = node_gt
		self.node_emb_type = node_emb_type
		self.node_emb = node_emb
		self.edge_threshold = edge_threshold
		
		if edges is None:
			self.edges = []
		else:
			self.edges = edges
						
		if node_pos is None:
			self.node_pos = []
		else:
			self.node_pos = node_pos
				
		self.edges = self.compute_graph_edges(self)
		self.edges = self.init_edges(self)
		
		self.node_emb = self.init_embedding(self)
		self.node_pos = self.init_pos(self)
	
	
	@staticmethod
	def get_triangulation(ep_centers):
		tri = Delaunay(ep_centers)
		return tri


	@staticmethod
	def compute_graph_edges(self):
	
		tri = self.get_triangulation(self.ep_centers)
		small_edges = []
		large_edges = []
		for tr in tri.simplices:
			for i in range(3):
				edge_idx0 = tr[i]
				edge_idx1 = tr[(i + 1)%3] # to always get the next pointidx in the triangle formed by delaunay

				# check the pair of points isn't already in the list, if yes contnue without saving to sprevent duplicates
				if (edge_idx1, edge_idx0) in small_edges:
					continue
				if (edge_idx1, edge_idx0) in large_edges:
					continue

				p0 = tri.points[edge_idx0]
				p1 = tri.points[edge_idx1]

				if np.linalg.norm(p1-p0) < self.edge_threshold:
					small_edges.append((edge_idx0, edge_idx1))
				else:
					large_edges.append((edge_idx0, edge_idx1))

		return small_edges

	
	@classmethod
	def init_edges(cls, self):
		edge_long = torch.tensor(self.edges, dtype=torch.long)
		return edge_long.t().contiguous()
	
	
	@classmethod
	def init_embedding(cls, self):
		return torch.tensor(self.node_emb, dtype=torch.float)
		
	@classmethod
	def init_pos(cls, self):
		return torch.tensor(self.ep_centers, dtype=torch.float)




def build_graph(ep_centers, node_emb, node_gt, node_emb_type, edge_threshold) -> GraphData:
	
	graph = GraphData(ep_centers,
					  node_gt,
					  node_emb_type,
					  node_emb,
					  edge_threshold)
	return graph


class CenterPos(BaseTransform):
	"""
	Centers x and y coordinate position around the origin.
	"""
	def __call__(self, data):
		data.pos = data.pos - data.pos.mean(dim=-2, keepdim=True)
		return data

class AddPosToNodeFeature(BaseTransform):
	"""
	Adds the position of the node as a node feature
	"""

	def __call__(self, data):
		data.x = torch.cat((data.pos, data.x), 1)
		data.num_node_features += 2
		return data


	
transforms_graph = transforms.Compose([
	CenterPos(),

])


transforms_graph_pos = transforms.Compose([
	CenterPos(),
	AddPosToNodeFeature(),

])



def node_classification(graph, graph_model, device):
	'''
	Apply Message Pasing to improve cell classification / node clasifcation i the graph
	Parameters
	----------
	graph: pytorch geometric graph object, tissue representatio of the tile
	graph_mode: graph convlutional neural network for node classification
	device: either CPU or GPU
	Returns
	-------
	y_preds: list int - node (cell) classification after message passing - 0: healthy epihelial cell and 1: malignant epithelial cell
	''' 
	if graph.node_emb_type == 515:
		gn = transforms_graph_pos(graph)
	else:
		gn = transforms_graph(graph)
	outputs = graph_model(gn.to(device))
	_, preds = torch.max(outputs, 1)
	y_preds = preds.cpu().numpy()
	return y_preds