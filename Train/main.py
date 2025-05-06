import os
import argparse
import time
from torch_geometric.data import Data
import warnings
warnings.filterwarnings('ignore')

import hyperopt
from hyperopt import hp
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK

import numpy as np
import csv

from src.data_utils import load_tiles_data, TileData
from src.graph_utils import build_graph, transforms_graph, transforms_graph_pos
from src.post_processing import apply_post_processing
from src.models import get_model_cell_emb, get_model_graph, get_resnet_emb
from src.utils import get_device, save_predictions, save_overlays

from src.HyperOpt_optimization import opt

def main(params: dict) -> None:
	
	#Use GPU if available
	device = get_device()
	
	#Define models
	emb_model, resnet_model = get_model_cell_emb(device, params["resnet_path"])

	train_graphs_all = []
	test_graphs_all = []

	#Iterate over folds 
	for i in range(params["nb_folds"]):
		
		train_Tiles = load_tiles_data(i, params["emb_path"] + str('/Partition'), params["ep_centers_path"] + str('/Partition'), params["ep_gt_path"] + str('/Partition'))
		test_Tiles = load_tiles_data(i, params["emb_path"] + str('/Test'), params["ep_centers_path"] + str('/Test'), params["ep_gt_path"] + str('/Test'))
		
		train_graphs = []
		for t, tile in enumerate(train_Tiles):			
			# Build Graph
			graph_data = build_graph(ep_centers = tile.ep_centers, node_emb = tile.emb_cells_lab, node_gt = tile.ep_gt, node_emb_type = params["node_emb_type"], edge_threshold = params["edge_threshold"])
			graph = Data(x=graph_data.node_emb, edge_index=graph_data.edges, pos = graph_data.node_pos, y = graph_data.node_gt, num_classes = 2)
			if params["node_emb_type"] != 515:
				graph_trans = transforms_graph(graph)
			else:
				graph_trans = transforms_graph_pos(graph)
			train_graphs.append(graph_trans)	
		train_graphs_all.append(train_graphs)
						

		test_graphs = []
		for t, tile in enumerate(test_Tiles):
			# Build Graph
			graph_data = build_graph(ep_centers = tile.ep_centers, node_emb = tile.emb_cells_lab, node_gt = tile.ep_gt, node_emb_type = params["node_emb_type"], edge_threshold = params["edge_threshold"])
			graph = Data(x=graph_data.node_emb, edge_index=graph_data.edges, pos = graph_data.node_pos, y = graph_data.node_gt, num_classes = 2)
			if params["node_emb_type"] != 515:
				graph_trans = transforms_graph(graph)
			else:
				graph_trans = transforms_graph_pos(graph)
			test_graphs.append(graph_trans)
		test_graphs_all.append(train_graphs)
		
	
	#------------------------------------------------------------------------------------------------------------------------------------------
	# Hyperopt Space search
	#------------------------------------------------------------------------------------------------------------------------------------------

	search_space = {
	"n_convs": hyperopt.pyll.scope.int(hp.choice("n_convs",[2, 3, 4, 5, 6, 8, 12, 16])),		# pyll.scope.int() converts the float output of hp.quiniform into an integer
	"hidden_size": hyperopt.pyll.scope.int(hp.choice("hidden_size", [64, 128, 256, 512, 1024])),	
	"weight_decay": hp.loguniform("weight_decay", np.log(0.00001), np.log(0.1)),			# hp.loguniform(label, low, high) returns a value drawn according to exp(uniform(low, high)) so that the logarithm of the return value is uniformly distributed
	"lr": hp.loguniform("lr", np.log(0.00001), np.log(0.001)),					 
	"step_size": hyperopt.pyll.scope.int(hp.quniform("step_size", 10, 51, 10)),			 
	"lr_decay": hp.uniform("lr_decay", 0.01, 1),							# hp.uniform(label, low, high) return a value uniformly between low and high
	}

	max_epochs = 150
	opt_run = True

	# Write column names
	with open(params["path_csv"] + '/' + params["model"] + '.csv', 'a') as file:
		writer = csv.writer(file) #this is the writer object
		writer.writerow(['Fold', 'No. conv layers', 'Hidden size', 'Learning rate', 'Learning rate decay', 'Weight decay', 'Step size', 'Avg. F1', 'Loss'])

	trials = opt(search_space, num_epochs=max_epochs, m=params["model"], iterations= params["num_combinations"], device=device, opt_run=opt_run, model_name=params["model"], input_size = params["node_emb_type"], nb_folds = params["nb_folds"], path_csv = params["path_csv"], trial_save_path = params["trial_save_path"], train_graphs_all = train_graphs_all, val_graphs_all = test_graphs_all)
	print(trials.best_trial)
		
		
if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	
	parser.add_argument(
			"--nb_folds",
			type=int,
			default=None,
			help="number of folds for model optimization",
			required=True,
		)

	parser.add_argument(
			"--tiles_path",
			type=str,
			default=None,
			help="path to tiles",
			required=True,
		)

	parser.add_argument(
			"--img_mod",
			type=str,
			default=None,
			help="Image modality of the tiles to be processed, for example png, tif, ...",
			required=True,
		)

	parser.add_argument(
			"--ep_centers_path",
			type=str,
			default=None,
			help="Path to epithelial centers coordinates mat file",
			required=True,
		) 

	parser.add_argument(
			"--emb_path",
			type=str,
			default=None,
			help="Path to saved node embeddings",
			required=True,
		) 

	parser.add_argument(
			"--ep_gt_path",
			type=str,
			default=None,
			help="Path to epithelial class ground truth",
			required=True,
		)


	parser.add_argument(
			"--resnet_path",
			type=str,
			default=None,
			help="Path to resnet checkpoint",
			required=True,
		)


	parser.add_argument(
			"--edge_threshold",
			type=int,
			default=None,
			help="Delaunay edge threshold",
			required=True,
		)

	parser.add_argument(
			"--node_emb_type",
			type=int,
			default=None,
			help="Node embedding type: 512 for resnet, 513 for resnet+cl, 515 for resnet+cl+pos",
			required=True,
		)

	parser.add_argument(
			"--model",
			type=str,
			default=None,
			help="Defines the type of graph conv layer: GCN, GraphSage, GATv2Net or GIN",
			required=True,
		)

	parser.add_argument(
			"--num_combinations",
			type=int,
			default=None,
			help="Defines the number of hyperopt parameters combinations",
			required=True,
		)

	parser.add_argument(
			"--path_csv",
			type=str,
			default=None,
			help="Defines the path to store the csv result of the hyperopt search",
			required=True,
		)

	parser.add_argument(
			"--trial_save_path",
			type=str,
			default=None,
			help="Path to save Trial object in case the hyperopt optimization has to be resumed in a later stage",
			required=True,
		)

	
	params = vars(parser.parse_args())
	main(params)