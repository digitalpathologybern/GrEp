import os
import argparse
import time
from torch_geometric.data import Data
import warnings
warnings.filterwarnings('ignore')


from src.data_utils import load_tiles_data, TileData
from src.graph_utils import build_graph, node_classification
from src.post_processing import apply_post_processing
from src.models import get_model_cell_emb, get_model_graph, get_resnet_emb
from src.utils import get_device, save_predictions, save_overlays


def main(params: dict) -> None:
	
	#Use GPU if available
	device = get_device()
	
	#Define models
	emb_model, resnet_model = get_model_cell_emb(device, params["resnet_path"])
	gcn_model = get_model_graph(device, params["gcn_path"])
	
	# check if save path already exist
	if os.path.isdir(params["save_path"]) == False:
		os.mkdir(params["save_path"])
	os.mkdir(params["save_path"] + 'nuclei_mat/')
	if (params["save_overlays"]) == "True":
		os.mkdir(params["save_path"] + 'overlay/')
	
	#start_time = time.perf_counter()

	#Iterate over tiles 
	for i, tile_name in enumerate(os.listdir(params["tiles_path"])):
		
		tile = load_tiles_data(params["img_mod"], params["tiles_path"], tile_name, params["ep_centers_path"])
		print("Processing tile no", i, ": ", tile.image_name)
		embs, resnet_class = get_resnet_emb(resnet_model, emb_model, tile.patches_trans, device)
		
		# Check that there are at least 3 epithelial cells on the tile to build the graph, otherwise cannot build delaunay trianglation
		if len(tile.ep_centers) >= 3:
			graph_data = build_graph(image_name = tile.image_name, ep_centers = tile.ep_centers, node_class = resnet_class, node_emb = embs, edge_threshold = 250)
			graph = Data(x=graph_data.node_emb, edge_index=graph_data.edges, pos = graph_data.node_pos, num_classes = 2)
			if len(graph.edge_index) < 1:
				print("No graph structure (epithelial cells too far apart), the ResNet classification is the new node class")
				final_node_classification = resnet_class
			else:
				graph_data.node_class = node_classification(graph, gcn_model, device)
				final_node_classification = graph_data.node_class
				
				#Post-processing
				final_node_classification = apply_post_processing(graph_data, threshold = 40)
		else:
			print("Less than 3 epithelial cells on the Tile, Graph cannot be built and the ResNet classification is the new node class")
			final_node_classification = resnet_class
		
		save_predictions(tile.image_name, tile.ep_centers, final_node_classification, params["save_path"])
		if (params["save_overlays"]) == "True":
			save_overlays(tile.image_name, tile.image, tile.ep_centers, final_node_classification, params["save_path"])
	
	'''
	# Uncomment this section if you want to measure time
	end_time = time.perf_counter()
	elapsed_time = end_time - start_time
	print("Elapsed time: ", elapsed_time)
	'''	
		
if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	
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
			"--resnet_path",
			type=str,
			default=None,
			help="Path to resnet checkpoint",
			required=True,
		)

	parser.add_argument(
			"--gcn_path",
			type=str,
			default=None,
			help="Path to GCN checkpoint",
			required=True,
		)

	parser.add_argument(
			"--save_path",
			type=str,
			default=None,
			help="Path to save .mat file of classified epithelial cells",
			required=True,
		)

	parser.add_argument(
			"--save_overlays",
			type=bool,
			default="False",
			help="Save overlay predictions True/False",
			required=False,
		)
	
	params = vars(parser.parse_args())
	main(params)