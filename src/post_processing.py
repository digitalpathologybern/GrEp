import os
import numpy as np
import networkx as nx
from scipy.spatial import Delaunay


def apply_post_processing(graph_data, threshold):
	edges_pp = compute_pp_edges(graph_data.ep_centers, threshold)
	nxGraph = get_nx_graph(graph_data.node_class, graph_data.ep_centers, edges_pp)
	sub = get_disconnected_components(nxGraph)
	final_node_classification = nodes_clustering(sub, graph_data.node_class)
	return final_node_classification


def get_triangulation(ep_centers):
		tri = Delaunay(ep_centers)
		return tri

def compute_pp_edges(ep_centers, threshold):
	tri = get_triangulation(ep_centers)
	small_edges = []
	large_edges = []
	for tr in tri.simplices:
		for i in range(3):
			edge_idx0 = tr[i]
			edge_idx1 = tr[(i + 1)%3] # to always get the next pointidx in the triangle formed by delaunay

			# check the couple of points hasn't already been visited from the other side (= starting from the other point)
			# if yes then continue because already in the array
			if (edge_idx1, edge_idx0) in small_edges:
				continue
			if (edge_idx1, edge_idx0) in large_edges:
				continue

			p0 = tri.points[edge_idx0]
			p1 = tri.points[edge_idx1]

			if np.linalg.norm(p1-p0) < threshold:
				small_edges.append((edge_idx0, edge_idx1))
			else:
				large_edges.append((edge_idx0, edge_idx1))

	return small_edges


def get_nx_graph(node_class, ep_centers, edges_pp):
	G = nx.Graph()
	nb_edges = len(edges_pp)

	# add the nodes
	nb_nodes = len(ep_centers)
	name_nodes = list(range(0, nb_nodes))
	G.add_nodes_from(name_nodes)

	# add node features
	features = dict(enumerate(node_class))
	nx.set_node_attributes(G, features, 'label')

	# add edges
	for e in range(nb_edges):
		cell0 = edges_pp[e][0]
		cell1 = edges_pp[e][1]
		cell0_coord = ep_centers[cell0]
		cell1_coord = ep_centers[cell1]
		dist = np.linalg.norm(cell1-cell0)
		G.add_edge(cell0, cell1, length=dist)

	return G


def get_disconnected_components(nx_Graph):
	return [nx_Graph.subgraph(c).copy() for c in nx.connected_components(nx_Graph)]


def nodes_clustering(sub, cl):

	#print(cl, cl[0])
	y_preds_new = cl.copy()

	for s in range(len(sub)):
		nodes_idx = list(sub[s].nodes)
		#print('nodes idx:', nodes_idx)

		# find the labels of these nodes
		nodes_lab = []
		new_nodes_lab = []

		if len(nodes_idx) == 1:
			nodes_lab.append(cl[nodes_idx[0]])
		else:
			for i in nodes_idx:
				lab = cl[i]
				nodes_lab.append(lab)
		#print(len(nodes_lab))

		min_lab = np.min(nodes_lab)
		max_lab = np.max(nodes_lab)
		mean_lab = np.mean(nodes_lab)
		if min_lab == max_lab: #means all the nodes have the same labels
			new_nodes_lab = nodes_lab
		else:
			new_nodes_lab = nodes_lab
			if mean_lab > 0.5: # means most are malignant -> put all cells malignant 
				for j in range(len(new_nodes_lab)):
					if new_nodes_lab[j] == 0:
						new_nodes_lab[j] = 1
			else:
				for j in range(len(new_nodes_lab)):
					if new_nodes_lab[j] == 1:
						new_nodes_lab[j] = 0

		kk = 0
		#print('nb nodes idx:', len(nodes_idx), ' nb new nodes lab:', len(new_nodes_lab))
		#print(len(y_preds_new))
		for k in nodes_idx:
			y_preds_new[k] = new_nodes_lab[kk]
			kk += 1

	return y_preds_new