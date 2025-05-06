import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GINConv, GATv2Conv, SAGEConv, global_mean_pool
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Dropout, LeakyReLU


# Models

class GCN(torch.nn.Module):

	def __init__(self, num_of_feat, num_layers, hidden):
		super(GCN, self).__init__()
		self.conv1 = GCNConv(num_of_feat, hidden)
		self.convs = torch.nn.ModuleList()
		for i in range(num_layers - 1):
			self.convs.append(GCNConv(hidden, hidden))
		self.fc1 = Linear(hidden, hidden)
		self.fc2 = Linear(hidden, hidden)
		self.fc3 = Linear(hidden, 2)


	def forward(self, data):
		x, edge_index, batch = data.x.float(), data.edge_index, data.batch

		x = F.leaky_relu(self.conv1(x, edge_index))
		x = F.dropout(x, p=0.5)
		for conv in self.convs:
			x = F.leaky_relu(conv(x, edge_index))
			x = F.dropout(x, p=0.5)
		x = F.leaky_relu(self.fc1(x))
		x = F.dropout(x, p=0.5, training=self.training)
		x = F.leaky_relu(self.fc2(x))
		x = F.dropout(x, p=0.5, training=self.training)
		x = self.fc3(x)

		return F.log_softmax(x, dim=1)


class GraphSage(torch.nn.Module):

	def __init__(self, num_of_feat, num_layers, hidden):
		super(GraphSage, self).__init__()
		self.conv1 = SAGEConv(num_of_feat, hidden)
		self.convs = torch.nn.ModuleList()
		for i in range(num_layers - 1):
			self.convs.append(SAGEConv(hidden, hidden))
		self.fc1 = Linear(hidden, hidden)
		self.fc2 = Linear(hidden, hidden)
		self.fc3 = Linear(hidden, 2)


	def forward(self, data):
		x, edge_index, batch = data.x.float(), data.edge_index, data.batch
		#print(x.shape(), edge_index.shape())

		x = F.leaky_relu(self.conv1(x, edge_index))
		x = F.dropout(x, p=0.5)
		for conv in self.convs:
			x = F.leaky_relu(conv(x, edge_index))
			x = F.dropout(x, p=0.5)
		x = F.leaky_relu(self.fc1(x))
		x = F.dropout(x, p=0.5, training=self.training)
		x = F.leaky_relu(self.fc2(x))
		x = F.dropout(x, p=0.5, training=self.training)
		x = self.fc3(x)
		
		return F.log_softmax(x, dim=1)



class GATv2Net(torch.nn.Module):

	def __init__(self, num_of_feat, num_layers, hidden):
		super(GATv2Net, self).__init__()
		self.conv1 = GATv2Conv(num_of_feat, hidden)
		self.convs = torch.nn.ModuleList()
		for i in range(num_layers - 1):
			self.convs.append(GATv2Conv(hidden, hidden))
		self.fc1 = Linear(hidden, hidden)
		self.fc2 = Linear(hidden, hidden)
		self.fc3 = Linear(hidden, 2)
		

	def forward(self, data):
		x, edge_index, batch = data.x, data.edge_index, data.batch
		
		x = F.leaky_relu(self.conv1(x, edge_index))
		x = F.dropout(x, p=0.5)
		for conv in self.convs:
			x = F.leaky_relu(conv(x, edge_index))
			x = F.dropout(x, p=0.5)
		x = F.leaky_relu(self.fc1(x))
		x = F.dropout(x, p=0.5, training=self.training)
		x = F.leaky_relu(self.fc2(x))
		x = F.dropout(x, p=0.5, training=self.training)
		x = self.fc3(x)

		return F.log_softmax(x, dim=1)



class GIN(torch.nn.Module):

	def __init__(self, num_of_feat, num_layers, hidden):
		super(GIN, self).__init__()
		self.conv1 = GINConv(Sequential(Linear(num_of_feat, hidden), LeakyReLU(), Linear(hidden, hidden)))
		self.convs = torch.nn.ModuleList()
		for i in range(num_layers - 1):
			self.convs.append(GINConv(Sequential(Linear(hidden, hidden), LeakyReLU(), Linear(hidden, hidden))))
		self.fc1 = Linear(hidden, hidden)
		self.fc2 = Linear(hidden, hidden)
		self.fc3 = Linear(hidden, 2)


	def forward(self, data):
		x, edge_index, batch = data.x.float(), data.edge_index, data.batch

		x = F.leaky_relu(self.conv1(x, edge_index))
		x = F.dropout(x, p=0.5)
		for conv in self.convs:
			x = F.leaky_relu(conv(x, edge_index))
			x = F.dropout(x, p=0.5)
		x = F.leaky_relu(self.fc1(x))
		x = F.dropout(x, p=0.5, training=self.training)
		x = F.leaky_relu(self.fc2(x))
		x = F.dropout(x, p=0.5, training=self.training)
		x = self.fc3(x)

		return F.log_softmax(x, dim=1)
