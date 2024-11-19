import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear

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