import numpy as np
np.random.seed(0)

from numpy.random import seed
seed(1)

import hyperopt
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK

import torch
torch.manual_seed(1)
import torch.optim as optim
from torch.nn import Linear
import os, glob

import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

import torch_geometric
from torch_geometric import transforms
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from torch_geometric.data import DataLoader
import torch_geometric.data
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.nn import GCNConv, GraphConv, GINConv, GATv2Conv
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Dropout, LeakyReLU

import torch_geometric.transforms as T

from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from tqdm import tqdm
import scipy.io

torch.set_printoptions(linewidth=120) # display options for output
torch.set_grad_enabled(True)

from torch_geometric.utils.convert import from_networkx
import random

import warnings
warnings.filterwarnings("ignore") 
import sklearn
from sklearn.model_selection import KFold
import os, argparse

import csv

import pickle

from src.GNN_models import GCN, GraphSage, GATv2Net, GIN


def train(loader, model, criterion, optimizer, device):
	
	y_pred = []
	y_true = []
	
	model.train()
	
	running_loss = 0.0
	running_corrects = 0
	i = 0
	
	for data in tqdm(loader):
		
		data = data.to(device)
		optimizer.zero_grad()
		
		output = model(data)
		_, preds = torch.max(output, 1)
		
		#print(len(output), len(data.y))
		loss = criterion(output, torch.tensor(data.y[0], dtype=torch.long).to(device))
		loss.backward()
		optimizer.step()
		
		running_loss += loss.item()
		running_corrects += torch.sum(preds == torch.tensor(data.y[0], dtype=torch.long).to(device))
		
		y_pred.extend(preds.detach().cpu().numpy())
		y_true.extend(data.y[0])
		
		i+= 1
		  
	epoch_loss = running_loss / i
	epoch_acc = running_corrects.double() / i
		
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	#print(y_pred[:20], y_true[:20])
	acc = (y_pred == y_true).mean()
	f1 = f1_score(y_true, y_pred, average = 'weighted')
	
	print('TRAIN: Epoch Loss: {}, Epoch Acc: {}, weighted F1: {} '.format(epoch_loss, acc, f1))
	
	#results = (epoch_loss, epoch_acc, f1, y_true, y_pred, masks)
	results = (epoch_loss, epoch_acc, f1, y_true, y_pred)
	
	return results



def eval_(loader, model, criterion, optimizer, device):
	
	y_pred = []
	y_true = []
	
	model.eval()
	
	running_loss = 0.0
	running_corrects = 0
	i = 0
	
	for data in tqdm(loader):
		
		data = data.to(device)
		optimizer.zero_grad()
		
		output = model(data)
		_, preds = torch.max(output, 1)
		#print(preds, data.y)
		
		#print('Predictions: ', len(preds), preds)
		#print('Output: ', len(output), output)
		#print('GT: ', len(data.y[0]), data.y[0])
		loss = criterion(output, torch.tensor(data.y[0], dtype=torch.long).to(device))
		
		running_loss += loss.item()
		running_corrects += torch.sum(preds == torch.tensor(data.y[0], dtype=torch.long).to(device))
		
		y_pred.extend(preds.detach().cpu().numpy())
		y_true.extend(data.y[0])
		
		i+=1
		  
	epoch_loss = running_loss / i
	epoch_acc = running_corrects.double() / i
		
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	acc = (y_pred == y_true).mean()
	f1 = f1_score(y_true, y_pred, average = 'weighted')
	
	print('VAL: Epoch Loss: {}, Epoch Acc: {}, weighted F1: {} '.format(epoch_loss, acc, f1))
	
	#results = (epoch_loss, epoch_acc, f1, y_true, y_pred, masks)
	results = (epoch_loss, epoch_acc, f1, y_true, y_pred)
	
	return results



def Folds_train_val(train_Graphs, val_Graphs, batch_size, num_epochs, n_convs, weight_decay, hidden_size, device, lr, step_size, lr_decay, m, fold, input_size, augment=False, opt=False, testing=False, it=None):

	"""
	the data of the pt1 dataset is split into train val and test in 4 different ways
	this function trains and validates using the train and val split of one of these 4 possible splits
	:param batch_size: (int)
	:param num_epochs: (int) number of epochs
	:param num_layers: (int) number of graph convolutional layers
	:param num_input_features: (int) number of node features
	:param hidden: (int) number of hidden representations per node
	:param device: (str) either "cpu" or "cuda"
	:param lr: learning rate
	:param step_size: indicates after how many epochs the learning rate is decreased by a factor of lr_decay
	:param lr_decay: factor by which the learning rate is decreased
	:param m: str, the model that should be trained
	:param folder: which dataset to use (33 or 4 node features)
	:param augment: boolean, determines whether the dataset should be augmented or not
	:param fold: int, determines which of the 4 possible splits is considered
	:param opt: (bool), determine whether the function is called during the hyperparameter optimization or not
	:return:
	"""

	print('nb convs: ', n_convs)
	print('hidden size: ', hidden_size)
	print('lr: ', lr)
	print('lr decay: ', lr_decay)
	print('weight decay: ', weight_decay)
	print('step size: ', step_size)
	
	# create a vector to store all the parameters and the final best loss and F1
	vect = []
	vect.append(n_convs)
	vect.append(hidden_size)
	vect.append(lr)
	vect.append(lr_decay)
	vect.append(weight_decay)
	vect.append(step_size)
	
	# get the training and validation data lists
	# augment data by adding/subtracting small random values from node features
	if augment:
		num_train = len(all_train_lists[fold])
		# num_train_aug = len(all_train_aug_lists[k])
		indices = list(range(0, num_train)) # get all original graphs

		# randomly select augmented graphs
		n_aug=5			# n_aug determines by which factor the dataset should be augmented
		choice = random.sample(range(1,n_aug), n_aug-1)
		for j in choice:
			indices.extend(random.sample(range(num_train*j, num_train*(j+1)),num_train))

		# create the train_data_list and val_data_list used for the DataLoader
		train_data_list = [all_train_aug_lists[fold][i] for i in indices] # contains all original graphs plus num_aug augmented graphs
		val_data_list = all_val_lists[fold]

		print("augm. train size: " + str(len(train_data_list)) + "   val size: "+ str(len(val_data_list)))


	else:
		# create the train_data_list and val_data_list used for the DataLoader
		train_data_list = train_Graphs
		val_data_list = val_Graphs
		# print("train size: " + str(len(train_data_list)) + "   val size: " + str(len(val_data_list)))

	print("train size: " + str(len(train_data_list)) + "   val size: " + str(len(val_data_list)))

	'''
	# initialize train loader
	train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True, drop_last=True)
	# initialize val loader
	val_loader = DataLoader(val_data_list, batch_size=batch_size, shuffle=True)
	'''
	
	train_dataset = torch_geometric.data.DataLoader(train_data_list, batch_size=1, shuffle = True)
	val_dataset = torch_geometric.data.DataLoader(val_data_list, batch_size=1, shuffle=True)

	print('num layers: ', str(n_convs))
	
	# initialize the model
	if m == 'GraphSage':
		model = GraphSage(num_of_feat=input_size, num_layers = n_convs, hidden=hidden_size).to(device)
	elif m == 'GATv2Net':
		model = GATv2Net(num_of_feat=input_size, num_layers = n_convs, hidden=hidden_size).to(device)
	elif m == 'GCN':
		model = GCN(num_of_feat=input_size, num_layers = n_convs, hidden=hidden_size).to(device)
	elif m == 'GIN':
		model = GIN(num_of_feat=input_size, num_layers = n_convs, hidden=hidden_size).to(device)
		
	optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay = weight_decay)
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=lr_decay, step_size = step_size)
	criterion = nn.NLLLoss().to(device)
	
	train_loss = []
	train_acc = []
	train_f1 = []
	val_loss = []
	val_acc = []
	val_f1 = []
	
	val_res_loss = [] #will contain the smallest validation loss
	val_res = [] # will contain the corresponding validation accuracy

	train_accs = [] # will contain the training accuracy of every epoch
	val_accs = [] # will contain the validation accuracy of every epoch

	losses = [] # will contain the training loss of every epoch
	val_losses = [] # will contain the validation loss of every epoch

	# compute training and validation accuracy for every epoch
	for epoch in range(num_epochs):
		if epoch == 0:
			train_results = eval_(train_dataset, model, criterion, optim, device)
			#train_accs.append(train_results[1].cpu().detach().numpy())
			train_accs.append(train_results[2])
			losses.append(train_results[0])

			val_results  = eval_(val_dataset, model, criterion, optim, device)  # compute the accuracy for the test data
			#running_val_acc = np.asarray([0, 0, val_results[1].cpu().detach().numpy()])
			running_val_acc = np.asarray([0, 0, val_results[2]])
			running_val_loss = np.asarray([0, 0, val_results[0]])
			#val_accs.append(val_results[1].cpu().detach().numpy())
			val_accs.append(val_results[2])
			val_losses.append(val_results[0])
			val_res_loss.append(running_val_loss)
			val_res = np.copy(running_val_acc)
			min_loss = val_results[0]
			best_f1 = val_results[2]
		# train the model
		train_results = train(train_dataset, model, criterion, optim, device)
		#train_loss.append(train_results[0])
		#train_acc.append(train_results[1])
		#train_f1.append(train_results[2])
		lr_scheduler.step()
		# ge train acc and loss
		#train_results  = eval_(train_dataset, model, criterion, optim, device)  # compute the accuracy for the training data
		#train_accs.append(train_results[1].cpu().detach().numpy())
		train_accs.append(train_results[2])
		losses.append(train_results[0])

		# get validation acc and loss
		val_results = eval_(val_dataset, model, criterion, optim, device)  # compute the accuracy for the validation data
		running_val_acc[0] = running_val_acc[1]
		running_val_acc[1] = running_val_acc[2]
		#running_val_acc[2] = val_results[1].cpu().detach().numpy()
		running_val_acc[2] = val_results[2]

		print('running val acc: ', running_val_acc)
		print('val res: ', val_res)

		if min_loss > val_results[0]:
			min_loss = val_results[0]
			best_f1 = val_results[2]
			val_res = np.copy(running_val_acc)
			
		'''
		if np.mean(running_val_acc) > np.mean(val_res) and not testing:		 # if this is current best save the list of predictions and corresponding labels
			val_res = np.copy(running_val_acc)

		if running_val_acc[2] > val_res[2] and testing:  # if this is current best save the list of predictions and corresponding labels
			img_name_res = img_name
			TP_TN_FP_FN_res = np.copy(TP_TN_FP_FN)
			val_res = np.copy(running_val_acc)
			torch.save(model, "Parameters/" + folder_short + m + "_fold" + str(fold) + "it"+str(it)+".pt")
		'''
		val_accs.append(val_acc)
		val_losses.append(val_loss)

	'''
	if stdev(losses[-20:]) < 0.05 and mean(train_accs[-20:])<0.55:
		boolean = True
	else:
		boolean = False
	'''
	print("best val accuracy:", best_f1)
	vect.append(best_f1)
	vect.append(min_loss)
	return(vect, val_res, best_f1, min_loss, np.asarray(train_accs, dtype="object"), np.asarray(val_accs, dtype="object"), np.asarray(losses, dtype="object"), np.asarray(val_losses, dtype="object"))   # the boolean tells that train_and_val was completed (good param combination)



def opt(search_space, num_epochs, m, iterations, device, opt_run, model_name, input_size, nb_folds, path_csv, trial_save_path, train_graphs_all, val_graphs_all):
	
	
	def objective_function(params):
		"""
		this functions takes a given set of hyperparameters and uses them to treain and validate a model.
		it returns the measure that we want to be minimized (in this case the negative accuracy)

		The objective function returns a dictionary.
		The fmin function looks for some special key-value pairs in the return value of the objective function
		which will be then passed to the optimization algorithm
		:param params: set of hyperparameters
		:return: dictionary containing the score and STATUS_OK
		"""
		val_acc = []
		vects_all = []
		vect_all_folds = []
		res_all_folds = []
		best_f1_all_folds = []
		min_loss_all_folds = []
		train_accs_all_folds = []
		val_accs_all_folds = []
		train_losses_all_folds = []
		val_losse_all_folds = []


		score_all_folds = []
		for i in range(nb_folds):
			print(i, len(train_graphs_all[i]))
			vect, res, best_f1, min_loss, train_accs, val_accs, train_losses, val_losses = Folds_train_val(**params, train_Graphs = train_graphs_all[i], val_Graphs = val_graphs_all[i], batch_size = 1, num_epochs = num_epochs, device = device, m = m, fold = i, input_size = input_size)
			vect_fold = ['fold' + str(i)]
			vect_fold.extend(vect)
			vect_all_folds.append(vect)
			res_all_folds.append(res)
			best_f1_all_folds.append(best_f1)
			min_loss_all_folds.append(min_loss)
			train_accs_all_folds.append(train_accs)
			val_accs_all_folds.append(val_accs)
			train_losses_all_folds.append(train_losses)
			val_losse_all_folds.append(val_losses)
			#keep track of all values
			with open(path_csv + '/' + m + '.csv', 'a') as file:
				writer = csv.writer(file) #this is the writer object
				writer.writerow(vect_fold) # this will list out the names of the columns which are always the first entrries
		
			#vects_all
			val_acc = np.mean(res)
			#score = mean(val_acc)								   # compute the average accuracy from the 10 train and validation runs
			score = val_acc
			score_all_folds.append(score)
		score_mean = np.mean(score_all_folds)
		print("Model: " + str(m))
		print("Parameters: " + str(params))
		print("score: " + str(score_mean))
		print("############################################")
		return {"loss": -score, "status": STATUS_OK, "values":  np.array(vect_all_folds).mean()}		  # return the dictionary as it is needed by the fmin function
		# validation accuracy is used as a measure of performance. Because fmin tries to minimize the returned value the negative accuracy is returned
		

	try:
		trials = pickle.load(open(trial_save_path + '/' +  m + '.hyperopt', "rb"))
	except:
		trials = Trials()


	best_param = fmin(				  # fmin returns a dictionary with the best parameters
		fn=objective_function,		  # fn is the function that is to be minimize
		space=search_space,			 # searchspace for the parameters
		algo=tpe.suggest,			   # Search algorithm: Tree of Parzen estimators
		max_evals=iterations,		   # number of parameter combinations that should be evalutated
		trials=trials				   # by passing a trials object we can inspect all the return values that were calculated during the experiment
		#rstate=np.random.RandomState(111)   
	)
	
	# save the trials to restart hyperopt later on
	pickle.dump(trials, open(trial_save_path + '/' +  m + '.hyperopt', "wb"))
	
	
	print("done")
	loss = [x["result"]["loss"] for x in trials.trials]	 # trials.trials is a list of dictionaries representing everything about the search
															# loss is a list of all scores (negative accuracies) that were obtained during the experiment
	best_param_values = [x for x in best_param.values()]	# best_param is a dictionary with the parameter name as key and the best value as value
															# best_param.values() output the best parameter values
															# --> best_param_values is a list of the parametervalues that performed best


	print("")
	print("##### Results " + m)
	print("Score best parameters: ", min(loss))		# min(loss) * -1 is the accuracy obtained with the best combination of parameter values
	print("Best parameters: ", best_param)				  # best_param is the dictionary containing the parameters as key and the best values as value
	#print("Time elapsed: ", time() - start)
	print("Parameter combinations evaluated: ", iterations)
	print("############################################")
	print("############################################")
	print("############################################")
	
	return trials
