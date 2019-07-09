
# -*- coding: utf-8 -*-
import os
import time
import pickle

class Config():
	#*****************************
	ID = 'default'# identify model : saving as ID+model like 'default_gcn_appr'
	acc = 0.00

	#%% --loading data
	dataset = 'pubmed'#'Dataset string.'
	val_size = 500



	#%% --model
	model = 'gcn_appr'#'Model string.'
	hidden1 = 16#'Number of units in hidden layer 1.'
	dropout = 0.0#'Dropout rate (1 - keep probability).')

	rank0 = 100
	rank1 = 100

	#(only for MLP)
	'''example:
	MLP: 
	3 hidden layers, [64,32,16], output dim = 3,
	set  MLP_hidden_layers=3, MLP_hidden_dim = [64,32,16].

	if MLP_hidden_layers==0, it's LR. 
	'''
	MLP_hidden_dim = [16] #exclude output layers
	MLP_hidden_layers = 1#the last layer hidden dim is num_of_classes


	#%% --training
	batch_size = 256
	learning_rate = 0.001#'Initial learning rate.'
	epochs = 200#'Number of epochs to train.'
	weight_decay = 5e-4#'Weight for L2 loss on embedding matrix.'
	early_stopping = 30#'Tolerance for early stopping (# of epochs).'
	
	max_degree = 3#'Maximum Chebyshev polynomial degree.'
	
	t_each_batch = 1

	#*****************************

	def parse(self, kwargs):
		'''
		根据字典kwargs 更新 config参数
		'''
		# 更新配置参数
		for k, v in kwargs.items():
			if not hasattr(self, k):
				raise Exception("Warning: config has not attribute <%s>" % k)
			setattr(self, k, v)

	def print_config(self):
		# 打印配置信息
		print('+++++++++++++++++++++++++++++++++++++++')
		print('user config:')
		for k, v in self.__class__.__dict__.items():
			if not k.startswith('__') and k != 'parse' and k != 'print_config':
				print('    {} : {}'.format(k, getattr(self, k)))
		print('+++++++++++++++++++++++++++++++++++++++')

def save_config(config, fine_tune=False):
	'''
	save config

	return saving dir_path
	'''
	file_dir = './snapshot/'+config.dataset+'_'+config.model+'_'+config.ID
	# like :  './snapshot/pubmed_gcn_appr_default'
	if fine_tune:
		file_dir += '/finetune_model/'+time.strftime("%Y%m%d_%H_%M", time.localtime())
		# like : './snapshot/pubmed_gcn_appr_default/finetune_model/20190704_17_48'
	if not os.path.exists(file_dir):
		os.makedirs(file_dir)

	fn = file_dir+'/config.pkl'
	with open(fn, 'wb') as f:
		pk = pickle.dump(config, f)

	return file_dir

def load_config(fn):
	'''
	load config from path: fn

	return config
	'''
	with open(fn, 'rb') as f:
		config = pickle.load(f)

	return config





	
	

	
	