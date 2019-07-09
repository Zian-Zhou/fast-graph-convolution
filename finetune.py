from __future__ import division
from __future__ import print_function

import fire
import time
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import models
import pickle

from utils.utils import *
from config import Config, save_config, load_config

import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import warnings
warnings.filterwarnings("ignore")

def iterate_minibatches_listinputs(inputs, batchsize, shuffle=False):
	assert inputs is not None
	numSamples = inputs[0].shape[0]
	if shuffle:
		indices = np.arange(numSamples)
		np.random.shuffle(indices)
	for start_idx in range(0, numSamples-batchsize+1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx+batchsize]
		else:
			excerpt = slice(start_idx, start_idx+batchsize)
		yield [input[excerpt] for input in inputs]

def finetune(exact_model_dir=None, dataset_model_id = 'pubmed_gcn_appr_default', learning_rate = None, lr_decay = 1, **kwargs):
	'''
	finetune function: similar to training code, just change learning_rate
	
	#file_path: file_dir model saved at. like './snapshot/pubmed_gcn_appr_default'

	exact_model_dir: finetune the exact model, which may be at './snapshot/pubmed_gcn_appr_default/' 
														or at './snapshot/pubmed_gcn_appr_default/finetune_model/20190704_20_25/'

	dataset_model_id: saving model name  like 'pubmed_gcn_appr_default'

	learning_rate: new learning_rate or None(using config.learning_rate)

	lr_decay: if learning_rate is None, using config.learning_rate * lr_decay. For finetuning, need to using lower learning_rate than that during training before
	
	'''
	if exact_model_dir is None:
		file_path = './snapshot/' + dataset_model_id#file_path = './snapshot/pubmed_gcn_appr_default'
	else:
		file_path = exact_model_dir

	config_path = file_path + '/config.pkl' # config_path = './snapshot/pubmed_gcn_appr_default/config.pkl'

	args = load_config(config_path)
	if learning_rate is not None:
		args.learning_rate = learning_rate
	else:
		args.learning_rate *= lr_decay
		print('Using new learning_rate: {}'.format(args.learning_rate))

	args.print_config()
	args.parse(kwargs)
	print('【Finetuning】: using learning_rate = {}.........\n'.format(args.learning_rate))

	#load data
	print('==============================================================================')
	print('《Loading data》 【{}】.....'.format(args.dataset))
	adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset, args.val_size)

	train_index = np.where(train_mask)[0]
	adj_train = adj[train_index, :][:, train_index]
	train_mask = train_mask[train_index]
	y_train = y_train[train_index]
	val_index = np.where(val_mask)[0]
	# adj_val = adj[val_index, :][:, val_index]
	val_mask = val_mask[val_index]
	y_val = y_val[val_index]
	test_index = np.where(test_mask)[0]
	# adj_test = adj[test_index, :][:, test_index]
	test_mask = test_mask[test_index]
	y_test = y_test[test_index]

	numNode_train = adj_train.shape[0]####训练节点数：18217

	
	print('train set:{} ; val set: {} ; test set: {}'.format(len(train_index),len(val_index),len(test_index)))
	print('train numNode:',numNode_train)
	print('==============================================================================')


	# Some prepocessing
	print('《Doing prepocessing》.........')
	features = nontuple_preprocess_features(features).todense()##### 19717*500  共500维特征
	train_features = features[train_index]##### 18217*500


	if args.model == 'gcn_appr':
		normADJ_train = nontuple_preprocess_adj(adj_train)#### adj+I:self connected
		normADJ = nontuple_preprocess_adj(adj)
		# normADJ_val = nontuple_preprocess_adj(adj_val)
		# normADJ_test = nontuple_preprocess_adj(adj_test)

		num_supports = 2
		model_func = models.GCN_APPRO

	elif args.model == 'dense_gcn_appr':
		normADJ_train = nontuple_preprocess_adj(adj_train)
		normADJ = nontuple_preprocess_adj(adj)

		num_supports = 1
		model_func = models.DENSE_GCN_APPR

	elif args.model == 'mlp':
		num_supports = 0
		model_func = models.MLP

	print('Done.')
	print('==============================================================================')


	# Define placeholders
	placeholders = {
		'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
		'features': tf.placeholder(tf.float32, shape=(None, features.shape[1])),
		'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
		'labels_mask': tf.placeholder(tf.int32),
		'dropout': tf.placeholder_with_default(0., shape=()),
		'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
		'epoch':tf.placeholder_with_default(0,shape=())
	}

	# initial model
	model = model_func(placeholders, input_dim=features.shape[-1], config=args, logging=True)

	# restore session
	sess = tf.Session()
	saver = tf.train.Saver()
	print('==============================================================================')
	print('Retoring Variables from checkpoint......')
	if os.path.exists(file_path+"/model_{:.5f}.ckpt.index".format(args.acc)):
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		saver.restore(sess, file_path+"/model_{:.5f}.ckpt".format(args.acc))
		print('\n【model】: {}/model_{:.5f}.ckpt'.format(file_path,args.acc))
		print('Done.\n')
		print('==============================================================================\n')
	else:
		print('Have not found {}\n'.format(file_path+"/model_{:.5f}.ckpt.index".format(args.acc)))

	# define model evaluate function
	def evaluate(features, support, labels, mask, placeholders):
		t_test = time.time()
		feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
		outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
		return outs_val[0], outs_val[1], (time.time() - t_test)
	'''# retore session, don't need to initial varaiables
	# initial varaiables
	sess.run(tf.global_variables_initializer())
	'''

	cost_val = []

	# gcn_appr： 2 layers
	if args.model == 'gcn_appr':
		rank0 = args.rank0
		rank1 = args.rank1

		#p0 = column_prop(normADJ_train)

		valSupport = [sparse_to_tuple(normADJ), sparse_to_tuple(normADJ[val_index, :])]#### 2 tuples: each tuple ~ coords values shape
		testSupport = [sparse_to_tuple(normADJ), sparse_to_tuple(normADJ[test_index, :])]

		t = time.time()

		print('***************************《Begin Training》............**************************')

		for epoch in range(args.epochs):
			t1 = time.time()

			n = 0

			for batch in iterate_minibatches_listinputs([normADJ_train, y_train, train_mask], batchsize=args.batch_size, shuffle=True):
				[normADJ_batch, y_train_batch, train_mask_batch] = batch
				if sum(train_mask_batch) < 1:
					continue

				# importance sampling
				p1 = column_prop(normADJ_batch)
				q1 = np.random.choice(np.arange(numNode_train), rank1, p=p1)  # top layer

				support1 = sparse_to_tuple(normADJ_batch[:, q1].dot(sp.diags(1.0 / (p1[q1] * rank1))))### 修改边权重，对应论文公式（10）

				p2 = column_prop(normADJ_train[q1, :])#### 第二层的采样根据第一层采样的节点做，而不是batch的节点
				q0 = np.random.choice(np.arange(numNode_train), rank0, p=p2)
				support0 = sparse_to_tuple(normADJ_train[q1, :][:, q0])#### shape: rank1*rank0
				features_inputs = sp.diags(1.0 / (p2[q0] * rank0)).dot(train_features[q0, :])  # selected nodes for approximation

				# Construct feed dictionary
				feed_dict = construct_feed_dict(features_inputs, [support0, support1], y_train_batch, train_mask_batch, placeholders)
				feed_dict.update({placeholders['dropout']: args.dropout})
				feed_dict.update({placeholders['epoch']: epoch})

				# each batch training times
				for _ in range(args.t_each_batch):
					outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)#, model.learning_rate

			# ----------------validation-----------
			cost, acc, duration = evaluate(features, valSupport, y_val, val_mask, placeholders)
			cost_val.append(cost)

			print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
				"train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost), #'lr={}'.format(outs[3]),
				"val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t1))

			# early stopping
			if epoch > args.early_stopping and cost_val[-1] > np.mean(cost_val[-(args.early_stopping + 1):-1]):
				print("Early stopping...")
				break

		train_duration = time.time() - t

		# -----------------testing--------------
		test_cost, test_acc,test_duration = evaluate(features, testSupport, y_test, test_mask, placeholders)
		print("rank1 = {};".format(rank1), "rank0 = {};".format(rank0), "cost=", "{:.5f};".format(test_cost),
			  "accuracy=", "{:.5f};".format(test_acc), "training time per epoch=", "{:.5f};".format(train_duration/epoch))

	elif args.model == 'dense_gcn_appr':
		rank0 = args.rank0

		valSupport = [sparse_to_tuple(normADJ[val_index, :])]#### 2 tuples: each tuple ~ coords values shape
		testSupport = [sparse_to_tuple(normADJ[test_index, :])]

		t = time.time()

		print('***************************《Begin Training》............**************************')

		for epoch in range(args.epochs):
			t1 = time.time()

			n = 0

			for batch in iterate_minibatches_listinputs([normADJ_train, y_train, train_mask], batchsize=args.batch_size, shuffle=True):
				[normADJ_batch, y_train_batch, train_mask_batch] = batch
				if sum(train_mask_batch) < 1:
					continue

				# importance sampling
				p0 = column_prop(normADJ_batch)
				q0 = np.random.choice(np.arange(numNode_train), rank0, p=p0)  # top layer

				support0 = sparse_to_tuple(normADJ_batch[:, q0].dot(sp.diags(1.0 / (p0[q0] * rank0))))### 修改边权重，对应论文公式（10）

				features_inputs = train_features[q0, :]  # selected nodes for approximation

				# Construct feed dictionary
				feed_dict = construct_feed_dict(features_inputs, [support0], y_train_batch, train_mask_batch, placeholders)
				feed_dict.update({placeholders['dropout']: args.dropout})
				feed_dict.update({placeholders['epoch']: epoch})

				# each batch training times
				for _ in range(args.t_each_batch):
					outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)#, model.learning_rate

			# ----------------validation-----------
			cost, acc, duration = evaluate(features, valSupport, y_val, val_mask, placeholders)
			cost_val.append(cost)

			print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
				"train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost), #'lr={}'.format(outs[3]),
				"val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t1))

			# early stopping
			if epoch > args.early_stopping and cost_val[-1] > np.mean(cost_val[-(args.early_stopping + 1):-1]):
				print("Early stopping...")
				break

		train_duration = time.time() - t

		# -----------------testing--------------
		test_cost, test_acc,test_duration = evaluate(features, testSupport, y_test, test_mask, placeholders)
		print("rank0 = {};".format(rank0), "cost=", "{:.5f};".format(test_cost),
			  "accuracy=", "{:.5f};".format(test_acc), "training time per epoch=", "{:.5f};".format(train_duration/epoch))

	elif args.model == 'mlp':
		t = time.time()
		print('***************************《Begin Training》............**************************')
		for epoch in range(args.epochs):
			t1 = time.time()
			for batch in iterate_minibatches_listinputs([train_features, y_train, train_mask], batchsize=args.batch_size, shuffle=True):
				[train_features_batch, y_train_batch, train_mask_batch] = batch
				if sum(train_mask_batch) < 1:
					continue

				#features_inputs = train_features_batch

				# Construct feed dictionary
				feed_dict = construct_feed_dict(train_features_batch, [], y_train_batch, train_mask_batch, placeholders)
				feed_dict.update({placeholders['dropout']: args.dropout})
				feed_dict.update({placeholders['epoch']: epoch})

				# each batch training times
				for _ in range(args.t_each_batch):
					outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)#, model.learning_rate

			# ----------------validation-----------
			cost, acc, duration = evaluate(features[val_index,:], [], y_val, val_mask, placeholders)
			cost_val.append(cost)

			print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
				"train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost), #'lr={}'.format(outs[3]),
				"val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t1))

			# early stopping
			if epoch > args.early_stopping and cost_val[-1] > np.mean(cost_val[-(args.early_stopping + 1):-1]):
				print("Early stopping...")
				break

		train_duration = time.time() - t

		# -----------------testing--------------
		test_cost, test_acc,test_duration = evaluate(features[test_index,:], [], y_test, test_mask, placeholders)
		print("MLP layers {}".format(args.MLP_hidden_dim),"cost=", "{:.5f};".format(test_cost),
			  "accuracy=", "{:.5f};".format(test_acc), "training time per epoch=", "{:.5f};".format(train_duration/epoch))


	# save config and finetuned model(tf_graph)
	if test_acc<args.acc:
		print('Have not got a better model, try other parameters!')
	else:
		args.acc = test_acc
		file_dir = save_config(args, fine_tune=True)#新目录
		saver.save(sess, file_dir+'/model_{:.5f}.ckpt'.format(test_acc))#finetune_model_
		print('The better model with acc:{:.5f} has been saved at dir:{}.'.format(test_acc, file_dir))

if __name__ == '__main__':
	fire.Fire()

'''
# using

# if want to finetune the model which has not been finetuned, just give model_id.
python finetune.py finetune --dataset_model_id=pubmed_gcn_appr_default --lr_decay=1.1

# if want to finetune model which has been finetuned so that path should be infered.
python finetune.py finetune --exact_model_dir=./snapshot/pubmed_gcn_appr_default/finetune_model/20190704_20_25 --lr_decay=0.1
'''