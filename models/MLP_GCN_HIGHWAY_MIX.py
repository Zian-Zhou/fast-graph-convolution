'''
MIX MLP&GCN THROUGH HIGHWAY NETWORK
'''

import tensorflow as tf
from .BasicModule import BasicModule

import sys
sys.path.append("..")

from utils.metrics import *
from utils.layers import *



class MLP_GCN_HIGHWAY_MIX(BasicModule):
	def __init__(self, placeholders, input_dim, config, **kwargs):
		super(MLP_GCN_HIGHWAY_MIX, self).__init__(**kwargs)
		
		self.gcn_activations = []
		self.mlp_activations = []
		self.gate_activations = []
		self.activations = []

		self.gcn_branch = []
		self.mlp_branch = []
		self.gate_layers = []
		self.layers = []

		self.config = config
		self.input_dim = input_dim
		self.output_dim = placeholders['labels'].get_shape().as_list()[1]
		self.placeholders = placeholders
		self.supports = placeholders['support']# just need one sampling support: [support]

		self.optimizer = tf.train.AdamOptimizer(learning_rate = config.learning_rate)

		#### 注意，placeholders['features']应该有两部分组成，第一部分用于mlp分支，第二部分用于GCN分支
		#### mlp分支主要计算batch节点的特征（仅自身节点信息），而GCN分支计算采样节点的特征（融合拓扑信息）
		self.mlp_inputs = placeholders['features'][0]
		self.gcn_inputs = placeholders['features'][1]

		self.build()

	def _loss(self):
		for var in self.layers[0].vars.values():
			self.loss += self.config.weight_decay * tf.nn.l2_loss(var)

		self.loss += softmax_cross_entropy(self.outputs, self.placeholders['labels'])

	def _accuracy(self):
		self.accuracy = accuracy(self.outputs, self.placeholders['labels'])

	def _f1score(self):
		self.f1score = micro_F1_score(self.outputs, self.placeholders['labels'])

	def build(self):
		'''
		build 2 branches
		'''
		with tf.variable_scope(self.name):
			self._build()

		# Build sequential layer model
		##############################################
		self.mlp_activations.append(self.mlp_inputs)
		for layer in self.mlp_branch:
			hidden = layer(self.mlp_activations[-1])   
			self.mlp_activations.append(hidden)

		self.gcn_activations.append(self.gcn_inputs)
		for layer in self.gcn_branch:
			hidden = layer(self.gcn_activations[-1])
			self.gcn_activations.append(hidden)

		self.gate_activations.append(self.mlp_activations[-1])
		for layer in self.gate_layers:
			hidden = layer(self.gate_activations[-1])
			self.gate_activations.append(hidden)
		
		self.activations.append( 
						self.mlp_activations[-1] * self.gate_activations[-1] +\
					 	self.gcn_activations[-1] * (1-self.gate_activations[-1])
					 )
		'''
		self.activations.append( 
						tf.concat([self.mlp_activations[-1] * self.gate_activations[-1] ,
					 	self.gcn_activations[-1] * (1-self.gate_activations[-1])],axis=1)
					 )
		'''
		for layer in self.layers:
			hidden = layer(self.activations[-1])
			self.activations.append(hidden)

		self.outputs = self.activations[-1]
		##############################################

		# Store model variables for easy access
		variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
		self.vars = {var.name: var for var in variables}

		# Build metrics
		self._loss()
		self._accuracy()
		self._f1score()

		self.opt_op = self.optimizer.minimize(self.loss)


	def _build(self):
		########## gcn
		self.gcn_branch.append(
			Dense(input_dim = self.input_dim,
					output_dim = self.config.hidden1,
					placeholders = self.placeholders,
					act = tf.nn.relu,
					dropout = True,
					sparse_inputs = False,
					logging = self.logging
				)
			)
		self.gcn_branch.append(
			GraphConvolution(input_dim = self.config.hidden1,
					output_dim = self.config.hidden1,
					placeholders = self.placeholders,
					support = self.supports[0],
					act = lambda x: x,
					dropout = True,
					logging = self.logging
				)
			)
		########## mlp
		self.mlp_branch.append(
			Dense(input_dim = self.input_dim,
					output_dim = self.config.hidden1,
					placeholders = self.placeholders,
					act = lambda x: x,#tf.nn.relu
					dropout = True,
					sparse_inputs = False,
					logging = self.logging
				)
			)
		########## gate layer
		self.gate_layers.append(
			CarryGate(input_dim = self.config.hidden1,
					output_dim = self.config.hidden1,#1
					placeholders = self.placeholders,
					act = tf.nn.sigmoid,
					bias = True,#True
					sparse_inputs = False,
					logging = self.logging
				)
			)
		########## output layer
		self.layers.append(
			Dense(input_dim = self.config.hidden1,
					output_dim = self.output_dim,
					placeholders = self.placeholders,
					act = lambda x:x,
					dropout = False,
					sparse_inputs = False,
					logging = self.logging
				)
			)


	def predict(self):
		return tf.nn.softmax(self.outputs)

