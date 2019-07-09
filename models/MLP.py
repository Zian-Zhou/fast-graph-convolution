'''
MLP layers

'''

import tensorflow as tf
from .BasicModule import BasicModule

import sys
sys.path.append("..")

from utils.metrics import *
from utils.layers import *



class MLP(BasicModule):
	def __init__(self, placeholders, input_dim, config, **kwargs):
		super(MLP, self).__init__(**kwargs)
		self.config = config
		self.inputs = placeholders['features']
		self.input_dim = input_dim
		self.output_dim = placeholders['labels'].get_shape().as_list()[1]
		self.placeholders = placeholders
	
		self.MLP_layers = self.config.MLP_hidden_layers   #  number of hidden layers: like 1
		if self.MLP_layers==0:
			self.hiddens = []
		else:
			self.hiddens = self.config.MLP_hidden_dim # list of dim of every hidden layers: like [16]
			assert self.MLP_layers == len(self.hiddens), "MLP_hidden_layers is not equal to len of MLP_hidden_dim, check args input!"

		self.optimizer = tf.train.AdamOptimizer(learning_rate = config.learning_rate)

		self.build()

	def _loss(self):
		for var in self.layers[0].vars.values():
			self.loss += self.config.weight_decay * tf.nn.l2_loss(var)

		#self.loss += softmax_cross_entropy(self.outputs, self.placeholders['labels'])
		self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
													self.placeholders['labels_mask'])
		

	def _accuracy(self):
		#self.accuracy = accuracy(self.outputs, self.placeholders['labels'])
		self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
										self.placeholders['labels_mask'])

	def _f1score(self):
		self.f1score = micro_F1_score(self.outputs, self.placeholders['labels'])

	def _build(self):
		#hidden layers
		self.hiddens = [self.input_dim] + self.hiddens # [16]——>[input_dim,16]
		for i in range(self.MLP_layers):
			self.layers.append(Dense(   input_dim = self.hiddens[i],
										output_dim = self.hiddens[i+1],
										placeholders = self.placeholders,
										act = tf.nn.relu,
										dropout = True,
										sparse_inputs = False,
										logging = self.logging))	

		# output layer
		# if self.MLP_layers==0, self.hiddens is [self.input_dim], so self.hiddens[-1] is self.input_dim
		self.layers.append(Dense(   input_dim = self.hiddens[-1],
									output_dim = self.output_dim,
									placeholders = self.placeholders,
									act = lambda x:x,
									dropout = True,
									sparse_inputs = False,
									logging = self.logging))

	def predict(self):
		return tf.nn.softmax(self.outputs)

