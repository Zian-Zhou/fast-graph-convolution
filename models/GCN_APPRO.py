'''
2 layers: GCN+GCN

SAMPLING 2 TIMES
'''


import tensorflow as tf
from .BasicModule import BasicModule

import sys
sys.path.append("..")

from utils.metrics import *
from utils.layers import *



class GCN_APPRO(BasicModule):
	def __init__(self, placeholders, input_dim, config, **kwargs):
		super(GCN_APPRO, self).__init__(**kwargs)
		self.config = config
		self.inputs = placeholders['features']
		self.input_dim = input_dim
		self.output_dim = placeholders['labels'].get_shape().as_list()[1]
		self.placeholders = placeholders
		self.supports = placeholders['support']

		self.optimizer = tf.train.AdamOptimizer(learning_rate = config.learning_rate)

		self.build()

	def _loss(self):
		for var in self.layers[0].vars.values():
			self.loss += self.config.weight_decay * tf.nn.l2_loss(var)

		self.loss += softmax_cross_entropy(self.outputs, self.placeholders['labels'])

	def _accuracy(self):
		self.accuracy = accuracy(self.outputs, self.placeholders['labels'])

	def _f1score(self):
		self.f1score = micro_F1_score(self.outputs, self.placeholders['labels'])

	def _build(self):
		self.layers.append(GraphConvolution( input_dim = self.input_dim,
											 output_dim = self.config.hidden1,
											 placeholders = self.placeholders,
											 support = self.supports[0],
											 act = tf.nn.relu,
											 dropout = True,
											 sparse_inputs = False,
											 logging = self.logging))

		self.layers.append(GraphConvolution( input_dim = self.config.hidden1,
											 output_dim = self.output_dim,
											 placeholders = self.placeholders,
											 support = self.supports[1],
											 act = lambda x: x,
											 dropout = True,
											 logging = self.logging))

	def predict(self):
		return tf.nn.softmax(self.outputs)

