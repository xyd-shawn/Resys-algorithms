"""
Simply implement PNN for CTR prediction
paper: https://arxiv.org/pdf/1611.00144.pdf
"""

import os
import copy

import numpy as np
import tensorflow as tf
from tqdm import tqdm


class PNN(object):
    def __init__(self, n_features_list, mode='inner', **config):
        self.n_features_list = n_features_list
        self.n_features_total = sum(self.n_features_list)
        self.n_fields = len(self.n_features_list)
        self.mode = mode
        self.embedding_dim = config.get('embedding_dim', 16)
        self.hidden_units = config.get('hidden_units', [200, 200, 200])
        self.batch_size = config.get('batch_size', 32)
        self.random_seed = config.get('random_seed', 1)
        self._build_model()

    def _build_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)        # set random seed

            # input data
            self.features = tf.placeholder(tf.int32, shape=[None, self.n_features_total], name='input_features')
            self.labels = tf.placeholder(tf.float32, shape=[None, 1], name='labels')
            self.dropout_keep = tf.placeholder(tf.float32, name='keep_prob')

            # split the input according to field information
            self.field_values = tf.split(self.features, num_or_size_splits=self.n_features_list, axis=1)

            # the first embedding layer
            with tf.variable_scope('embedding_layer'):
                self.W0 = dict()
                self.embedding_res = dict()
                for i in range(self.n_fields):
                    self.W0[i] = tf.Variable(tf.truncated_normal(shape=[self.n_features_list[i], self.embedding_dim], mean=0.0, stddev=0.1),
                                             name='embedding_%d' % i)
                    self.embedding_res[i] = tf.sparse_tensor_dense_matmul(self.field_values[i], self.W0[i])

            # the second layer, which implements inner product or outer product
            with tf.variable_scope('product_layer'):
                self.b1 = tf.Variable(tf.constant(0.0, shape=[1, self.hidden_units[0]]), name='bias_1')
                embedding_concat = tf.concat([self.embedding_res[i] for i in range(self.n_fields)], axis=1)
                self.Wl = tf.Variable(tf.truncated_normal(shape=[self.embedding_dim * self.n_fields, self.hidden_units[0]], mean=0.0, stddev=0.1),
                                      name='W_l')
                tf.matmul(embedding_concat, self.Wl)
