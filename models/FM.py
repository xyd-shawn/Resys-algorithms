"""
Simply implement Factorization Machines
paper: https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
"""

import numpy as np
import tensorflow as tf


class FM(object):
    def __init__(self, feature_type='mixed', task_type='pred_CTR', n_features=10, feature_size=None, **config):
        self.feature_type = feature_type
        self.task_type = task_type
        self.n_features = n_features
        self.feature_size = feature_size
        if self.feature_type in ['discrete', 'categorical']:
            if not self.feature_size:
                print('Error! feature_size should be the maximum value of features')
        else:
            if not n_features == feature_size:
                print('Error! feature_size should equal to n_features')
        self.factor_dim = config.get('factor_dim', 16)
        self.batch_size = config.get('batch_size', 32)
        self.n_epochs = config.get('n_epochs', 100)
        self.optimizer_type = config.get('optimizer_type', 'sgd')
        self.lr = config.get('learning_rate', 0.01)
        self.momentum = config.get('momentum', 0.9)
        self.random_seed = config.get('random_seed', 1)
        self._build_model()

    def _build_model(self):

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            self.labels = tf.placeholder(tf.float32, shape=[None, 1], name='labels')

            # compute the linear part and the interaction part of fm respectively
            self.b = tf.Variable(tf.constant(0.0), name='bias')
            self.w = tf.Variable(tf.truncated_normal(shape=[self.feature_size, 1], mean=0.0, stddev=0.1),
                                 name='feature_weight')
            self.v = tf.Variable(tf.truncated_normal(shape=[self.feature_size, self.factor_dim], mean=0.0, stddev=0.1),
                                 name='feature_interaction')
            if self.feature_type in ['discrete', 'categorical']:
                self.features = tf.placeholder(tf.int32, shape=[None, self.n_features], name='input_features')
                self.linear_embedding = tf.nn.embedding_lookup(self.w, self.features)
                self.linear_part = tf.add(tf.reduce_sum(self.linear_embedding, axis=1),
                                          self.b * tf.ones_like(self.labels))
                self.interact_embedding = tf.nn.embedding_lookup(self.v, self.features)
                self.embed_sum_square = tf.square(tf.reduce_sum(self.interact_embedding, axis=1))
                self.embed_square_sum = tf.reduce_sum(tf.square(self.interact_embedding), axis=1)
            else:
                self.features = tf.placeholder(tf.float32, shape=[None, self.n_features], name='input_features')
                self.linear_part = tf.add(tf.sparse_tensor_dense_matmul(self.features, self.w),
                                          self.b * tf.ones_like(self.labels))
                self.embed_sum_square = tf.square(tf.sparse_tensor_dense_matmul(self.features, self.v))
                self.embed_square_sum = tf.sparse_tensor_dense_matmul(tf.square(self.features), tf.square(self.v))
            self.subtract_embed = tf.subtract(self.embed_sum_square, self.embed_square_sum)
            self.interact_part = tf.multiply(0.5, tf.reduce_sum(self.subtract_embed, axis=1, keep_dims=True))
            self.fm = tf.add(self.linear_part, self.interact_part)

            # compute loss and accuracy
            if self.task_type == 'pred_CTR':
                self.predictions = tf.sigmoid(self.fm)
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.fm))
                self.correct = tf.equal(tf.cast(tf.greater(self.predictions, 0.5), tf.float32), self.labels)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))
            else:
                self.loss = tf.nn.l2_loss(tf.subtract(self.fm, self.labels))

            # build optimizer
            if self.optimizer_type.lower() == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            elif self.optimizer_type.lower() == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            elif self.optimizer_type.lower() == 'rmsprop':
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, momentum=self.momentum)
            elif self.optimizer_type.lower() == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.momentum)

    def fit(self, X, y, n_epochs):
        for i in range(n_epochs):
            pass

    def predict(self, X):
        pass

    def evaluate(self, X, y):
        pass
