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
        self.random_seed = config.get('random_seed', 1)
        self._build_model()

    def _build_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            b = tf.Variable(tf.constant(0.0), name='bias')
            w = tf.Variable(tf.truncated_normal(shape=[self.feature_size, 1], mean=0.0, stddev=0.1),
                            name='feature_weight')
            v = tf.Variable(tf.truncated_normal(shape=[self.feature_size, self.factor_dim], mean=0.0, stddev=0.1),
                            name='feature_interaction')
            if self.feature_type in ['discrete', 'categorical']:
                self.features = tf.placeholder(tf.int32, shape=[None, self.n_features], name='input_features')
            else:
                self.features = tf.placeholder(tf.float32, shape=[None, self.n_features], name='input_features')

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def evaluate(self, X, y):
        pass
