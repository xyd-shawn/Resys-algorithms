"""
Simply implement FFM for CTR prediction
paper: https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf
"""

import tensorflow as tf


class FFM(object):
    def __init__(self, n_features, n_fields, **config):
        self.n_features = n_features
        self.n_fields = n_fields
        self.factor_dim = config.get('factor_dim', 8)
        self.optimizer_type = config.get('optimizer_type', 'sgd')
        self.lr = config.get('learning_rate', 0.01)
        self.momentum = config.get('momentum', 0.9)
        self.random_seed = config.get('random_seed', 1)
        self._build_model()

    def _build_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)        # 设置随机种子

            self.features = tf.placeholder(tf.float32, shape=[None, self.n_features], name='input_feature')
            self.fields = tf.placeholder(tf.int32, shape=[self.features], name='feature_field')
            self.labels = tf.placeholder(tf.float32, shape=[None, 1], name='labels')

            # compute FFM
            # linear part
            self.b = tf.Variable(tf.constant(0.0), name='bias')
            self.w = tf.Variable(tf.truncated_normal(shape=[self.n_features, 1], mean=0.0, stddev=0.1),
                                 name='feature_weights')
            self.linear_part = tf.add(tf.sparse_tensor_dense_matmul(self.features, self.w),
                                      self.b * tf.ones_like(self.labels))
            # interaction part
            self.v = tf.Variable(tf.truncated_normal(shape=[self.n_features, self.n_fields, self.factor_dim], mean=0.0, stddev=0.1),
                                 name='feature_interaction')

    def fit(self, X, y, field_info):
        pass

    def evaluate(self, X, y):
        pass

    def predict(self, X):
        pass