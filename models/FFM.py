"""
Simply implement FFM for CTR prediction
paper: https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf
"""

import os
import copy

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.contrib.layers import l2_regularizer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class FFM(object):
    def __init__(self, n_features, n_fields, feature_field, **config):
        self.n_features = n_features
        self.n_fields = n_fields
        self.feature_field = feature_field
        self.factor_dim = config.get('factor_dim', 8)
        self.batch_size = config.get('batch_size', 32)
        self.norm_coef = config.get('norm_coef', 0.0)
        self.optimizer_type = config.get('optimizer_type', 'sgd')
        self.lr = config.get('learning_rate', 0.01)
        self.momentum = config.get('momentum', 0.9)
        self.save_path = config.get('save_path', '../logs/FFM/')
        self.random_seed = config.get('random_seed', 1)
        self._build_model()

    def _build_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)        # set random seed
            # input data
            self.features = tf.placeholder(tf.float32, shape=[None, self.n_features], name='input_feature')
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
            self.interact_part = tf.constant(0.0)
            for i in range(self.n_features):
                for j in range(i + 1, self.n_features):
                    self.interact_part += tf.multiply(tf.reduce_sum(tf.multiply(self.v[i, self.feature_field[j]], self.v[j, self.feature_field[i]])),
                                                      tf.multiply(self.features[:, i], self.features[:, j]))
            self.ffm = tf.add(self.linear_part, self.interact_part)

            # compute loss and accuracy
            self.predictions = tf.sigmoid(self.ffm)
            self.loss_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.ffm))
            self.loss = self.loss_f + l2_regularizer(self.norm_coef)(self.w) + l2_regularizer(self.norm_coef)(self.v)
            self.correct = tf.equal(tf.cast(tf.greater(self.predictions, 0.5), tf.float32), self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

            # build optimizer
            if self.optimizer_type.lower() == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            elif self.optimizer_type.lower() == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            elif self.optimizer_type.lower() == 'rmsprop':
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, momentum=self.momentum)
            elif self.optimizer_type.lower() == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.momentum)
            self.optimizer_loss = self.optimizer.minimize(self.loss)

            # create session and init
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=tf_config)
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def load_pretrain_weights(self):
        self.saver.restore(self.sess, self.save_path)

    def fit(self, X, y, n_epochs):
        train_X = copy.deepcopy(X)
        sz = train_X.shape[0]
        n_batches = sz // self.batch_size
        train_y = copy.deepcopy(y).reshape([sz, 1])
        train_loss_list = []
        for epoch in range(1, n_epochs + 1):
            train_ind = np.random.permutation(sz)
            X_train, y_train = train_X[train_ind], train_y[train_ind]
            epoch_loss_list = []
            for i in tqdm(range(n_batches)):
                X_batch = X_train[(i * self.batch_size):((i + 1) * self.batch_size)]
                y_batch = y_train[(i * self.batch_size):((i + 1) * self.batch_size)]
                train_loss, _ = self.sess.run([self.loss, self.optimizer_loss],
                                              feed_dict={self.features: X_batch, self.labels: y_batch})
                epoch_loss_list.append(train_loss)
            epoch_loss = np.mean(epoch_loss_list)[0]
            print('>>> Epoch %d: loss %f' % (epoch, epoch_loss))
            train_loss_list.append(epoch_loss)
            if epoch % 1000 == 0:
                self.saver.save(self.sess, self.save_path, global_step=epoch)

    def predict(self, X):
        test_X = copy.deepcopy(X)
        pred = self.sess.run(self.predictions, feed_dict={self.features: test_X})
        pred = tf.cast(tf.greater(pred, 0.5), tf.int32)
        return pred

    def evaluate(self, X, y):
        pass
