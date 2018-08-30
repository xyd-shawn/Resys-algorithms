"""
Simply implement Factorization Machines
paper: https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
"""

import os
import sys
import copy

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.contrib.layers import l2_regularizer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class FM(object):
    def __init__(self, feature_type='mixed', task_type='pred_CTR', n_features=10, feature_size=None, use_pretrain=False, **config):
        self.feature_type = feature_type
        self.task_type = task_type
        self.n_features = n_features
        self.feature_size = feature_size
        if self.feature_type in ['discrete', 'categorical']:
            if not self.feature_size:
                print('Error, feature_size should be the maximum value of features!')
                sys.exit(0)
        else:
            if not n_features == feature_size:
                print('Error, feature_size should equal to n_features!')
                sys.exit(0)
        self.use_pretrain = use_pretrain
        self.factor_dim = config.get('factor_dim', 16)
        self.norm_coef = config.get('norm_coef', 0.0)
        self.batch_size = config.get('batch_size', 32)
        self.optimizer_type = config.get('optimizer_type', 'sgd')
        self.lr = config.get('learning_rate', 0.01)
        self.momentum = config.get('momentum', 0.9)
        self.random_seed = config.get('random_seed', 1)
        self.save_path = config.get('save_path', '../logs/FM/')
        self.fm_save_path = config.get('fm_save_path', '../tmp/FM/')
        self.fm_load_path = config.get('fm_load_path', '../tmp/FM/')
        self._build_model()

    def _build_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)        # set random seed
            # input data
            self.labels = tf.placeholder(tf.float32, shape=[None, 1], name='labels')

            # compute the linear part and the interaction part of fm respectively
            if self.use_pretrain:        # use pretrain weights
                fm_bias, fm_weights, fm_interaction = self.load_weights_from_npy()
                self.b = tf.Variable(fm_bias, name='bias')
                self.w = tf.Variable(fm_weights, name='feature_weights')
                self.v = tf.Variable(fm_interaction, name='feature_interaction')
            else:
                self.b = tf.Variable(tf.constant(0.0), name='bias')
                self.w = tf.Variable(tf.truncated_normal(shape=[self.feature_size, 1], mean=0.0, stddev=0.1),
                                     name='feature_weights')
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
                self.loss_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.fm))
                self.correct = tf.equal(tf.cast(tf.greater(self.predictions, 0.5), tf.float32), self.labels)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))
            else:
                self.predictions = self.fm
                self.loss_f = tf.nn.l2_loss(tf.subtract(self.predictions, self.labels))
            self.loss = self.loss_f + l2_regularizer(self.norm_coef)(self.w) + l2_regularizer(self.norm_coef)(self.v)

            # build optimizer
            if self.optimizer_type.lower() == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            elif self.optimizer_type.lower() == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            elif self.optimizer_type.lower() == 'rmsprop':
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, momentum=self.momentum)
            elif self.optimizer_type.lower() == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.momentum)

            # create session and init
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=tf_config)
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def load_pretrain_weights(self):
        self.saver.restore(self.sess, self.save_path)

    def load_weights_from_npy(self):
        fm_bias = np.load(self.fm_load_path + 'fm_bias.npy')
        fm_weights = np.load(self.fm_load_path + 'fm_weights.npy')
        fm_interaction = np.load(self.fm_load_path + 'fm_interaction.npy')
        return fm_bias, fm_weights, fm_interaction

    def save_weights_to_npy(self):
        fm_bias, fm_weights, fm_interaction = self.sess.run([self.b, self.w, self.v])
        np.save(self.fm_save_path + 'fm_bias.npy', fm_bias)
        np.save(self.fm_save_path + 'fm_weights.npy', fm_weights)
        np.save(self.fm_save_path + 'fm_interaction.npy', fm_interaction)
        print('finish saving model weights')

    def fit(self, X, y, n_epochs, need_save_weights=False):
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
                train_loss, _ = self.sess.run([self.loss, self.optimizer.minimize(self.loss)],
                                              feed_dict={self.features: X_batch, self.labels: y_batch})
                epoch_loss_list.append(train_loss)
            epoch_loss = np.mean(epoch_loss_list)[0]
            print('>>> Epoch %d: loss %f' % (epoch, epoch_loss))
            train_loss_list.append(epoch_loss)
            if epoch % 1000 == 0:
                self.saver.save(self.sess, self.save_path, global_step=epoch)
        if need_save_weights:
            self.save_weights_to_npy()

    def predict(self, X):
        test_X = copy.deepcopy(X)
        pred = self.sess.run(self.predictions, feed_dict={self.features: test_X})
        if self.task_type == 'pred_CTR':
            pred = tf.cast(tf.greater(pred, 0.5), tf.int32)
        return pred

    def evaluate(self, X, y):
        pass
