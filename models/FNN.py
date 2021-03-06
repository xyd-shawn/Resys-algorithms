"""
Simply implement FNN for CTR prediction
paper: https://arxiv.org/pdf/1601.02376.pdf
"""

import os
import sys
import copy

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.contrib.layers import l2_regularizer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class FNN(object):
    def __init__(self, n_features_list, **config):
        self.n_features_list = n_features_list
        self.n_features_total = sum(n_features_list)
        self.n_fields = len(self.n_features_list)
        self.embedding_dim = config.get('embedding_dim', 8)
        self.batch_size = config.get('batch_size', 32)
        self.hidden_units = config.get('hidden_units', [200, 200])
        self.norm_coef = config.get('norm_coef', 0.0)
        self.optimizer_type = config.get('optimizer_type', 'sgd')
        self.lr = config.get('learning_rate', 0.01)
        self.momentum = config.get('momentum', 0.9)
        self.save_path = config.get('save_path', '../logs/FNN/')
        self.fm_params_path = config.get('fm_params_path', '../tmp/FM/')
        self.random_seed = config.get('random_seed', 1)
        self.keep_prob = config.get('keep_prob', 0.5)
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
            pretrain_fm_weights = self._process_fm_params()
            with tf.variable_scope('embedding_layer'):
                self.b0 = tf.Variable(tf.constant(0.0), name='bias_0')
                self.W0 = dict()
                l1 = self.b0 * tf.ones_like(self.labels)
                for i in range(self.n_fields):
                    self.W0[i] = tf.Variable(pretrain_fm_weights[i], name='embedding_%d' % i)
                    l1 = tf.concat([l1, tf.sparse_tensor_dense_matmul(self.field_values[i], self.W0[i])], axis=1)
                self.layer1 = l1        # [None, n_fields * embedding_dim + 1]
                self.layer1 = tf.nn.dropout(self.layer1, keep_prob=self.dropout_keep)

            # the second layer, which is the first fully connected layer
            with tf.variable_scope('fc_layer_1'):
                self.b1 = tf.Variable(tf.constant(0.0, shape=[1, self.hidden_units[0]]), name='bias_1')
                self.W1 = tf.Variable(tf.truncated_normal(shape=[self.n_fields * self.embedding_dim + 1, self.hidden_units[0]], mean=0.0, stddev=0.1),
                                      name='W_1')
                self.layer2 = tf.nn.relu(tf.matmul(self.layer1, self.W1) + self.b1)

            # the third layer, which is the second fully connected layer
            with tf.variable_scope('fc_layer_2'):
                self.b2 = tf.Variable(tf.constant(0.0, shape=[1, self.hidden_units[1]]), name='bias_2')
                self.W2 = tf.Variable(tf.truncated_normal(shape=[self.hidden_units[0], self.hidden_units[1]], mean=0.0, stddev=0.1),
                                      name='W_2')
                self.layer3 = tf.nn.relu(tf.matmul(self.layer2, self.W2) + self.b2)

            # the last layer, which is the output layer
            with tf.variable_scope('out_layer'):
                self.b3 = tf.Variable(tf.constant(0.0, shape=[1, 1]), name='bias_3')
                self.W3 = tf.Variable(tf.truncated_normal(shape=[self.hidden_units[1], 1], mean=0.0, stddev=0.1),
                                      name='W_3')
                self.fnn = tf.matmul(self.layer3, self.W3) + self.b3

            # compute loss and accuracy
            with tf.variable_scope('loss'):
                self.predictions = tf.sigmoid(self.fnn)
                self.loss_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.fnn))
                total_vars = tf.trainable_variables()
                self.loss = self.loss_f + tf.add_n([l2_regularizer(self.norm_coef)(v) for v in total_vars])
                self.correct = tf.equal(tf.cast(tf.greater(self.predictions, 0.5), tf.float32), self.labels)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

            # build optimizer
            with tf.variable_scope('optimizer'):
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

    def _process_fm_params(self):
        fm_weights = np.load(self.fm_params_path + 'fm_weights.npy')
        fm_interaction = np.load(self.fm_params_path + 'fm_interaction.npy')
        factor_dim = fm_interaction.shape[1]
        if not (factor_dim + 1) == self.embedding_dim:
            print('Error, the embedding dim should equal to factor dim plus one!')
            sys.exit(0)
        n_features_list = copy.deepcopy(self.n_features_list)
        n_features_list.insert(0, 0)
        cum_features_list = np.cumsum(n_features_list, dtype=int)
        pretrain_fm_weights = dict()
        for i in range(self.n_fields):
            pretrain_fm_weights[i] = np.zeros((n_features_list[i], self.embedding_dim))
            pretrain_fm_weights[i][:, 0] = fm_weights[cum_features_list[i]:cum_features_list[i + 1], :]
            pretrain_fm_weights[i][:, 1:] = fm_interaction[cum_features_list[i]:cum_features_list[i + 1], :]
        return pretrain_fm_weights

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
                                              feed_dict={self.features: X_batch,
                                                         self.labels: y_batch,
                                                         self.dropout_keep: self.keep_prob})
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
