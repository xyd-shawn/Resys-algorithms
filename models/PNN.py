"""
Simply implement PNN for CTR prediction
paper: https://arxiv.org/pdf/1611.00144.pdf
"""

import os
import copy

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.contrib.layers import l2_regularizer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class PNN(object):
    def __init__(self, n_features_list, mode='inner', **config):
        self.n_features_list = n_features_list
        self.n_features_total = sum(self.n_features_list)
        self.n_fields = len(self.n_features_list)
        self.mode = mode
        self.embedding_dim = config.get('embedding_dim', 16)
        self.hidden_units = config.get('hidden_units', [200, 200, 200])
        self.batch_size = config.get('batch_size', 32)
        self.lr = config.get('learning_rate', 0.01)
        self.momentum = config.get('momentum', 0.9)
        self.norm_coef = config.get('norm_coef', 0.0)
        self.optimizer_type = config.get('optimizer_type', 'adam')
        self.save_path = config.get('save_path', '../logs/PNN/')
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
            with tf.variable_scope('embedding_layer'):
                self.W0 = dict()
                self.embedding_res = dict()
                for i in range(self.n_fields):
                    self.W0[i] = tf.Variable(tf.truncated_normal(shape=[self.n_features_list[i], self.embedding_dim], mean=0.0, stddev=0.1),
                                             name='embedding_%d' % i)
                    self.embedding_res[i] = tf.matmul(self.field_values[i], self.W0[i])

            # the second layer, which implements inner product or outer product
            with tf.variable_scope('product_layer'):
                self.b1 = tf.Variable(tf.constant(0.0, shape=[1, self.hidden_units[0]]), name='bias_1')
                embedding_concat = tf.concat([self.embedding_res[i] for i in range(self.n_fields)], axis=1)
                self.Wz = tf.Variable(tf.truncated_normal(shape=[self.embedding_dim * self.n_fields, self.hidden_units[0]], mean=0.0, stddev=0.1),
                                      name='W_z')
                self.lz = tf.matmul(embedding_concat, self.Wz)
                self.lp = tf.ones_like(self.lz)
                if self.mode == 'inner':
                    self.Wp = tf.Variable(tf.truncated_normal(shape=[self.n_fields, self.hidden_units[0]], mean=0.0, stddev=0.1),
                                          name='W_p')
                    self.lp_list = []
                    for i in range(self.hidden_units[0]):
                        f_sum = tf.zeros_like(self.embedding_res[0])
                        for j in range(self.n_fields):
                            f_sum += (self.Wp[j, i] * self.embedding_res[j])
                        self.lp_list.append(tf.square(tf.norm(f_sum, axis=1)))
                    self.lp = tf.stack(self.lp_list, axis=1)
                else:
                    f_sum = tf.add_n([self.embedding_res[i] for i in range(self.n_fields)])
                    f_sum_transpose = tf.expand_dims(f_sum, axis=1)
                    f_sum = tf.expand_dims(f_sum, axis=-1)
                    self.p = tf.reshape(tf.matmul(f_sum, f_sum_transpose), shape=[-1, self.embedding_dim ** 2])
                    self.Wp = tf.Variable(tf.truncated_normal(shape=[self.embedding_dim ** 2, self.hidden_units[0]], mean=0.0, stddev=0.1),
                                          name='W_p')
                    self.lp = tf.matmul(self.p, self.Wp)
                self.layer1 = tf.nn.relu(self.lz + self.lp + self.b1)
                self.layer1 = tf.nn.dropout(self.layer1, keep_prob=self.dropout_keep)

            # the third layer, which is the first fully connected layer
            with tf.variable_scope('fc_layer_1'):
                self.b2 = tf.Variable(tf.constant(0.0, shape=[1, self.hidden_units[1]]), name='bias_2')
                self.W2 = tf.Variable(tf.truncated_normal(shape=[self.hidden_units[0], self.hidden_units[1]], mean=0.0, stddev=0.1),
                                      name='W_2')
                self.layer2 = tf.nn.relu(tf.matmul(self.layer1, self.W2) + self.b2)

            # the forth layer, which is the second fully connected layer
            with tf.variable_scope('fc_layer_2'):
                self.b3 = tf.Variable(tf.constant(0.0, shape=[1, self.hidden_units[2]]), name='bias_3')
                self.W3 = tf.Variable(tf.truncated_normal(shape=[self.hidden_units[1], self.hidden_units[2]], mean=0.0, stddev=0.1),
                                      name='W_3')
                self.layer3 = tf.nn.relu(tf.matmul(self.layer2, self.W3) + self.b3)

            # the last layer, which is the out layer
            with tf.variable_scope('output_layer'):
                self.b4 = tf.Variable(tf.constant(0.0, shape=[1, 1]), name='bias_4')
                self.W4 = tf.Variable(tf.truncated_normal(shape=[self.hidden_units[2], 1], mean=0.0, stddev=0.1),
                                      name='W_4')
                self.pnn = tf.matmul(self.layer3, self.W4) + self.b4

            # compute loss and accuracy
            with tf.variable_scope('loss'):
                self.predictions = tf.sigmoid(self.pnn)
                self.loss_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.pnn))
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
