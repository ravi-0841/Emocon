#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:19:14 2019

@author: ravi
"""

from __future__ import division, print_function, absolute_import
import tensorflow as tf
from sklearn.externals import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def predict_pitch(input_data, context, emo):
    input_feats             = input_data[:,:129]
    input_pitch             = input_data[:,129:]
    scaler                  = joblib.load('./data/'+emo+'-DNN/'+emo+'-pitch-scaler-modif5.pkl')
    input_feats             = scaler.transform(input_feats)
    
    # Network Parameters
    tf.reset_default_graph()
    n_hidden_1              = 256#256
    n_hidden_2              = 128#128
    n_hidden_3              = 128#128
    n_hidden_4              = 64#64
    
    num_input               = input_feats.shape[1]
    num_pitch               = input_pitch.shape[1]
    num_output              = 1
    
    X_feats                 = tf.placeholder("float", [None, num_input])
    X_pitch                 = tf.placeholder("float", [None, num_pitch])
    keep_prob               = tf.placeholder(tf.float32)
    global_step             = tf.Variable(0, trainable=False)
    tf_variance             = tf.Variable(1.0, trainable=False)
    training_mode           = tf.placeholder(tf.bool)
    
    weights = {
        'h1' : tf.Variable(tf.random_normal([num_input, n_hidden_1])),
        'h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3' : tf.Variable(tf.random_normal([n_hidden_2+num_pitch, n_hidden_3])),
        'h4' : tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
        'out': tf.Variable(tf.random_normal([n_hidden_4+3, num_output]))
    }
    
    biases = {
        'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
        'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
        'b3' : tf.Variable(tf.random_normal([n_hidden_3])),
        'b4' : tf.Variable(tf.random_normal([n_hidden_4])),
        'out': tf.Variable(tf.random_normal([num_output]))
    }
    
    def neural_net(x_feats, x_pitch, dropout, training):
        layer_1     = tf.add(tf.matmul(x_feats, weights['h1']), biases['b1'])
        layer_1     = tf.nn.leaky_relu(layer_1)
        layer_1     = tf.layers.batch_normalization(layer_1, momentum=0.9, training=training)
        layer_1     = tf.nn.dropout(layer_1, dropout)
    
        layer_2     = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2     = tf.nn.leaky_relu(layer_2)
        layer_2     = tf.layers.batch_normalization(layer_2, momentum=0.9, training=training)
        layer_2     = tf.nn.dropout(layer_2, dropout)
        layer_2     = tf.concat([layer_2, x_pitch], 1)
    
        layer_3     = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        layer_3     = tf.nn.leaky_relu(layer_3)
        layer_3     = tf.layers.batch_normalization(layer_3, momentum=0.9, training=training)
        layer_3     = tf.nn.dropout(layer_3, dropout)
        
        layer_4     = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
        layer_4     = tf.nn.leaky_relu(layer_4)
        layer_4     = tf.layers.batch_normalization(layer_4, momentum=0.9, training=training)
        layer_4     = tf.nn.dropout(layer_4, dropout)
        layer_4     = tf.concat([layer_4, x_pitch[:,context-1:context+2]], 1)
    
        out_layer   = tf.add(tf.matmul(layer_4, weights['out']), \
                                biases['out'])
        return tf.nn.relu(out_layer)
    
    regressor       = neural_net(X_feats, X_pitch, keep_prob, training_mode)

    saver           = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './data/'+emo+'-DNN/'+emo+'-pitch-mle-modif5.ckpt')
        predictions = sess.run(regressor, feed_dict={X_feats:input_feats, \
                                                     X_pitch:input_pitch, \
                                                     keep_prob:1.0, \
                                                     training_mode:False})
    return predictions