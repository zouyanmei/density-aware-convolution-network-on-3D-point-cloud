import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util

def se_net(input_x, ratio, layer_name, is_training, bn_decay):
    with tf.name_scope(layer_name) :
        out_dim = input_x.get_shape()[-1].value
        squeeze = global_avg_pool(input_x)
        squeeze = tf.reshape(squeeze, [-1,1,out_dim])
        excitation = tf_util.conv1d(squeeze, out_dim//ratio, 1, padding='SAME', bn=True, is_training=is_training, scope=layer_name+'fc1', bn_decay=bn_decay)
        excitation = tf.nn.relu(excitation)
        excitation = tf_util.conv1d(excitation, out_dim, 1, padding='SAME', bn=True, is_training=is_training, scope=layer_name+'fc2', bn_decay=bn_decay)
        excitation = tf.nn.sigmoid(excitation)
        excitation = tf.reshape(excitation, [-1,1,1,out_dim])
        scale = input_x * excitation
        return scale

def se_net_res(input_x, ratio, layer_name, is_training, bn_decay):
    with tf.name_scope(layer_name) :
        out_dim = input_x.get_shape()[-1].value
        squeeze = global_avg_pool(input_x)
        squeeze = tf.reshape(squeeze, [-1,1,out_dim])
        excitation = tf_util.conv1d(squeeze, out_dim//ratio, 1, padding='SAME', bn=True, is_training=is_training, scope=layer_name+'fc1', bn_decay=bn_decay)
        excitation = tf.nn.relu(excitation)
        excitation = tf_util.conv1d(excitation, out_dim, 1, padding='SAME', bn=True, is_training=is_training, scope=layer_name+'fc2', bn_decay=bn_decay)
        excitation = tf.nn.sigmoid(excitation)
        excitation = tf.reshape(excitation, [-1,1,1,out_dim])
        scale = input_x * excitation
        scale = scale + input_x
        return scale

