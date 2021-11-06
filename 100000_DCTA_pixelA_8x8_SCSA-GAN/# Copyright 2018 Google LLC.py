# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np
import ops

def conv1x1(input_, output_dim,
            init=tf.contrib.layers.xavier_initializer(), name='conv1x1'):
  k_h = 1
  k_w = 1
  d_h = 1
  d_w = 1
  with tf.variable_scope(name):
    w = tf.get_variable(
        'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=init)
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
    return conv

def sn_conv1x1(input_, output_dim, update_collection,
              init=tf.contrib.layers.xavier_initializer(), name='sn_conv1x1'):
  with tf.variable_scope(name):
    k_h = 1
    k_w = 1
    d_h = 1
    d_w = 1
    w = tf.get_variable(
        'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=init)
    w_bar = ops.spectral_normed_weight(w, num_iters=1, update_collection=update_collection)

    conv = tf.nn.conv2d(input_, w_bar, strides=[1, d_h, d_w, 1], padding='SAME')
    return conv

def sn_non_local_block_sim(x, update_collection, name, init=tf.contrib.layers.xavier_initializer()):
  with tf.variable_scope(name):
    batch_size, h, w, num_channels = x.get_shape().as_list()
    location_num = h * w
    downsampled_num = location_num // 4

    # theta path
    # theta = sn_conv1x1(x, num_channels // 8, update_collection, init, 'sn_conv_theta')
    # theta = tf.reshape(
    #     theta, [batch_size, location_num, num_channels // 8])
    print("x=",x)                                      # x_shape=(64, 32, 32, 256)
    piexl_data = tf.transpose(x, [0, 3, 1, 2])         # shape=(64, 256, 32, 32)
    print("piexl_data=",piexl_data)

    piexl_zero_zero = piexl_data[:,:,::2,::2]          # shape=(64, 256, 16, 16)
    print("piexl_zero_zero=",piexl_zero_zero)          
    piexl_zero_one  = piexl_data[:,:,::2,1::2]         # shape=(64, 256, 16, 16)
    print("piexl_zero_one=",piexl_zero_one)
    piexl_one_zero  = piexl_data[:,:,1::2,::2]         # shape=(64, 256, 16, 16)
    print("piexl_one_zero=",piexl_one_zero)
    piexl_one_one  = piexl_data[:,:,1::2,1::2]         # shape=(64, 256, 16, 16)
    print("piexl_one_one=",piexl_one_one)

    dct_zero_zero = ((piexl_zero_zero + piexl_one_zero) + (piexl_zero_one + piexl_one_one))*2     # shape=(64, 256, 16, 16)
    print("dct_zero_zero=",dct_zero_zero)
    dct_zero_one  = ((piexl_zero_zero + piexl_one_zero) - (piexl_zero_one + piexl_one_one))*2     # shape=(64, 256, 16, 16)
    print("dct_zero_one=",dct_zero_one)
    dct_one_zero  = ((piexl_zero_zero - piexl_one_zero) + (piexl_zero_one - piexl_one_one))*2     # shape=(64, 256, 16, 16)
    print("dct_one_zero=",dct_one_zero)
    dct_one_one   = ((piexl_zero_zero - piexl_one_zero) - (piexl_zero_one - piexl_one_one))*2     # shape=(64, 256, 16, 16)
    print("dct_one_one=",dct_one_one)
    # theta = tf.concat([dct_zero_zero, dct_zero_one, dct_one_zero, dct_one_one], axis=1)           # shape=(64, 1024, 16, 16)
    # print("theta=",theta)
    # theta = tf.transpose(theta, [0, 2, 3, 1])                                                     # shape=(64, 16, 16, 1024)
    # print("theta_transpose=",theta)
    # print("location_num=",location_num)   # 1024
    # print("num_channels=",num_channels)   # 256



    attn = tf.matmul(dct_zero_zero, dct_zero_one, transpose_b=True)
    print("attn_1=",attn)
    attn = tf.matmul(attn, dct_one_zero, transpose_b=True)
    print("attn_2=",attn)
    attn = tf.matmul(attn, dct_one_one, transpose_b=True)
    print("attn_3=",attn)
    attn = tf.transpose(attn, [0, 2, 3, 1])
    print("attn=",attn)


    # phi path
    # phi = sn_conv1x1(x, num_channels // 8, update_collection, init, 'sn_conv_phi')
    # phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2)
    # phi = tf.reshape(
    #     phi, [batch_size, downsampled_num, num_channels // 8])
    
    # phi = theta
    
    # attn = tf.matmul(theta, phi, transpose_b=True)
    # attn = tf.nn.softmax(attn)
    # print(tf.reduce_sum(attn, axis=-1))
    # attn = tf.matmul(theta, phi, transpose_b=True)                                               # shape=(64, 16, 16, 16)
    # print("attn_matmul=",attn)   # (64,16,16,16)
    attn = tf.nn.softmax(attn)                                                                   # shape=(64, 16, 16, 16)
    print("attn_softmax=",attn)

    # g path
    # g = sn_conv1x1(x, num_channels // 2, update_collection, init, 'sn_conv_g')
    # g = tf.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2)
    # g = tf.reshape(
    #   g, [batch_size, downsampled_num, num_channels // 2])
    print("x=",x)
    g = sn_conv1x1(x, num_channels, update_collection, init, 'sn_conv_g')                    # shape=(64, 32, 32, 128)
    print("g_sn_conv1x1=",g)
    g = tf.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2)                            # shape=(64, 16, 16, 128)
    print("g_max_pooling=",g)

    attn_g = tf.matmul(attn, g, transpose_b=True)                                                                   # shape=(64, 16, 16, 128)
    # attn_g = tf.reshape(attn_g, [batch_size, h, w, num_channels // 2])
    print("attn_g_matmul=",attn_g)   # (64,16,16,128)
    print("h=",h)                          #32
    print("w=",w)                          #32
    print("num_channels=",num_channels)    # 256

    sigma = tf.get_variable(
        'sigma_ratio', [], initializer=tf.constant_initializer(0.0))                              
    print("sigma=",sigma)
    attn_g = sn_conv1x1(attn_g, num_channels, update_collection, init, 'sn_conv_attn')            # shape=(64, 16, 16, 256)
    print("attn_g_sn_conv1x1",attn_g)    
    attn_g = ops.deconv2d(attn_g, [batch_size, 32, 32, num_channels],
             k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
             name='deconv2d', init_bias=0.)
    print("attn_g_deconv2d=",attn_g)                                                              # 
    print("x=",x)                                                                                 # shape=(64, 32, 32, 256)
    return x + sigma * attn_g