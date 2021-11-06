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

    print("x=",x)   

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

    
    #b00********************************************************************


    x_zero_zero = tf.transpose(dct_zero_zero, [0, 2, 3, 1])                                         # shape=(64, 16, 16, 256)
    print("x_zero_zero=",x_zero_zero)

    # theta path
    print("x=",x)
    theta_00 = sn_conv1x1(x_zero_zero, num_channels //8 , update_collection, init, 'sn_conv_theta')
    print('theta_00_sn_conv', theta_00)
    #print(x.get_shape())
    # theta = tf.reshape(                                                                     # shape=(64, 256, 32)
    #     theta, [batch_size, location_num //4, num_channels // 8])
    # print("theta_rehape=",theta)


    # phi path
    # phi_00 = sn_conv1x1(x_zero_zero, num_channels //8 , update_collection, init, 'sn_conv_phi')     # shape=(64, 16, 16, 256)
    # print("phi_00_sn_conv=",phi_00)
    # phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2)
    # print("phi_max_pool=",phi)
    # phi = tf.reshape(
    #     phi, [batch_size, downsampled_num, num_channels])
    # print("phi_rehape=",phi)


    # attn_00 = tf.matmul(theta_00, phi_00, transpose_b=True)
    # print("attn_00_matmul=",attn_00)
    # attn_00 = tf.nn.softmax(attn_00)                                                         # shape=(64, 16, 16, 16)
    # print(tf.reduce_sum(attn_00, axis=-1))
    # print("attn_00_softmax=",attn_00)

    

    ############################################################################
    #b01********************************************************************


    x_zero_one = tf.transpose(dct_zero_one, [0, 2, 3, 1])                                         # shape=(64, 16, 16, 256)
    print("x_zero_one=",x_zero_one)

    # theta path
    print("x=",x)
    theta_01 = sn_conv1x1(x_zero_one, num_channels //8 , update_collection, init, 'sn_conv_theta')
    print('theta_01_sn_conv', theta_01)
    #print(x.get_shape())
    # theta = tf.reshape(                                                                     # shape=(64, 256, 32)
    #     theta, [batch_size, location_num //4, num_channels // 8])
    # print("theta_rehape=",theta)


    # phi path
    # phi_01 = sn_conv1x1(x_zero_one, num_channels //8 , update_collection, init, 'sn_conv_phi')     # shape=(64, 16, 16, 256)
    # print("phi_01_sn_conv=",phi_01)
    # phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2)
    # print("phi_max_pool=",phi)
    # phi = tf.reshape(
    #     phi, [batch_size, downsampled_num, num_channels])
    # print("phi_rehape=",phi)


    attn_01 = tf.matmul(theta_00, theta_01, transpose_b=True)
    print("attn_01_matmul=",attn_01)
    attn_01 = tf.nn.softmax(attn_01)                                                         # shape=(64, 16, 16, 16)
    # print(tf.reduce_sum(attn_01, axis=-1))
    # print("attn_01_softmax=",attn_01)


    #b10********************************************************************


    x_one_zero = tf.transpose(dct_one_zero, [0, 2, 3, 1])                                         # shape=(64, 16, 16, 256)
    print("x_one_zero=",x_one_zero)

    # theta path
    print("x=",x)
    theta_10 = sn_conv1x1(x_one_zero, num_channels //8 , update_collection, init, 'sn_conv_theta')
    print('theta_10_sn_conv', theta_10)
    #print(x.get_shape())
    # theta = tf.reshape(                                                                     # shape=(64, 256, 32)
    #     theta, [batch_size, location_num //4, num_channels // 8])
    # print("theta_rehape=",theta)


    # phi path
    # phi_10 = sn_conv1x1(x_one_zero, num_channels //8 , update_collection, init, 'sn_conv_phi')     # shape=(64, 16, 16, 256)
    # print("phi_10_sn_conv=",phi_10)
    # phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2)
    # print("phi_max_pool=",phi)
    # phi = tf.reshape(
    #     phi, [batch_size, downsampled_num, num_channels])
    # print("phi_rehape=",phi)


    # attn_10 = tf.matmul(theta_10, phi_10, transpose_b=True)
    # print("attn_10_matmul=",attn_10)
    # attn_10 = tf.nn.softmax(attn_10)                                                         # shape=(64, 16, 16, 16)
    # print(tf.reduce_sum(attn_10, axis=-1))
    # print("attn_10_softmax=",attn_10)



    #b11********************************************************************


    x_one_one = tf.transpose(dct_one_one, [0, 2, 3, 1])                                         # shape=(64, 16, 16, 256)
    print("x_one_one=",x_one_one)

    # theta path
    print("x=",x)
    theta_11 = sn_conv1x1(x_one_one, num_channels //8 , update_collection, init, 'sn_conv_theta')
    print('theta_11_sn_conv', theta_11)
    #print(x.get_shape())
    # theta = tf.reshape(                                                                     # shape=(64, 256, 32)
    #     theta, [batch_size, location_num //4, num_channels // 8])
    # print("theta_rehape=",theta)


    # phi path
    # phi_11 = sn_conv1x1(x_one_one, num_channels //8 , update_collection, init, 'sn_conv_phi')     # shape=(64, 16, 16, 256)
    # print("phi_11_sn_conv=",phi_11)
    # phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2)
    # print("phi_max_pool=",phi)
    # phi = tf.reshape(
    #     phi, [batch_size, downsampled_num, num_channels])
    # print("phi_rehape=",phi)


    attn_11 = tf.matmul(theta_10, theta_11, transpose_b=True)
    print("attn_11_matmul=",attn_11)
    attn_11 = tf.nn.softmax(attn_11)                                                         # shape=(64, 16, 16, 16)
    # print(tf.reduce_sum(attn_11, axis=-1))
    # print("attn_11_softmax=",attn_11)



    ##################################
    # attn1=tf.matmul(attn_00, attn_01, transpose_b=True)
    # attn2=tf.matmul(attn_10, attn_11, transpose_b=True)
    # attn_dct=tf.matmul(attn1, attn2, transpose_b=True)
    attn_dct=attn_01+attn_11
    # attn_dct = tf.nn.softmax(attn_dct)
    print("attn_dct=",attn_dct)

    ##################################
    # pixel attention

    # theta path
    print("x=",x)
    theta = sn_conv1x1(x, num_channels //8 , update_collection, init, 'sn_conv_theta')
    print('theta_sn_conv', theta)
    #print(x.get_shape())
    # theta = tf.reshape(                                                                     # shape=(64, 256, 32)
    #     theta, [batch_size, location_num //4, num_channels // 8])
    # print("theta_rehape=",theta)


    # phi path
    phi = sn_conv1x1(x, num_channels //8 , update_collection, init, 'sn_conv_phi')     # shape=(64, 16, 16, 256)
    print("phi_sn_conv=",phi)
    # phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2)
    # print("phi_max_pool=",phi)
    # phi = tf.reshape(
    #     phi, [batch_size, downsampled_num, num_channels])
    # print("phi_rehape=",phi)


    attn_pixel = tf.matmul(theta, phi, transpose_b=True)
    print("attn_pixel_matmul=",attn_pixel)
    attn_pixel = tf.nn.softmax(attn_pixel)                        # shape=(64, 32, 32, 32)            
    print(tf.reduce_sum(attn_pixel, axis=-1))
    print("attn_pixel=",attn_pixel)
    attn_pixel = tf.layers.max_pooling2d(inputs=attn_pixel, pool_size=[2, 2], strides=2) 
    print("attn_pixel_max_pool=",attn_pixel)


    ##################################
    attn = tf.matmul(attn_dct, attn_pixel)          # shape=(64, 16, 16, 32)
    print("attn_matmul=",attn)
    ##################################

    # g path
    channels=attn.get_shape().as_list()[-1]
    g = sn_conv1x1(x, channels, update_collection, init, 'sn_conv_g')  # shape=(64, 32, 32, 128)
    print("g_sn_conv=",g)
    g = tf.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2)          # shape=(64, 16, 16, 128)
    print("g_max_pool=",g)
    # g = tf.reshape(                                                             
    #   g, [batch_size, downsampled_num, num_channels // 2])
    # print("g_reshape=",g)

    
    attn_g = tf.matmul(attn, g, transpose_b=True)                                                                     
    print("attn_g_matmul=",attn_g)
    # attn_g = tf.reshape(attn_g, [batch_size, h//2, w//2, num_channels // 2])     
    # print("attn_g_reshape=",attn_g)


    
    attn_g = sn_conv1x1(attn_g, num_channels, update_collection, init, 'sn_conv_attn')
    print("attn_g_sn_conv1x1",attn_g)
    attn_g = ops.deconv2d(attn_g, [batch_size, h, w, num_channels],         # num_channels
             k_h=2, k_w=2, d_h=2, d_w=2, stddev=0.02,
             name='attn_g_deconv2d', init_bias=0.)
    print("attn_g_deconv2d=",attn_g)                                                            
    print("x=",x) 


    sigma = tf.get_variable(
        'sigma_ratio', [], initializer=tf.constant_initializer(0.0))
    print("sigma=",sigma)
    return x + sigma * attn_g


  


    
