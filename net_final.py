# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 17:02:40 2018

@author: gxd
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 16:40:33 2018

@author: gxd
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflearn
from utils import Conv2D, DenseBlock, Transformer, BatchNorm

#%% 
def warp_img(batch_size, imga, imgb, reuse, scope='MCNet'):
#imga-warp target
#imgb-img need warp
    
    n, h, w, c = imga.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], activation_fn=tflearn.activations.prelu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            weights_regularizer=slim.l2_regularizer(0.05), #weight_regularization-L2_loss
                            biases_initializer=tf.constant_initializer(0.0)), \
             slim.arg_scope([slim.conv2d_transpose], activation_fn=tflearn.activations.prelu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            weights_regularizer=slim.l2_regularizer(0.05), #weight_regularization-L2_loss
                            biases_initializer=tf.constant_initializer(0.0)):
            # Large down-scaling motion estimation
            with tf.variable_scope('Large', reuse=reuse):
                inputs = tf.concat([imga, imgb], 3, name='cat1')
                a1 = slim.conv2d(inputs, 24, [7, 7], stride=2, scope='a1')
                a2 = slim.conv2d(a1, 24, [5, 5], scope='a2')
                a3 = slim.conv2d(a2, 24, [7, 7], stride=2, scope='a3')
                a4 = slim.conv2d(a3, 24, [5, 5], scope='a4')               
                a5 = slim.conv2d(a4, 32, [3, 3], activation_fn=tf.nn.tanh, scope='a5')
                a5_hr = tf.reshape(a5, [n, int(h / 4), int(w / 4), 2, 4, 4])
                a5_hr = tf.transpose(a5_hr, [0, 1, 4, 2, 5, 3])
                a5_hr = tf.reshape(a5_hr, [n, h, w, 2])
                imgb_warp1 = Transformer(batch_size, c, a5_hr, imgb, [h, w])
            
            
            # X4 down-scaling motion estimation
            with tf.variable_scope('X4', reuse=reuse):
                a5_pack = tf.concat([inputs, a5_hr, imgb_warp1], 3, name='cat2')
                b1 = slim.conv2d(a5_pack, 24, [5, 5], stride=2, scope='b1')
                b2 = slim.conv2d(b1, 24, [3, 3], scope='b2')
                b3 = slim.conv2d(b2, 24, [5, 5], stride=2, scope='b3')
                b4 = slim.conv2d(b3, 24, [3, 3], scope='b4')
                b5 = slim.conv2d(b4, 32, [3, 3], activation_fn=tf.nn.tanh, scope='b5')
                b5_hr = tf.reshape(b5, [n, int(h / 4), int(w / 4), 2, 4, 4])
                b5_hr = tf.transpose(b5_hr, [0, 1, 4, 2, 5, 3])
                b5_hr = tf.reshape(b5_hr, [n, h, w, 2])
                uv1 = a5_hr + b5_hr
                imgb_warp2 = Transformer(batch_size, c, uv1, imgb, [h, w])
            
            
            # X2 down-scaling motion estimation
            with tf.variable_scope('X2', reuse=reuse):
                b5_pack = tf.concat([inputs, uv1, imgb_warp2], 3, name='cat3')
                c1 = slim.conv2d(b5_pack, 24, [5, 5], stride=2, scope='c1')
                c2 = slim.conv2d(c1, 24, [3, 3], scope='c2')
                c3 = slim.conv2d(c2, 24, [3, 3], scope='c3')
                c4 = slim.conv2d(c3, 24, [3, 3], scope='c4')
                c5 = slim.conv2d(c4, 8, [3, 3], activation_fn=tf.nn.tanh, scope='c5')
                c5_hr = tf.reshape(c5, [n, int(h / 2), int(w / 2), 2, 2, 2])
                c5_hr = tf.transpose(c5_hr, [0, 1, 4, 2, 5, 3])
                c5_hr = tf.reshape(c5_hr, [n, h, w, 2])
                uv2 = uv1 + c5_hr
                imgb_warp3 = Transformer(batch_size, c, uv2, imgb, [h, w])
            
    return imgb_warp3

#%%
def derain_net(is_train, input_x, num_frame, reuse = False, scope='DerainNet'): 
    growth_k = 12
    stride_hw = 1
    padding = 'SAME'
    nb_layers=6
    with tf.variable_scope(scope, reuse=reuse):
        # feature extration
        c1 = Conv2D(input_x, [7, 7, num_frame, 2*growth_k], stride_hw, padding, name=scope+'_conv1')
        
        # complex feature extraction
        c2 = DenseBlock(c1, is_train, nb_layers, 2*growth_k, growth_k, stride_hw, padding, block_name=scope+'_DenseBlock')
        
        # non-linear mapping
        c2 = BatchNorm(c2, is_train, name=scope+'_BN1')
        c2 = tf.nn.relu(c2)
        c3 = Conv2D(c2, [1, 1, growth_k, growth_k/2], stride_hw, padding, name=scope+'_conv2')
        
        
        # residual reconstruction
        c3 = BatchNorm(c3, is_train, name=scope+'_BN2')
        c3 = tf.nn.relu(c3)
        res = Conv2D(c3, [5, 5, growth_k/2 ,1], stride_hw, padding, name=scope+'_conv3')
    
    return res
        

#%%
def train_network(is_train, F0, Fm1, Fm2, Fp1, Fp2, batch_size):
#F0: current frame 'ft'
#Fm1, Fm2: forward frame 'ft-1, ft-2'
#Fp1, Fp2: backward frame  'ft+1, ft+2'
        
        # MC-subnet
        Fm1to0 = warp_img(batch_size, F0, Fm1, False)
        Fm2to0 = warp_img(batch_size, F0, Fm2, True)
        Fp1to0 = warp_img(batch_size, F0, Fp1, True)
        Fp2to0 = warp_img(batch_size, F0, Fp2, True)
        
        #F_warped = tf.concat([F0, Fm1to0, Fm2to0, Fp1to0, Fp2to0], 3)
        F_warped = tf.concat([Fm2to0, Fm1to0, F0, Fp1to0, Fp2to0], 3)
        
        tf.summary.image('Fm1to0_batch', Fm1to0)
        tf.summary.image('Fm2to0_batch', Fm2to0)
        tf.summary.image('Fp1to0_batch', Fp1to0)
        tf.summary.image('Fp2to0_batch', Fp2to0)
        
        
        # Derain-subnet 
        F_res = derain_net(is_train, F_warped, reuse=False, num_frame=5)
        tf.summary.image('F_res_batch', F_res)
        
        # residual learning        
        F_derained = tf.add(F0, F_res)
        #F_derained = tf.subtract(F0, F_res)
        tf.summary.image('F_derained_batch', F_derained)
        
        return F_derained


def eval_network(is_train, F0, Fm1, Fm2, Fp1, Fp2, batch_size):
#F0: current frame 'ft'
#Fm1, Fm2: forward frame 'ft-1, ft-2'
#Fp1, Fp2: backward frame  'ft+1, ft+2'
        
        # MC-subnet
        Fm1to0 = warp_img(batch_size, F0, Fm1, True)
        Fm2to0 = warp_img(batch_size, F0, Fm2, True)
        Fp1to0 = warp_img(batch_size, F0, Fp1, True)
        Fp2to0 = warp_img(batch_size, F0, Fp2, True)
        
        #F_warped = tf.concat([F0, Fm1to0, Fm2to0, Fp1to0, Fp2to0], 3)
        F_warped = tf.concat([Fm2to0, Fm1to0, F0, Fp1to0, Fp2to0], 3)
        
        tf.summary.image('Fm1to0_batch', Fm1to0)
        tf.summary.image('Fm2to0_batch', Fm2to0)
        tf.summary.image('Fp1to0_batch', Fp1to0)
        tf.summary.image('Fp2to0_batch', Fp2to0)
        
        
        # Derain-subnet 
        F_res = derain_net(is_train, F_warped, reuse=True, num_frame=5)
        tf.summary.image('F_res_batch', F_res)
        
        # residual learning        
        F_derained = tf.add(F0, F_res)
        tf.summary.image('F_derained_batch', F_derained)
        
        return F_derained 

def test_network(is_train, F0, Fm1, Fm2, Fp1, Fp2, batch_size):
#F0: current frame 'ft'
#Fm1, Fm2: forward frame 'ft-1, ft-2'
#Fp1, Fp2: backward frame  'ft+1, ft+2'
        
        # MC-subnet
        Fm1to0 = warp_img(batch_size, F0, Fm1, False)
        Fm2to0 = warp_img(batch_size, F0, Fm2, True)
        Fp1to0 = warp_img(batch_size, F0, Fp1, True)
        Fp2to0 = warp_img(batch_size, F0, Fp2, True)
        
        #F_warped = tf.concat([F0, Fm1to0, Fm2to0, Fp1to0, Fp2to0], 3)
        F_warped = tf.concat([Fm2to0, Fm1to0, F0, Fp1to0, Fp2to0], 3)
        
        tf.summary.image('Fm1to0_batch', Fm1to0)
        tf.summary.image('Fm2to0_batch', Fm2to0)
        tf.summary.image('Fp1to0_batch', Fp1to0)
        tf.summary.image('Fp2to0_batch', Fp2to0)
        
        
        # Derain-subnet 
        F_res = derain_net(is_train, F_warped, reuse=False, num_frame=5)
        tf.summary.image('F_res_batch', F_res)
        
        # residual learning        
        F_derained = tf.add(F0, F_res)
        #F_derained = tf.subtract(F0, F_res)
        tf.summary.image('F_derained_batch', F_derained)
        
        return F_derained


def train_or_test_network_drain_only(is_train, F0, Fm1, Fm2, Fp1, Fp2, batch_size):
#F0: current frame 'ft'
#Fm1, Fm2: forward frame 'ft-1, ft-2'
#Fp1, Fp2: backward frame  'ft+1, ft+2'
        
        # MC-subnet
        #Fm1to0 = warp_img(batch_size, F0, Fm1, False)
        #Fm2to0 = warp_img(batch_size, F0, Fm2, True)
        #Fp1to0 = warp_img(batch_size, F0, Fp1, True)
        #Fp2to0 = warp_img(batch_size, F0, Fp2, True)
        
        #F_warped = tf.concat([F0, Fm1to0, Fm2to0, Fp1to0, Fp2to0], 3)
        
        
        #tf.summary.image('Fm1to0_batch', Fm1to0)
        #tf.summary.image('Fm2to0_batch', Fm2to0)
        #tf.summary.image('Fp1to0_batch', Fp1to0)
        #tf.summary.image('Fp2to0_batch', Fp2to0)
        
        #F_warped = tf.concat([F0, Fm1, Fm2, Fp1, Fp2], 3)
        F_warped = tf.concat([Fm2, Fm1, F0, Fp1, Fp2], 3)
        
        # Derain-subnet 
        F_res = derain_net(is_train, F_warped, reuse=False, num_frame=5)
        tf.summary.image('F_res_batch', F_res)
        
        # residual learning        
        F_derained = tf.add(F0, F_res)
        #F_derained = tf.subtract(F0, F_res)
        tf.summary.image('F_derained_batch', F_derained)
        
        return F_derained

def eval_network_drain_only(is_train, F0, Fm1, Fm2, Fp1, Fp2, batch_size):
#F0: current frame 'ft'
#Fm1, Fm2: forward frame 'ft-1, ft-2'
#Fp1, Fp2: backward frame  'ft+1, ft+2'
        
        # MC-subnet
        #Fm1to0 = warp_img(batch_size, F0, Fm1, False)
        #Fm2to0 = warp_img(batch_size, F0, Fm2, True)
        #Fp1to0 = warp_img(batch_size, F0, Fp1, True)
        #Fp2to0 = warp_img(batch_size, F0, Fp2, True)
        
        #F_warped = tf.concat([F0, Fm1to0, Fm2to0, Fp1to0, Fp2to0], 3)
        
        
        #tf.summary.image('Fm1to0_batch', Fm1to0)
        #tf.summary.image('Fm2to0_batch', Fm2to0)
        #tf.summary.image('Fp1to0_batch', Fp1to0)
        #tf.summary.image('Fp2to0_batch', Fp2to0)
        
        #F_warped = tf.concat([F0, Fm1, Fm2, Fp1, Fp2], 3)
        F_warped = tf.concat([Fm2, Fm1, F0, Fp1, Fp2], 3)
        
        # Derain-subnet 
        F_res = derain_net(is_train, F_warped, reuse=True, num_frame=5)
        tf.summary.image('F_res_batch', F_res)
        
        # residual learning        
        F_derained = tf.add(F0, F_res)
        #F_derained = tf.subtract(F0, F_res)
        tf.summary.image('F_derained_batch', F_derained)
        
        return F_derained