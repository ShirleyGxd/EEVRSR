# -*- coding: utf-8 -*-
"""
@author: gxd
"""

import tensorflow as tf
import numpy as np
import skimage.measure as ski

#%%
def BatchNorm(input_x, is_train, decay=0.999, name='BatchNorm'):
    from tensorflow.python.training import moving_averages
    from tensorflow.python.ops import control_flow_ops
    
    axis = list(range(len(input_x.get_shape()) - 1))
    fdim = input_x.get_shape()[-1:]
    
    with tf.variable_scope(name):
        beta = tf.get_variable('beta', fdim, initializer=tf.constant_initializer(value=0.0))
        gamma = tf.get_variable('gamma', fdim, initializer=tf.constant_initializer(value=1.0))
        moving_mean = tf.get_variable('moving_mean', fdim, initializer=tf.constant_initializer(value=0.0), trainable=False)
        moving_variance = tf.get_variable('moving_variance', fdim, initializer=tf.constant_initializer(value=0.0), trainable=False)
  
        def mean_var_with_update():
            batch_mean, batch_variance = tf.nn.moments(input_x, axis)
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, batch_mean, decay, zero_debias=True)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, batch_variance, decay, zero_debias=True)
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                return tf.identity(batch_mean), tf.identity(batch_variance)

        mean, variance = control_flow_ops.cond(is_train, mean_var_with_update, lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(input_x, mean, variance, beta, gamma, 1e-3) #, tf.stack([mean[0], variance[0], beta[0], gamma[0]])


#%%
def Conv2D(input_x, kernel_shape, stride_hw, padding, name='Conv2d', W_initializer=tf.contrib.layers.xavier_initializer(uniform=True), bias=True):
    strides=[1, stride_hw, stride_hw, 1]
    
    with tf.variable_scope(name):
        W = tf.get_variable("W", kernel_shape, initializer=W_initializer)
        if bias is True:
            b = tf.get_variable("b", (kernel_shape[-1]),initializer=tf.constant_initializer(value=0.0))
        else:
            b = 0
        
    return tf.nn.conv2d(input_x, W, strides, padding) + b


#%%
def Dense_B_Layer(input_x, is_train, in_channels, growth_k, stride_hw, padding, layer_name):    
    kernel_shape_1=[1, 1, in_channels, 4*growth_k]
    kernel_shape_2=[3, 3, 4*growth_k, growth_k]
    
    with tf.variable_scope(layer_name):
        x = BatchNorm(input_x, is_train, name=layer_name+'_BN1')
        x = tf.nn.relu(x)
        x = Conv2D(x, kernel_shape_1, stride_hw, padding, name=layer_name+'_conv1x1') 
        
        x = BatchNorm(x, is_train, name=layer_name+'_BN2')
        x = tf.nn.relu(x)
        y = Conv2D(x, kernel_shape_2, stride_hw, padding, name=layer_name+'_conv3x3') 
    
    return y


#%%
def DenseBlock(x, is_train, nb_layers, in_channels, growth_k, stride_hw, padding, block_name='DenseBlock'):
    with tf.variable_scope(block_name):
        for i in range(nb_layers):
            t=Dense_B_Layer(x, is_train, in_channels, growth_k, stride_hw, padding, layer_name=block_name+'_Dense_B_Layer'+str(i+1))
            x = tf.concat([x, t], 3)
            in_channels += growth_k
    
    return t



#%%
def Transformer(batch, chan, flow, U , out_size, name='SpatialTransformer', **kwargs):

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _repeat2(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1)
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(rep, tf.reshape(x, (1, -1)))
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            x = tf.cast(_repeat2(tf.range(0, width), height * num_batch), 'float32') + x * 64 
            y = tf.cast(_repeat2(_repeat(tf.range(0, height), width), num_batch), 'float32') + y * 64 

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32') #left
            x1 = x0 + 1 #right
            y0 = tf.cast(tf.floor(y), 'int32') #up
            y1 = y0 + 1 #down

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width*height
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)

            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0 
            idx_b = base_y1 + x0 
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1 

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a) 
            Ib = tf.gather(im_flat, idx_b) 
            Ic = tf.gather(im_flat, idx_c) 
            Id = tf.gather(im_flat, idx_d) 

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
			
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):

            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=tf.stack([1, width])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
            return grid

    def _transform(x_s, y_s, input_dim, out_size):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(input_dim)[0]
            height = tf.shape(input_dim)[1]
            width = tf.shape(input_dim)[2]
            num_channels = tf.shape(input_dim)[3]

            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]

            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(
                input_dim, x_s_flat, y_s_flat,
                out_size)

            output = tf.reshape(
                input_transformed, tf.stack([batch, out_height, out_width, chan]))
            return output

    with tf.variable_scope(name):
        dx, dy = tf.split(flow, 2, 3)
        output = _transform(dx, dy, U, out_size)
        return output

    