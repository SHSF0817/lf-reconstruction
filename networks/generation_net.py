import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
from tensorflow.contrib.layers import layer_norm

class generation_net():
    def net(self,inputs,num_img):
        def upsampling(x):
            b,d,h,w,c = x.get_shape().as_list()
            expand_x = x[:,:,tf.newaxis,:,tf.newaxis,:,tf.newaxis,:]  # shape: [B,D,1,H,1,W,1,C]
            expand_x = tf.tile(expand_x, [1, 1, 2, 1, 2, 1, 2, 1])
            output_x = tf.reshape(expand_x, [b,2*d, 2*h,2*w,c])
            return output_x

        with tf.variable_scope('mpi_pre_net',reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv3d],trainable=True):
                cnv1_1 = slim.conv3d(inputs,8,3,scope='conv3d', stride=1)
                cnv1_1 = layer_norm(cnv1_1,scope='LayerNorm')
                cnv1_2 = slim.conv3d(cnv1_1,num_outputs=16,kernel_size=3,scope='conv3d_1', stride=2)
                cnv1_2 = layer_norm(cnv1_2, scope='LayerNorm_1')

                cnv2_1 = slim.conv3d(cnv1_2,16,3,scope='conv3d_2', stride=1)
                cnv2_1 = layer_norm(cnv2_1, scope='LayerNorm_2')
                cnv2_2 = slim.conv3d(cnv2_1,32,3,scope='conv3d_3', stride=2)
                cnv2_2 = layer_norm(cnv2_2, scope='LayerNorm_3')

                cnv3_1 = slim.conv3d(cnv2_2,32,3,scope='conv3d_4', stride=1)
                cnv3_1 = layer_norm(cnv3_1, scope='LayerNorm_4')
                cnv3_2 = slim.conv3d(cnv3_1,32,3,scope='conv3d_5', stride=1)
                cnv3_2 = layer_norm(cnv3_2, scope='LayerNorm_5')
                cnv3_3 = slim.conv3d(cnv3_2,64,3,scope='conv3d_6', stride=2)
                cnv3_3 = layer_norm(cnv3_3, scope='LayerNorm_6')

                cnv4_1 = slim.conv3d(cnv3_3,64,3, scope='conv3d_7', stride=1, rate=2)
                cnv4_1 = layer_norm(cnv4_1, scope='LayerNorm_7')
                cnv4_2 = slim.conv3d(cnv4_1,64,3, scope='conv3d_8', stride=1, rate=2)
                cnv4_2 = layer_norm(cnv4_2, scope='LayerNorm_8')
                cnv4_3 = slim.conv3d(cnv4_2,64,3, scope='conv3d_9', stride=1, rate=2)
                cnv4_3 = layer_norm(cnv4_3, scope='LayerNorm_9')

                concat_1 = tf.concat([cnv4_3,cnv3_3],-1)
                nnup5 = upsampling(concat_1)

                cnv5_1 = slim.conv3d(nnup5,32,3,scope='conv3d_10', stride=1)
                cnv5_1 = layer_norm(cnv5_1, scope='LayerNorm_10')
                cnv5_2 = slim.conv3d(cnv5_1,32,3,scope='conv3d_11', stride=1)
                cnv5_2 = layer_norm(cnv5_2, scope='LayerNorm_11')
                cnv5_3 = slim.conv3d(cnv5_2,32,3,scope='conv3d_12', stride=1)
                cnv5_3 = layer_norm(cnv5_3, scope='LayerNorm_12')

                concat_2 = tf.concat([cnv5_3,cnv2_2],-1)
                nnup6 = upsampling(concat_2)

                cnv6_1 = slim.conv3d(nnup6,16,3,scope='conv3d_13',stride=1)
                cnv6_1 = layer_norm(cnv6_1, scope='LayerNorm_13')
                cnv6_2 = slim.conv3d(cnv6_1,16,3,scope='conv3d_14',stride=1)
                cnv6_2 = layer_norm(cnv6_2, scope='LayerNorm_14')

                concat_3 = tf.concat([cnv6_2,cnv1_2],-1)
                nnup7 = upsampling(concat_3)

                cnv7_1 = slim.conv3d(nnup7,8,3,scope='conv3d_15', stride=1)
                cnv7_1 = layer_norm(cnv7_1, scope='LayerNorm_15')
                cnv7_2 = slim.conv3d(cnv7_1,8,3,scope='conv3d_16', stride=1)
                cnv7_2 = layer_norm(cnv7_2, scope='LayerNorm_16')
                cnv7_3 = slim.conv3d(cnv7_2,num_img,3,scope='conv3d_17', stride=1,activation_fn=None)
            return cnv7_3

    def get_mpi(self,inputs,num_outputs):

        output = self.net(inputs, num_outputs) # [B, D, H, W, num_outputs] 

        ### alpha channel
        alpha = tf.nn.sigmoid(output[..., 0:1])

        ### color channel 
        zeros = tf.zeros_like(alpha)
        color_weights = tf.nn.softmax(tf.concat([output[..., 1:2], zeros], -1))
        B, D, H, W, _ = inputs.get_shape().as_list()
        inputs_reshape = tf.reshape(inputs, [B, D, H, W, 2, 3])
        color = tf.reduce_sum(inputs_reshape*color_weights[...,tf.newaxis], -2) # blend the focal stack and corresponding ref image

        mpi = tf.concat([color, alpha], -1)
        return mpi








