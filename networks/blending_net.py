import tensorflow as tf
import numpy as np

""" blending_net from https://github.com/google-research/google-research/tree/master/mpi_extrapolation."""

def blending_net(inputs, scope='refine_net'):
  """3D encoder-decoder conv net to predict refined MPI."""

  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

    ksize = 3
    def conv(net, width, ksize=ksize, s=1, d=1):
      """Conv helper function."""

      return tf.layers.conv3d(
          net,
          width,
          ksize,
          strides=s,
          padding='SAME',
          dilation_rate=(d, d, d),
          activation=tf.nn.relu)

    def down_block(net, width, ksize=ksize, do_down=False):
      """Strided convs followed by convs."""

      down = conv(net, width, ksize, 2) if do_down else net
      out = conv(conv(down, width), width)
      return out, out

    def tf_repeat(tensor, repeats):
      """NN upsampling from https://github.com/tensorflow/tensorflow/issues/8246."""

      with tf.variable_scope('repeat'):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
      return repeated_tensor

    def up_block(net, skip, width, ksize=ksize):
      """Nearest neighbor upsampling followed by convs."""

      ch = net.get_shape().as_list()[-1]
      net_repeat = tf_repeat(net, [1, 2, 2, 2, 1])
      net_repeat.set_shape([None, None, None, None, ch])
      up = net_repeat
      up = tf.cond(
          tf.equal(tf.shape(up)[1],
                   tf.shape(skip)[1]), lambda: up, lambda: up[:, :-1, Ellipsis])
      up = tf.cond(
          tf.equal(tf.shape(up)[2],
                   tf.shape(skip)[2]), lambda: up, lambda: up[:, :, :-1, Ellipsis])
      out = tf.concat([up, skip], -1)
      out = conv(conv(out, width, ksize), width, ksize)
      return out

    skips = []
    net = inputs
    sh = inputs.get_shape().as_list()
    w_list = [np.maximum(8, sh[-1]), 16, 32, 64, 128]
    for i in range(len(w_list)):
      net, skip = down_block(net, w_list[i], do_down=(i > 0))
      skips.append(skip)
    net = skips.pop()

    w_list = [64, 32, 16, 8]
    for i in range(len(w_list)):
      with tf.variable_scope('up_block{}'.format(i)):
        skip = skips.pop()
        net = up_block(net, skip, w_list[i])

    ###  blending mpi
    chout = 4 
    net = tf.layers.conv3d(net, chout, ksize, padding='SAME', activation=None)
    sh= net.get_shape().as_list()
    zeros_tensor = tf.zeros([sh[0], sh[1], sh[2], sh[3], 1])
    blend_weights_source = tf.nn.softmax(tf.concat([net[..., 0:3],zeros_tensor], -1), -1)
    blend_weights = tf.tile(blend_weights_source[..., tf.newaxis], [1, 1, 1, 1, 1, 3]) # shape [1, D, H, W, 4, 3]
    colors = tf.reshape(inputs, [sh[0], sh[1], sh[2], sh[3], 4, 3])
    mpi_color = tf.reduce_sum(blend_weights * colors, axis=-2)
    mpi_alpha = tf.nn.sigmoid(net[..., 3:4])
    mpi = tf.concat([mpi_color, mpi_alpha], -1)

    return mpi,blend_weights_source