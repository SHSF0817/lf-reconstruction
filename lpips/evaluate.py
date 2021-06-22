# import numpy as np
# import tensorflow as tf
# import lpips_tf
# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# batch_size = 32
# image_shape = (batch_size, 64, 64, 3)
# image0 = np.random.random(image_shape)
# image1 = np.random.random(image_shape)
# image0_ph = tf.placeholder(tf.float32)
# image1_ph = tf.placeholder(tf.float32)

# distance_t = lpips_tf.lpips(image0_ph, image1_ph, model='net-lin', net='alex')

# with tf.Session() as session:
#     distance = session.run(distance_t, feed_dict={image0_ph: image0, image1_ph: image1})
#     print(distance)