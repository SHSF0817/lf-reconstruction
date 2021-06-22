import os
import configparser
import pickle
import tensorflow as tf
import numpy as np

def get_batch(path, sence,  h, w, bs, data_type):

    with tf.name_scope('data_batch'):
        def load_single_image(filename):
            contents = tf.read_file(filename)
            images = tf.image.convert_image_dtype(tf.image.decode_png(contents), tf.float32)
            # images = images[:h,:w,:]
            images_resize = tf.image.resize_images(images, [h, w])
            images_resize.set_shape([h,w,3])
            return images_resize
        if data_type == 'HCI':
            imgs_glob = 'input*.png'
        elif data_type == 'Mic':
            imgs_glob = '*.png'
        else:
            imgs_glob = 'lf*.png'
        
        filepath = tf.data.Dataset.list_files(path + '/' + sence + '/' + imgs_glob, shuffle=False)
        dataset = filepath.map(load_single_image,num_parallel_calls=bs)
        dataset = dataset.batch(bs).repeat()
        iterator = dataset.make_initializable_iterator()
    return iterator

# def get_batch(root_path, imgs_glob,  h, w, batch_size):
#     with tf.name_scope('data_batch'):
#         assert tf.gfile.Glob(root_path)
#         def load_image_data(scene_name):
#             imgs_path = tf.data.Dataset.list_files(scene_name + '/' + imgs_glob, shuffle=False)
#             images = tf.contrib.data.get_single_element(imgs_path.map(load_single_image).batch(81))

#             config = configparser.ConfigParser()
#             config.read( scene_name + '/' + 'parameters.cfg')
#             disps= tf.constant([[config.get('meta','disp_min'), config.get('meta','disp_max')]], tf.float32)

#             return images, disps

#         def load_single_image(filename):
#             contents = tf.read_file(filename)
#             images = tf.image.convert_image_dtype(tf.image.decode_png(contents), tf.float32)
#             images_resize = tf.image.resize_images(images, [h, w])
#             images_resize.set_shape([h,w,3])
#             return images_resize

#         scenes_path = tf.data.Dataset.list_files(root_path + '/*', shuffle=True)
#         lf_sences = scenes_path.map(load_image_data)
#         lf_sences = lf_sences.batch(batch_size).repeat()
#         iterator = lf_sences.make_initializable_iterator()
#     return iterator