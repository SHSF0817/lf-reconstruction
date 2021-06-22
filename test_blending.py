import os
import configparser
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from progressbar import *

from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage import color as skicolor
from PIL import Image
import csv

from utils.data_loader import get_batch
from networks.generation_net import generation_net
from pipeline_blending import Pipeline


slim = tf.contrib.slim
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

flags = tf.app.flags
flags.DEFINE_string('model_path','model/final_model_hci/','the path of model')
flags.DEFINE_string('model_name','model-final','the name of model')
flags.DEFINE_integer('num_depths', 64, 'the number of layers in focal stack')
flags.DEFINE_integer('img_h', 512, 'the height of input subviews')
flags.DEFINE_integer('img_w', 512, 'the width of input subviews')
flags.DEFINE_integer('ang_dim', 9, 'the angular resolution of light field')
flags.DEFINE_integer('num_imgs', 81, 'the number of input subviews')
flags.DEFINE_integer('batch_size', 1, 'the size of single batch')
# flags.DEFINE_string('test_data_path','/home/B/hx_data/full_data/test','the path of light field datasets')
flags.DEFINE_string('test_data_path','/home/B/hx_data/DLFD_test','the path of light field datasets')
flags.DEFINE_string('dataset_name','Inria','HCI or Inria')
flags.DEFINE_float('resize_scale', 1, 'resize the test images')
flags.DEFINE_string('results_path','inria-44/','the path of output results')

FLAGS = flags.FLAGS

def random_views(input_lf, train=True):
    _, angular_dim, _, img_h, img_w, _ = np.shape(input_lf)
    if train:
        offset = np.random.randint(1, angular_dim)
        [tl_row, tl_col] = np.random.randint(0, angular_dim-offset, size=2)
        [target_row, target_col] = np.random.randint(0, angular_dim, size=2)
    else:
        offset = 8
        [tl_row, tl_col] = [0, 0]
        [target_row, target_col] = [4, 4]
    source_uv = np.array([[tl_row, tl_col], [tl_row, tl_col+offset], [tl_row+offset, tl_col], [tl_row+offset, tl_col+offset]])
    
    img_tl = input_lf[:, source_uv[0,0], source_uv[0,1], ...]
    img_tr = input_lf[:, source_uv[1,0], source_uv[1,1], ...]
    img_bl = input_lf[:, source_uv[2,0], source_uv[2,1], ...]
    img_br = input_lf[:, source_uv[3,0], source_uv[3,1], ...]
    source_imgs = np.concatenate([img_tl, img_tr, img_bl, img_br], axis=0) # shape [4, H, W ,3]
    
    target_uv = np.array([[target_row, target_col]])
    target_imgs = input_lf[:, target_uv[0,0], target_uv[0,1], ...]
    # target_imgs2 = input_lf[:, target_uv[0,0]+1, target_uv[0,1], ...]
    target_imgs = np.concatenate([target_imgs], axis=0) # shape [2, H, W, 3]

    input_data = {}
    input_data['source_imgs'] = source_imgs
    input_data['source_uv'] = source_uv
    input_data['target_imgs'] = target_imgs
    input_data['target_uv'] = target_uv

    return input_data

def read_disp_range(dataset_name, path, sence_names, resize_scale=1.0):
    disp_range_dict = {}
    if dataset_name == 'HCI':
        for scene in sence_names:
            config = configparser.ConfigParser()
            config.read( path + '/' + scene + '/' + 'parameters.cfg')
            disp_min = float(config.get('meta','disp_min'))
            disp_max = float(config.get('meta','disp_max'))
            disp_range_dict[scene] = [[disp_min * resize_scale, disp_max * resize_scale]]
    
    if dataset_name == 'Inria':
        for scene in sence_names:
            disp_map = np.load(path + '/' + scene + '/' + 'disparity_1_1.npy')
            disp_min = float(np.min(disp_map))
            disp_max = float(np.max(disp_map))
            disp_range_dict[scene] = [[disp_min * resize_scale, disp_max * resize_scale]]
    
    return disp_range_dict
    



def normal(img):
    img = img - np.min(img)
    img = img / np.max(img)
    return img

def test():

    input_lf = tf.placeholder(tf.float32,[FLAGS.batch_size,FLAGS.ang_dim, FLAGS.ang_dim, FLAGS.img_h, FLAGS.img_w, 3]) #  [N,H,W,3]
    source_imgs = tf.placeholder(tf.float32,[ 4, FLAGS.img_h, FLAGS.img_w, 3])
    source_uv = tf.placeholder(tf.int32,[4, 2])
    target_imgs = tf.placeholder(tf.float32,[ 1, FLAGS.img_h, FLAGS.img_w, 3])
    target_uv = tf.placeholder(tf.int32,[1, 2])
    disp_range = tf.placeholder(tf.float32, [FLAGS.batch_size, 2])
    input_scene = tf.placeholder(tf.string)

    ####################################################################################
    
    net = generation_net()
    pipeline_test = Pipeline(net, input_lf, source_imgs, source_uv, target_imgs, target_uv, disp_range, FLAGS.num_depths)
    final_view_test, disp_map_test = pipeline_test.inference()
    loss_test = pipeline_test.loss(final_view_test)
    lpips_test = pipeline_test.lpips_value(final_view_test)

    test_data = get_batch(FLAGS.test_data_path, input_scene, FLAGS.img_h, FLAGS.img_w, FLAGS.num_imgs, FLAGS.dataset_name)
    next_test_element = test_data.get_next()
    ####################################################################################
    test_scene_names = list(filter(lambda scene: os.path.splitext(scene)[1] == '', os.listdir(FLAGS.test_data_path)))
    test_disp = read_disp_range(FLAGS.dataset_name, FLAGS.test_data_path, test_scene_names, FLAGS.resize_scale)

    variables_mpi= slim.get_trainable_variables()
    saver = tf.train.Saver(variables_mpi)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
    
        print('--- Loading the model......')
        saver.restore(sess, FLAGS.model_path + FLAGS.model_name)
        # saver_pre.restore(sess, pre_model_path)

        print('--- Starting testing ---')

        if not tf.gfile.IsDirectory(FLAGS.results_path):
            tf.gfile.MakeDirs(FLAGS.results_path)

        psnr_list = []
        ssim_list = []
        lpips_list = []
        test_loss = 0
        for scene in test_scene_names:
            sess.run(test_data.initializer, feed_dict={input_scene:scene})
            img_batch = sess.run(next_test_element)
            img_batch = img_batch.reshape([FLAGS.batch_size, FLAGS.ang_dim, FLAGS.ang_dim, FLAGS.img_h, FLAGS.img_w, 3])
            test_patch = random_views(img_batch, False)
            part_loss, out_pre_view, disp_map, lp_value = sess.run([loss_test, final_view_test, 
                                                         disp_map_test, lpips_test], 
                                                            feed_dict={
                                                            input_lf:img_batch, 
                                                            source_imgs: test_patch['source_imgs'],
                                                            source_uv: test_patch['source_uv'],
                                                            target_imgs: test_patch['target_imgs'],
                                                            target_uv: test_patch['target_uv'],
                                                            disp_range:test_disp[scene]})
            test_loss += part_loss
            out_pre_view = np.clip(out_pre_view, 0, 1.0)

            plt.imsave( FLAGS.results_path + scene + '_center.png',out_pre_view[0,...])
            plt.imsave( FLAGS.results_path + scene + '_gt.png',test_patch['target_imgs'][0,...])
            disp_map = Image.fromarray(normal(disp_map[0,...,0]) * 255)
            disp_map = disp_map.convert('L')
            disp_map.save(FLAGS.results_path + scene + '_disp.png')

            ### evaluate results
            view_pred = skicolor.rgb2gray(out_pre_view[0,...])
            gt_img = skicolor.rgb2gray(np.squeeze(test_patch['target_imgs'][0,...]))
            view_psnr = psnr(gt_img, view_pred)
            view_ssim = ssim(gt_img, view_pred)
            psnr_list.append(view_psnr)
            ssim_list.append(view_ssim)
            lpips_list.append(lp_value)


            print('Scene: '+ scene + '---Loss: ' + str('%.3f' % part_loss) + \
                    '---SSIM: ' + str('%.3f' % view_ssim) + '---PSNR: ' + str('%.3f' % view_psnr))
            results = [scene, view_ssim, view_psnr, lp_value[0]]
            with open(FLAGS.results_path + 'results_blend2'+ '.csv','a',newline='') as f:
                csv_writer = csv.writer(f,dialect='excel')
                csv_writer.writerow(results)
        print(' mean ssim: ' + str('%.3f' % np.mean(ssim_list)) + ' mean psnr: ' + str('%.3f' % np.mean(psnr_list)) \
                                                + ' test loss: ' + str('%.3f' % (test_loss / len(test_scene_names))))
        results = ['mean', np.mean(ssim_list), np.mean(psnr_list), np.mean(lpips_list)]                                        
        with open(FLAGS.results_path + 'results_blend2'+ '.csv','a',newline='') as f:
            csv_writer = csv.writer(f,dialect='excel')
            csv_writer.writerow(results)
        print('done!')

if __name__ == '__main__':
    test()
