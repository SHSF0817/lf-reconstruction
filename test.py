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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

flags = tf.app.flags
flags.DEFINE_string('model_path','models/final_model_hci/','the path of model')
flags.DEFINE_string('model_name','model-final','the name of model')
flags.DEFINE_integer('num_depths', 64, 'the number of layers in focal stack')
flags.DEFINE_integer('img_h', 512, 'the height of input subviews')
flags.DEFINE_integer('img_w', 512, 'the width of input subviews')
flags.DEFINE_integer('ang_dim', 9, 'the angular resolution of light field')
flags.DEFINE_integer('ang_out', 9, 'the angular resolution of light field')
flags.DEFINE_integer('num_imgs', 81, 'the number of input subviews')
flags.DEFINE_integer('batch_size', 1, 'the size of single batch')
flags.DEFINE_string('test_data_path','/home/B/hx_data/full_data/test','the path of light field datasets')
# flags.DEFINE_string('test_data_path','/home/B/hx_data/DLFD_test','the path of light field datasets')
# flags.DEFINE_string('test_data_path','./lf_mic/187-187','the path of light field datasets')
flags.DEFINE_string('dataset_name','HCI','HCI or Inria or Mic')
flags.DEFINE_float('resize_scale', 1, 'resize the test images')
flags.DEFINE_string('results_path','./hci_results_ours','the path of output results')

FLAGS = flags.FLAGS

def random_views(input_lf, u, v):
    ang_res = FLAGS.ang_out
    offset = ang_res -1 
    [tl_row, tl_col] = [(FLAGS.ang_dim-ang_res) // 2, (FLAGS.ang_dim-ang_res) // 2]
    source_uv = np.array([[tl_row, tl_col], [tl_row, tl_col+offset], [tl_row+offset, tl_col], [tl_row+offset, tl_col+offset]])
    
    img_tl = input_lf[:, source_uv[0,0], source_uv[0,1], ...]
    img_tr = input_lf[:, source_uv[1,0], source_uv[1,1], ...]
    img_bl = input_lf[:, source_uv[2,0], source_uv[2,1], ...]
    img_br = input_lf[:, source_uv[3,0], source_uv[3,1], ...]
    source_imgs = np.concatenate([img_tl, img_tr, img_bl, img_br], axis=0) # shape [4, H, W ,3]
    
    target_uv = np.array([[tl_row + u, tl_col + v]])
    target_imgs = input_lf[:, tl_row + u, tl_col + v, ...]

    input_data = {}
    input_data['source_imgs'] = source_imgs
    input_data['source_uv'] = source_uv
    input_data['target_imgs'] = target_imgs
    input_data['target_uv'] = target_uv

    return input_data

def read_disp_range(dataset_name, path, sence_names, resize_scale=1.0):
    disp_range_dict = {}
    if dataset_name == 'HCI' or dataset_name == 'Mic':
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

        all_eval = np.zeros([len(test_scene_names), FLAGS.ang_out, FLAGS.ang_out, 3])
        dataset_ssim = 0
        dataset_psnr = 0
        dataset_lpips = 0
        for idx, scene in enumerate(test_scene_names):
            if not tf.gfile.IsDirectory(FLAGS.results_path + scene):
                tf.gfile.MakeDirs(FLAGS.results_path + scene)
            sess.run(test_data.initializer, feed_dict={input_scene:scene})
            img_batch = sess.run(next_test_element)
            img_batch = img_batch.reshape([FLAGS.batch_size, FLAGS.ang_dim, FLAGS.ang_dim, FLAGS.img_h, FLAGS.img_w, 3])

            scene_eval = np.zeros([FLAGS.ang_out, FLAGS.ang_out, 3])
            scene_ssim = 0
            scene_psnr = 0
            scene_lpips = 0
            for u in range(FLAGS.ang_out):
                for v in range(FLAGS.ang_out):
                    test_patch = random_views(img_batch, u, v)
                    out_pre_view, view_lpips = sess.run([final_view_test,lpips_test], 
                                                            feed_dict={
                                                            input_lf:img_batch, 
                                                            source_imgs: test_patch['source_imgs'],
                                                            source_uv: test_patch['source_uv'],
                                                            target_imgs: test_patch['target_imgs'],
                                                            target_uv: test_patch['target_uv'],
                                                            disp_range:test_disp[scene]})
                    out_pre_view = np.clip(out_pre_view, 0, 1.0)
                    plt.imsave( FLAGS.results_path + scene + '/out_%02d_%02d.png' % (u, v), out_pre_view[0,...])

                    # evaluate results
                    view_pred = skicolor.rgb2gray(out_pre_view[0,...])
                    gt_img = skicolor.rgb2gray(np.squeeze(test_patch['target_imgs'][0,...]))

                    view_psnr = psnr(gt_img, view_pred)
                    view_ssim = ssim(gt_img, view_pred)

                    scene_eval[u,v,0] = view_ssim
                    scene_eval[u,v,1] = view_psnr
                    scene_eval[u,v,2] = view_lpips[0]

                    if not ((u == 0 and v ==0) or (u == 0 and v == FLAGS.ang_out-1) or
                            (u == FLAGS.ang_out-1 and v ==0) or (u == FLAGS.ang_out-1 and v == FLAGS.ang_out-1)):
                        scene_ssim = scene_ssim + view_ssim
                        scene_psnr = scene_psnr + view_psnr
                        scene_lpips = scene_lpips + view_lpips

            view_num = FLAGS.ang_out * FLAGS.ang_out - 4
            scene_ssim_mean = scene_ssim / view_num
            scene_psnr_mean = scene_psnr / view_num
            scene_lpips_mean = scene_lpips / view_num
            print('Scene: '+ scene + '---SSIM: ' + str('%.3f' % scene_ssim_mean) + '---PSNR: ' + str('%.3f' % scene_psnr_mean)
                                   + '---LPIPS: ' + str('%.3f' % scene_lpips_mean))
            scene_eval_mean = [scene, scene_ssim_mean, scene_psnr_mean, scene_lpips_mean]
            with open(FLAGS.results_path + scene + '/results'+ '.csv','a',newline='') as f:
                csv_writer = csv.writer(f,dialect='excel')
                csv_writer.writerows(scene_eval[:,:,0])
                csv_writer.writerows(scene_eval[:,:,1])
                csv_writer.writerows(scene_eval[:,:,2])
                csv_writer.writerow(scene_eval_mean)
            all_eval[idx,...] = scene_eval
            dataset_ssim = dataset_ssim + scene_ssim_mean
            dataset_psnr= dataset_psnr + scene_psnr_mean
            dataset_lpips = dataset_lpips + scene_lpips_mean
        
        dataset_ssim_mean = dataset_ssim / len(test_scene_names)
        dataset_psnr_mean = dataset_psnr / len(test_scene_names)
        dataset_lpips_mean = dataset_lpips / len(test_scene_names)
        all_eval_mean = np.mean(all_eval, axis=0)
        print(' mean ssim: ' + str('%.3f' % dataset_ssim_mean) + ' mean psnr: ' + str('%.3f' % dataset_psnr_mean) 
                                                                       + ' mean lpips: ' + str('%.3f' % dataset_lpips_mean))
        all_mean = ['mean', dataset_ssim_mean, dataset_psnr_mean, dataset_lpips_mean]                                        
        with open(FLAGS.results_path + 'all_results_mean'+ '.csv','a',newline='') as f:
            csv_writer = csv.writer(f,dialect='excel')
            csv_writer.writerows(all_eval_mean[:,:,0])
            csv_writer.writerows(all_eval_mean[:,:,1])
            csv_writer.writerows(all_eval_mean[:,:,2])
            csv_writer.writerow(all_mean)
        print('done!')

if __name__ == '__main__':
    test()
