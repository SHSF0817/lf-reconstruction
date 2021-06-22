import os
import random
import configparser
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import progressbar
import csv
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage import color as skicolor
from PIL import Image

from utils.data_loader import get_batch
from networks.blending_net import blending_net
from pipeline_generation import Pipeline

slim = tf.contrib.slim
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '6'

flags = tf.app.flags
flags.DEFINE_string('model_path','model/4_21_32_hci/','the path of model')
flags.DEFINE_string('model_name','model','the name of model')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')
flags.DEFINE_integer('num_depths', 32, 'the number of layers in focal stack')
flags.DEFINE_integer('img_h', 512, 'the height of input views')
flags.DEFINE_integer('img_w', 512, 'the width of input views')
flags.DEFINE_integer('patch_h', 128, 'the height of input subviews')
flags.DEFINE_integer('patch_w', 128, 'the width of input subviews')
flags.DEFINE_integer('valid_w', 512, 'the width of input subviews')
flags.DEFINE_integer('valid_h', 512, 'the width of input subviews')
flags.DEFINE_integer('ang_dim', 9, 'the angular resolution of light field')
flags.DEFINE_integer('num_imgs', 81, 'the number of input subviews')
flags.DEFINE_integer('batch_size', 1, 'the size of single batch')
flags.DEFINE_integer('max_step', 80000, 'the epochs of training')
flags.DEFINE_boolean('continue_train', False, 'Continue training from previous checkpoint')

### HCI dataset
flags.DEFINE_string('train_data_path','/home/B/hx_data/full_data/additional','the path of light field datasets')
flags.DEFINE_string('valid_data_path','/home/B/hx_data/full_data/test','the path of light field datasets')

# # ### Inria DLFD dataset
# flags.DEFINE_string('train_data_path','/home/B/hx_data/DLFD_train','the path of light field datasets')
# flags.DEFINE_string('valid_data_path','/home/B/hx_data/DLFD_test','the path of light field datasets')


flags.DEFINE_string('dataset_name','HCI','HCI or Inria')
flags.DEFINE_float('resize_scale', 1.0, 'resize the test images')
flags.DEFINE_string('results_path','ours_32_layers/','the path of output results')

FLAGS = flags.FLAGS

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
    input_data = {}
    input_data['source_imgs'] = source_imgs
    input_data['source_uv'] = source_uv
    input_data['target_imgs'] = target_imgs
    input_data['target_uv'] = target_uv

    return input_data

def normal(img):
    img = img - np.min(img)
    img = img / np.max(img)
    return img

def train():

    input_lf = tf.placeholder(tf.float32,[FLAGS.batch_size,FLAGS.ang_dim, FLAGS.ang_dim, FLAGS.patch_h, FLAGS.patch_w, 3]) #  [N,H,W,3]
    source_imgs = tf.placeholder(tf.float32,[ 4, FLAGS.patch_h, FLAGS.patch_w, 3])
    source_uv = tf.placeholder(tf.int32,[4, 2])
    target_imgs = tf.placeholder(tf.float32,[ 1, FLAGS.patch_h, FLAGS.patch_w, 3])
    target_uv = tf.placeholder(tf.int32,[1, 2])
    disp_range = tf.placeholder(tf.float32, [FLAGS.batch_size, 2])
    input_scene = tf.placeholder(tf.string)

    input_lf_valid = tf.placeholder(tf.float32,[FLAGS.batch_size,FLAGS.ang_dim, FLAGS.ang_dim, FLAGS.valid_h, FLAGS.valid_w, 3]) #  [N,H,W,3]
    source_imgs_valid = tf.placeholder(tf.float32,[ 4, FLAGS.valid_h, FLAGS.valid_w, 3])
    target_imgs_valid = tf.placeholder(tf.float32,[ 1, FLAGS.valid_h, FLAGS.valid_w, 3])

    ### mpi network
    net = blending_net()

    ### training pipeline
    pipeline_train = Pipeline(net, input_lf, source_imgs, source_uv, target_imgs, target_uv, disp_range, FLAGS.num_depths)
    final_view_train, disp_map_train = pipeline_train.inference()
    loss_train = pipeline_train.loss(final_view_train)

    ### valid pipeline
    pipeline_valid = Pipeline(net, input_lf_valid, source_imgs_valid, source_uv, target_imgs_valid, target_uv, disp_range, FLAGS.num_depths)
    final_view_valid, disp_map_valid = pipeline_valid.inference()
    loss_valid = pipeline_valid.loss(final_view_valid)

    train_data = get_batch(FLAGS.train_data_path, input_scene, FLAGS.img_h, FLAGS.img_w, FLAGS.num_imgs, FLAGS.dataset_name)
    # train_data = get_batch(FLAGS.train_data_path, 'input_Cam???.png', FLAGS.img_h, FLAGS.img_w, FLAGS.batch_size)
    next_train_element = train_data.get_next()
    next_train_element = tf.random_crop(next_train_element, [FLAGS.num_imgs, FLAGS.patch_h, FLAGS.patch_w, 3])

    valid_data = get_batch(FLAGS.valid_data_path, input_scene, FLAGS.valid_h, FLAGS.valid_w, FLAGS.num_imgs, FLAGS.dataset_name)
    # valid_data = get_batch(FLAGS.valid_data_path, 'input_Cam???.png', FLAGS.patch_h, FLAGS.patch_w, FLAGS.batch_size)
    next_valid_element = valid_data.get_next()

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    train_op = optimizer.minimize(loss_train)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    incr_global_step = tf.assign(global_step, global_step + 1)
    variables_mpi= slim.get_variables()
    saver = tf.train.Saver(variables_mpi, max_to_keep=3)

    variables_vgg = slim.get_variables(scope='vgg_19')
    vgg_model_path = './models/vgg19_model/vgg_19.ckpt'
    saver_vgg = tf.train.Saver(variables_vgg, max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
    
        if FLAGS.continue_train == True:
            print('--- Loading the model......')
            # learning_curve = pickle.load(open(FLAGS.model_path  + 'learning_curve.dat','rb'))
            saver.restore(sess, FLAGS.model_path + FLAGS.model_name)
        else:
            print('--- Initialization......')
            learning_curve = []

        print('--- Loading vgg19 weights...')
        saver_vgg.restore(sess, vgg_model_path)

        print('--- Starting training ---')
        train_scene_names = list(filter(lambda scene: os.path.splitext(scene)[1] == '',os.listdir(FLAGS.train_data_path)))
        train_disp = read_disp_range(FLAGS.dataset_name, FLAGS.train_data_path, train_scene_names, FLAGS.resize_scale)

        valid_scene_names = list(filter(lambda scene: os.path.splitext(scene)[1] == '',os.listdir(FLAGS.valid_data_path)))
        valid_disp = read_disp_range(FLAGS.dataset_name, FLAGS.valid_data_path, valid_scene_names, FLAGS.resize_scale)

        if not tf.gfile.IsDirectory(FLAGS.model_path):
            tf.gfile.MakeDirs(FLAGS.model_path)

        if not tf.gfile.IsDirectory(FLAGS.results_path):
            tf.gfile.MakeDirs(FLAGS.results_path)

        best_ssim = 0
        step = 0
        while step < FLAGS.max_step:
            valid_loss = 0.0
            random.shuffle(train_scene_names)
            progress = progressbar.ProgressBar()
            for scene in progress(train_scene_names):
                sess.run(train_data.initializer, feed_dict={input_scene:scene})
                img_batch = sess.run(next_train_element)
                img_batch = img_batch.reshape([FLAGS.batch_size, FLAGS.ang_dim, FLAGS.ang_dim, FLAGS.patch_h, FLAGS.patch_w, 3])
                train_patch = random_views(img_batch, True)
                _, step= sess.run([train_op, incr_global_step], 
                                             feed_dict={
                                                 input_lf:img_batch, 
                                                 source_imgs: train_patch['source_imgs'],
                                                 source_uv: train_patch['source_uv'],
                                                 target_imgs: train_patch['target_imgs'],
                                                 target_uv: train_patch['target_uv'],
                                                 disp_range:train_disp[scene]})
            epoch = step // len(train_scene_names)
            if epoch % 10 == 0:
                print('--- Starting testing ---')
                psnr_list = []
                ssim_list = []
                for scene in valid_scene_names:
                    sess.run(valid_data.initializer, feed_dict={input_scene:scene})
                    img_batch = sess.run(next_valid_element)
                    img_batch = img_batch.reshape([FLAGS.batch_size, FLAGS.ang_dim, FLAGS.ang_dim, FLAGS.valid_h, FLAGS.valid_w, 3])
                    valid_patch = random_views(img_batch, False)
                    part_loss, out_pre_view, disp_map = \
                    sess.run([loss_valid, final_view_valid, disp_map_valid],
                                                                feed_dict={
                                                                    input_lf_valid:img_batch, 
                                                                    source_imgs_valid: valid_patch['source_imgs'],
                                                                    source_uv: valid_patch['source_uv'],
                                                                    target_imgs_valid: valid_patch['target_imgs'],
                                                                    target_uv: valid_patch['target_uv'],
                                                                    disp_range:valid_disp[scene]})
                    valid_loss += part_loss
                    out_pre_view = np.clip(out_pre_view, 0, 1.0)

                    plt.imsave( FLAGS.results_path + scene + '_rendered.png',out_pre_view[0,...])
                    plt.imsave( FLAGS.results_path + scene + '_gt.png',valid_patch['target_imgs'][0,...])
                    disp_map = Image.fromarray(normal(disp_map[0,...,0]) * 255)
                    disp_map = disp_map.convert('L')
                    disp_map.save(FLAGS.results_path + scene + '_disp_' +'.png')


                    ### evaluate results
                    view_pred = skicolor.rgb2gray(out_pre_view[0,...])
                    gt_img = skicolor.rgb2gray(np.squeeze(valid_patch['target_imgs'][0,...]))
                    view_psnr = psnr(gt_img, view_pred)
                    view_ssim = ssim(gt_img, view_pred)
                    psnr_list.append(view_psnr)
                    ssim_list.append(view_ssim)

                    print('%-30s:' % scene + 'Loss: ' + str('%.3f' % part_loss) + '--SSIM: ' + str('%.3f' % view_ssim) + '--PSNR: ' + str('%.3f' % view_psnr))
                
            
                mean_ssim = np.mean(ssim_list)
                mean_psnr = np.mean(psnr_list)
                mean_loss = valid_loss / len(valid_scene_names)
                print('Epoch: '+ str(epoch) + ' mean ssim: ' + str('%.3f' % mean_ssim) + ' mean psnr: ' + str('%.3f' % mean_psnr)
                    + ' mean loss: ' + str('%.3f' % mean_loss) )
                
                ### write to CSV file
                results = [epoch, mean_ssim, mean_psnr, mean_loss]
                with open('ours_32.csv','a',newline='') as f:
                    csv_writer = csv.writer(f,dialect='excel')
                    csv_writer.writerow(results)
                
                
                if mean_ssim >= best_ssim:
                    best_ssim = mean_ssim
                    print('--- Saving model ---')
                    saver.save(sess, FLAGS.model_path + FLAGS.model_name, step)

        print('done!')

if __name__ == '__main__':
    train()
