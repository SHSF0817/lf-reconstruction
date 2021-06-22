import tensorflow as tf
from lpips import lpips_tf
slim = tf.contrib.slim

def disp_warp(imgs, source_uv, t_row, t_col, disp_range):
    # with tf.device('/gpu:2'):
    N, D, H, W, C = imgs.get_shape().as_list() # [N, D, H, W, C]
    imgs = tf.reshape(imgs, [N*D, H, W, C]) # [N*D, H, W, C]

    min_disp = disp_range[0,0]
    max_disp = disp_range[0,1]
    disparities = tf.linspace(min_disp, max_disp, D)

    v_disp = tf.cast(tf.reshape(source_uv[:,0] - t_row, [N,1]), tf.float32) * tf.reshape(disparities,[1,D]) # [N, D]
    h_disp = tf.cast(tf.reshape(source_uv[:,1] - t_col, [N,1]), tf.float32) * tf.reshape(disparities,[1,D]) # [N, D]
    disp_map = tf.concat([h_disp[...,tf.newaxis], v_disp[...,tf.newaxis]], -1) # [N, D, 2]
    disp_map = tf.tile(disp_map[:, :, tf.newaxis, tf.newaxis, :], [1, 1, H, W, 1]) # [N, D, H, W, 2]

    # warp
    xg,yg = tf.meshgrid(tf.range(H), tf.range(W), indexing='ij')
    coords_t = tf.stack([yg, xg], 2)
    tile_coords = tf.tile(coords_t[tf.newaxis, tf.newaxis,...], [N, D, 1, 1, 1]) # [N, D, H, W, 2]
    warp_coords = tf.cast(tile_coords, disp_map.dtype) - disp_map
    warp_coords = tf.reshape(warp_coords, [N*D, H, W, 2])

    warp_imgs = tf.squeeze(tf.contrib.resampler.resampler(imgs, warp_coords)) # [N*D, H, W, C]
    warp_imgs = tf.reshape(warp_imgs, [N, D, H, W, C])
    return warp_imgs

def get_focal_stack(source_imgs, source_uv, disp_range, num_depths, ref_idx):

    N = source_imgs.get_shape().as_list()[0]
    t_row = source_uv[ref_idx, 0]
    t_col = source_uv[ref_idx, 1]

    tile_imgs = tf.tile(source_imgs[:,tf.newaxis,...], [1, num_depths, 1, 1, 1]) # shape [N, D, H, W, 3]
    warp_imgs = disp_warp(tile_imgs, source_uv, t_row, t_col, disp_range)

    # add warp num to sovle the black edges
    tile_num = tf.ones_like(tile_imgs)
    warp_num = disp_warp(tile_num, source_uv, t_row, t_col, disp_range)
    fock_stack = tf.reduce_sum(warp_imgs, 0, keepdims=True) / tf.reduce_sum(warp_num, 0, keepdims=True) # [1, D, H, W, C]
    tile_ref_images = tf.tile(source_imgs[ref_idx:ref_idx+1, tf.newaxis, ...], [1, num_depths, 1, 1, 1]) # shape [1, D, H, W, C]

    return fock_stack, tile_ref_images

def render_views(mpis, source_uv, target_uv, disp_range):

    t_row = target_uv[0, 0]
    t_col = target_uv[0, 1]

    warped_mpis = disp_warp(mpis, source_uv, t_row, t_col, disp_range)

    mpis_color = warped_mpis[..., 0:3]  # [N, D, H, W, 3]
    mpis_alpha = warped_mpis[..., 3:4]  # [N, D, H, W, 1]

    ### blend four views
    weights = mpis_alpha * tf.cumprod(1. - mpis_alpha + 1e-10, axis=1, exclusive=True, reverse=True) # [N, D, H, W, 1]
    rendered_views = tf.reduce_sum(weights * mpis_color, axis=1) # [N, H, W, 3]
    occ_map = tf.reduce_sum(weights, axis=1) # [N, H, W, 1]

    D = mpis.get_shape().as_list()[1]
    disp_buffer = tf.linspace(disp_range[0,0], disp_range[0,1], D) # shape [D]
    disp_buffer = tf.reshape(disp_buffer, [1, D, 1, 1, 1])
    disp_maps = tf.reduce_sum(weights * disp_buffer, axis=1) # shape [N, H, W, 1]

    num_imgs = tf.shape(source_uv)[0]
    distances = tf.sqrt(tf.reduce_sum(tf.square(tf.cast(source_uv - tf.tile(target_uv, [num_imgs, 1]), tf.float32)), axis=1) + 1e-10)
    weights = tf.exp(-distances*(disp_range[0,1] / 2))  # shape [N]
    weights = tf.reshape(weights,[num_imgs, 1, 1, 1])
    final_view = tf.reduce_sum(rendered_views*weights, axis=0, keepdims=True) / (tf.reduce_sum(occ_map*weights, 0, keepdims=True) + 1e-10) # [1, H, W, 3]
    disp_map = tf.reduce_sum(disp_maps*weights, axis=0, keepdims=True) / (tf.reduce_sum(occ_map*weights, 0, keepdims=True) +1e-10) # [1, H, W, 3]

    return final_view, disp_map, disp_maps

def all_in_focal(focal_stacks, mpis_alpha):
    weights = mpis_alpha * tf.cumprod(1. - mpis_alpha + 1e-10, axis=1, exclusive=True, reverse=True) # [N, D, H, W, 1]
    all_in_focal_imgs = tf.reduce_sum(weights * focal_stacks, axis=1) # [N, H, W, 3]
    return all_in_focal_imgs

def views_fusion(rendered_views, occ_map, disp_maps, source_uv, target_uv, disp_range):

    num_imgs = tf.shape(source_uv)[0]
    distances = tf.sqrt(tf.reduce_sum(tf.square(tf.cast(source_uv - tf.tile(target_uv, [num_imgs, 1]), tf.float32)), axis=1) + 1e-10)
    weights = tf.exp(-distances*(disp_range[0,1] / 2))  # shape [N]
    weights = tf.reshape(weights,[num_imgs, 1, 1, 1])
    final_view = tf.reduce_sum(rendered_views*weights, axis=0, keepdims=True) / (tf.reduce_sum(occ_map*weights, 0, keepdims=True) + 1e-10) # [1, H, W, 3]
    disp_map = tf.reduce_sum(disp_maps*weights, axis=0, keepdims=True) / (tf.reduce_sum(occ_map*weights, 0, keepdims=True) +1e-10) # [1, H, W, 3]

    return final_view, disp_map

class Pipeline():

    def __init__(self, net, input_lf, source_imgs, source_uv, target_imgs, target_uv, disp_range, num_depths, train=True):
        self.in_net = net
        self.input_lf = input_lf
        self.source_imgs = source_imgs
        self.source_uv = source_uv
        self.target_imgs = target_imgs
        self.target_uv = target_uv
        self.disp_range = disp_range
        self.num_depths = num_depths
        self.train = train

    def inference(self):
        fs_tl, ref_tl = get_focal_stack(self.source_imgs, self.source_uv, self.disp_range, self.num_depths, ref_idx=0,) # shape [1, D, H, W, 3]
        fs_tr, ref_tr = get_focal_stack(self.source_imgs, self.source_uv, self.disp_range, self.num_depths, ref_idx=1)
        fs_bl, ref_bl = get_focal_stack(self.source_imgs, self.source_uv, self.disp_range, self.num_depths, ref_idx=2)
        fs_br, ref_br = get_focal_stack(self.source_imgs, self.source_uv, self.disp_range, self.num_depths, ref_idx=3)

        fs_all = tf.concat([fs_tl, fs_tr, fs_bl, fs_br], axis=0)
        net_input_tl = tf.concat([fs_tl, ref_tl], axis=-1) # [1, D, H, W, 2*C]
        net_input_tr = tf.concat([fs_tr, ref_tr], axis=-1) # [1, D, H, W, 2*C]
        net_input_bl = tf.concat([fs_bl, ref_bl], axis=-1) # [1, D, H, W, 2*C]
        net_input_br = tf.concat([fs_br, ref_br], axis=-1) # [1, D, H, W, 2*C]

        mpi_tl = self.in_net.get_mpi(net_input_tl, 2) # shape [1,32,256,256,4]
        mpi_tr = self.in_net.get_mpi(net_input_tr, 2) # shape [1,32,256,256,4]
        mpi_bl = self.in_net.get_mpi(net_input_bl, 2) # shape [1,32,256,256,4]
        mpi_br = self.in_net.get_mpi(net_input_br, 2) # shape [1,32,256,256,4]

        mpis = tf.concat([mpi_tl, mpi_tr, mpi_bl, mpi_br], 0) # shape [4, D, H, W, 4]
        final_view, disp_map, disp_maps = render_views(mpis, self.source_uv, self.target_uv, self.disp_range) # shape [4, H, W, 3]
        if self.train == True:
            return final_view, disp_map
        else:
            return final_view, disp_map, disp_maps, fs_all

    def VGG_net19(self, inputs):
        # with tf.device('/gpu:1'):
        with tf.variable_scope('vgg_19',reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d],trainable=False):
                f1 = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                down_f1 = slim.max_pool2d(f1, [2, 2], scope='pool1')
                f2 = slim.repeat(down_f1, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                down_f2 = slim.max_pool2d(f2, [2, 2], scope='pool2')
                f3 = slim.repeat(down_f2, 4, slim.conv2d, 256, [3, 3], scope='conv3')
                down_f3 = slim.max_pool2d(f3, [2, 2], scope='pool3')
                f4 = slim.repeat(down_f3, 4, slim.conv2d, 512, [3, 3], scope='conv4')
                down_f4 = slim.max_pool2d(f4, [2, 2], scope='pool4')
                f5 = slim.repeat(down_f4, 4, slim.conv2d, 512, [3, 3], scope='conv5')
        return [inputs,f1,f2,f3,f4,f5]

    def per_loss(self, real, fake):
        weights = [1,0.385,0.208,0.270,0.179,6.667]
        real_feature_list = self.VGG_net19(real)
        fake_feature_list = self.VGG_net19(fake)
        loss = 0
        for _, (weight, real_fs, fake_fs) in enumerate(zip(weights, real_feature_list, fake_feature_list)):
            loss = loss + weight*tf.losses.absolute_difference(real_fs, fake_fs)
        return loss
    
    def focal_loss(self, real, fake):
        loss = tf.losses.absolute_difference(real, fake)
        return loss

    def gradient_loss(self, real, fake):
        gx_real = real[:,1:,:,:] - real[:,:-1,:,:]
        gy_real = real[:,:,1:,:] - real[:,:,:-1,:]

        gx_fake = fake[:,1:,:,:] - fake[:,:-1,:,:]
        gy_fake = fake[:,:,1:,:] - fake[:,:,:-1,:]

        gx_loss = tf.losses.absolute_difference(gx_real, gx_fake)
        gy_loss = tf.losses.absolute_difference(gy_real, gy_fake)

        return gx_loss + gy_loss

    def loss(self, pre_view):  
        lp = self.per_loss(self.target_imgs, pre_view)

        loss_all = lp
        return loss_all

    def lpips_value(self, pre_view):
        lpips = lpips_tf.lpips(pre_view, self.target_imgs, model='net-lin', net='alex')
        return lpips
