import math
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from torch.utils import data
import augmentations
import utils


class DataHandler(data.Dataset):
    def __init__(self, data_path, config, upsample_scale, device, video_only_flag=False):
        """
        create a DataHandler instance. save the path and the config as parameters in the class
        :param data_path: the absolute path the the video frames
        :param config: configurations for the current run, across and within probabilities etc.
        """
        self.data_path = data_path
        self.config = config
        self.upsample_scale = upsample_scale
        self.device = device

        # shape: frames,height,width,channels
        self.video_tensor = np.asarray(utils.read_seq_from_folder(data_path, config["prefix"], config["dtype"]))
        self.crop_size = self.calc_final_crop_size()
        if video_only_flag:  # Used for eval only - no need to calc and load rest of object
            return

        if config['debug']:
            self.crops_on_video_tensor = np.zeros_like(self.video_tensor)
        self.video_shape = self.video_tensor.shape
        self.blur_flag = self.extract_blur_flag()

        # calc the probability maps used for selecting resized video to take crop from
        self.create_probability_maps()

        self.across_new_z_sample_range = None
        self.shift_range_hor = None
        self.shift_range_ver = None

        self.calc_possible_resize_ranges()
        self.printed_across_fail_warning = False  # To print warning only once

    def extract_blur_flag(self):
        hr_lr_relation = self.config['data']['params']['hr_lr_relation']
        if hr_lr_relation == 0:  # full exposure
            blur_flag = True
        elif hr_lr_relation == 1:  # delta exposure
            blur_flag = False
        return blur_flag

    def __len__(self):
        # return self.video_shape[0]
        return self.config['num_iter_per_epoch']

    def __getitem__(self, item):
        hr_gt, lr = self.get_training_couple()
        permutation_np_to_torch = (3, 0, 1, 2)  # move channels to first
        hr_gt = np.transpose(hr_gt, permutation_np_to_torch)
        lr = np.transpose(lr, permutation_np_to_torch)

        return hr_gt.astype('float32'), lr.astype('float32')

    def calc_final_crop_size(self):
        final_crop_spatial = self.config['data']['params']['augmentation_params']['crop_sizes'][
            'crop_size_spatial']
        final_crop_temporal = self.config['data']['params']['augmentation_params']['crop_sizes'][
            'crop_size_temporal']
        return [final_crop_temporal, final_crop_spatial, final_crop_spatial, 3]

    def create_probability_maps(self):
        self.spatial_resize_options = self.config["data"]["params"]["augmentation_params"]["spatial_resize_options"]
        self.temporal_jump_options = self.config["data"]["params"]["augmentation_params"]["temporal_jump_options"]
        self.spatial_resize_probabilities = np.array(
            [x ** 2 for x in self.spatial_resize_options])  # prob relative to volume
        self.spatial_resize_probabilities = self.spatial_resize_probabilities / np.sum(
            self.spatial_resize_probabilities)
        self.temporal_jump_probabilities = [1 / x for x in self.temporal_jump_options]  # prob relative to volume
        self.temporal_jump_probabilities = self.temporal_jump_probabilities / np.sum(
            self.temporal_jump_probabilities)
        self.resize_probability_maps = {}

        # For runtime by using utils.tensor_3d_choice, flattens resize_probability_map to a 1d vector for probability of each frame
        self.frame_probability_maps = {}

        for sp_idx, sp_val in enumerate(self.spatial_resize_options):
            resize_scale = [1, sp_val, sp_val, 1]
            spatially_resized_tensor = augmentations.resize_tensor(self.video_tensor, "cubic", resize_scale,
                                                                   device=self.device)
            for temp_idx, temp_val in enumerate(self.temporal_jump_options):
                space_temp_resized_tensor = augmentations.blur_sample_tensor(spatially_resized_tensor, sample_axis=0,
                                                                             sample_jump=temp_val,
                                                                             blur_flag=self.blur_flag)
                if np.any(np.greater(self.crop_size, space_temp_resized_tensor.shape)):
                    continue  # resized tensor too small for crops
                print(f'added sp_val: {sp_val}, temp_val: {temp_val} to probability maps')
                video_gradients_3d = self.calc_gradient_magnitude(space_temp_resized_tensor, 'grad')

                crop_filter_odd_size = [2 * (x // 2) + 1 for x in self.crop_size[
                                                                  0:3]]  # To make filter odd size - needed for placing origin at top-left
                crop_filter = np.zeros(crop_filter_odd_size)
                crop_filter[0:self.crop_size[0], 0:self.crop_size[1], 0:self.crop_size[2]] = np.ones(
                    self.crop_size[0:3])
                crop_filter = crop_filter / np.sum(crop_filter)

                prob_map_method = 'cutoff'  # 'cutoff' or 'no_cutoff'

                if prob_map_method == 'cutoff':
                    cutoff_percentile = self.config['data']['params']['gradient_percentile']
                    video_gradients_3d = video_gradients_3d > np.percentile(video_gradients_3d, cutoff_percentile)
                    video_gradients_3d = video_gradients_3d.astype(float)

                #correlate with an averaging filter to calc avg gradient in resulting crop
                crop_probability_map = np.abs(
                    ndimage.filters.correlate(video_gradients_3d, crop_filter, mode='constant', cval=0.0,
                                              origin=[-int((x - 1) / 2) for x in list(
                                                  crop_filter.shape)]))  # shift center of filter to TL, since we use the prob map for TL

                #to make sure no negative probabilities
                crop_probability_map[np.where(crop_probability_map < 0)] = 0

                # zero elements that would crop outside the video
                crop_probability_map_without_edges = np.zeros(crop_probability_map.shape)
                crop_probability_map_without_edges[0:crop_probability_map.shape[0] - self.crop_size[0] + 1,
                0:crop_probability_map.shape[1] - self.crop_size[1] + 1,
                0:crop_probability_map.shape[2] - self.crop_size[2] + 1] = crop_probability_map[
                                                                           0:crop_probability_map.shape[0] -
                                                                             self.crop_size[0] + 1,
                                                                           0:crop_probability_map.shape[1] -
                                                                             self.crop_size[1] + 1,
                                                                           0:crop_probability_map.shape[2] -
                                                                             self.crop_size[2] + 1]

                #place in the dictionary after normalizing
                self.resize_probability_maps[(sp_val, temp_val)] = crop_probability_map_without_edges / np.sum(
                    crop_probability_map_without_edges)

                # For runtime by using utils.tensor_3d_choice, sums on all but frames
                self.frame_probability_maps[(sp_val, temp_val)] = np.sum(self.resize_probability_maps[(sp_val, temp_val)],axis=(1,2))


    def calc_possible_resize_ranges(self):
        """
        calculates the possible resize ranges for the different augmentations.
        Tries to use the ranges in the config, but limits to the sizes possible by the training video to be cropped.
        Also prints a "warning" of the resizes used instead of those intended, for debugging purposes.
        The 0.99/1.01 factors inside are to ensure we won't go overboard
        """
        final_crop_spatial = self.config['data']['params']['augmentation_params']['crop_sizes'][
            'crop_size_spatial']
        final_crop_temporal = self.config['data']['params']['augmentation_params']['crop_sizes'][
            'crop_size_temporal']

        # Find for ACROSS
        # Here need to take into account that after the resize may still need to sample the new_z, so we start with that.

        across_new_z_max_sample_possible = min((math.floor(self.video_shape[1] - 1) / final_crop_temporal),
                                               (math.floor(self.video_shape[2] - 1) / final_crop_temporal))
        # how much we can sample and remain with temporal crop size
        conf_across_new_z_sample_range = self.config['data']['params']['augmentation_params']['across'][
            'new_z_sample_range']
        if across_new_z_max_sample_possible < 1:
            assert False, f'assertion error in calc_possible_resize_ranges: spatial axes not large enough for temporal crop.'
        self.across_new_z_sample_range = [conf_across_new_z_sample_range[0],
                                          min(across_new_z_max_sample_possible,
                                              conf_across_new_z_sample_range[1])]  # The min of wanted and possible

        # Find for SHIFT
        # Shift has two limits-resize and shift for each frame. we'll first limit the frame shift range
        shift_max_each_frame_ver = math.floor(
            (self.video_shape[1] - final_crop_spatial - 1) / final_crop_temporal)
        shift_max_each_frame_hor = math.floor(
            (self.video_shape[2] - final_crop_spatial - 1) / final_crop_temporal)
        if shift_max_each_frame_ver < 1 or shift_max_each_frame_hor < 1:
            assert False, f'assertion error in calc_possible_resize_ranges: shift is not possible even with one pixel shift per frame.'
        conf_shift_range_ver = self.config['data']['params']['augmentation_params']['shift']['range_ver']
        conf_shift_range_hor = self.config['data']['params']['augmentation_params']['shift']['range_hor']

        if conf_shift_range_ver[1] > shift_max_each_frame_ver:  # wanted shift too much per frame, limit to max possible
            self.shift_range_ver = [-shift_max_each_frame_ver, shift_max_each_frame_ver]
            print(
                f'wanted ver shift per frame in SHIFT too large. Instead of wanted {conf_shift_range_ver}, took {self.shift_range_ver}')
        else:
            self.shift_range_ver = conf_shift_range_ver
        if conf_shift_range_hor[1] > shift_max_each_frame_hor:  # wanted shift too much per frame, limit to max possible
            self.shift_range_hor = [-shift_max_each_frame_hor, shift_max_each_frame_hor]
            print(
                f'wanted hor shift per frame in SHIFT too large. Instead of wanted {conf_shift_range_hor}, took {self.shift_range_hor}')
        else:
            self.shift_range_hor = conf_shift_range_hor

    def get_training_couple(self):
        """
        Draws augmentation by params in config
        and returns
        :return: tuple (lr,hr)
        """
        # step1: Draw augmentation type from probabilities in config. 0-within, 1-across, 2-shift
        augmentation_type = self.augmentation_type()

        # step2: Draw crop accordingly
        if augmentation_type == 'within':
            training_couple = self.create_within_training_couple()
        elif augmentation_type == 'across':
            training_couple = self.create_across_training_couple()
        elif augmentation_type == 'shift':
            training_couple = self.create_shift_training_couple()
        else:
            assert False, f"assertion error in get_training_couple, type={augmentation_type}"

        return training_couple

    @staticmethod
    def calc_gradient_magnitude(input_tensor, grad_type='3d', temporal_weight=1, cutoff_max_grad=0.2):
        """
        takes a crop (tensor) and calculates the gradient magnitude (space, or space-time), element-wise.
        Assumes tensor is in order: frames, vertical, horizontal, channels .
        :param drawn_crop: np array - tensor to calculate the gradient magnitude on.
        :param grad_type: '2d' - gradient magnitude only spatial, no temporal (Sobel 2d)
                         '3d' - gradient magnitude of both spatial and temporal (Sobel 3d)
        :return: np array, size as input_tensor, with gradient magnitude on each pixel
        """

        def sobel_filters(in_tensor, grad_type='grad', cutoff_max_grad=cutoff_max_grad):
            """
            helper function for calc_gradient_magnitude
            define and apply gradient, 2d or 3d
            :param tensor: np array, image or video to apply gradients
            :param flag: direction of gradients, 'space' - only on a single frame, 'space-time' - sobel on frame and [-1,1] on temporal dimension
            :return: gradient map
            """
            gray_tensor = in_tensor[:, :, :, 0] * 0.2125 + in_tensor[:, :, :, 1] * 0.7154 + in_tensor[:, :, :,
                                                                                            2] * 0.0721
            Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
            Kx_3d = np.expand_dims(Kx, axis=0)
            Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
            Ky_3d = np.expand_dims(Ky, axis=0)

            Ix = np.clip(ndimage.filters.convolve(gray_tensor, Kx_3d), a_min=-cutoff_max_grad, a_max=cutoff_max_grad)
            Iy = ndimage.filters.convolve(gray_tensor, Ky_3d)

            Ix2 = np.square(Ix)
            Iy2 = np.square(Iy)

            if grad_type == '3d':
                Kz_3d = np.transpose(Kx_3d, (2, 1, 0))
                Iz = ndimage.filters.convolve(gray_tensor, Kz_3d)
                Iz2 = np.square(Iz)

                return np.sqrt(Ix2 + Iy2 + temporal_weight * Iz2)
            elif grad_type == '2d':  # only on spatial dimensions
                return np.sqrt(Ix2 + Iy2)
            else:
                assert False, f'assertion error in calc_gradient_magnitude, flag not valid, is {grad_type}'

        def grad_prob_map(in_tensor):
            """
            helper function for calc_gradient_magnitude, based on orig code
            :param input_tensor: np array, rgb video to apply gradients
            :return: gradient probabilities
            """
            gray_tensor = in_tensor[:, :, :, 0] * 0.2125 + in_tensor[:, :, :, 1] * 0.7154 + in_tensor[:, :, :,
                                                                                            2] * 0.0721
            gx, gy, gt = np.gradient(gray_tensor)
            gsum = np.abs(gx) + np.abs(gy) + np.abs(gt)
            return gsum

        # check that the the drawn tensor in now empty
        assert input_tensor is not None

        if grad_type == '2d':
            sobel_output = sobel_filters(input_tensor, grad_type, cutoff_max_grad=cutoff_max_grad)
        elif grad_type == '3d':
            sobel_output = sobel_filters(input_tensor, grad_type, cutoff_max_grad=cutoff_max_grad)
        elif grad_type == 'grad':
            sobel_output = grad_prob_map(input_tensor)
        else:
            assert False, f'assertion error in calc_gradient_magnitude, var_type not valid, is {grad_type}'

        return sobel_output

    def augmentation_type(self):
        """
        draw augmentation type by probabilities in config
        :return: string of augmentation type
        """
        augment_dict = {0: 'within', 1: 'across', 2: 'shift'}
        within_prob = self.config['data']['params']['augmentation_params']['within']['probability']
        across_prob = self.config['data']['params']['augmentation_params']['across']['probability']
        shift_prob = self.config['data']['params']['augmentation_params']['shift']['probability']
        probabilities = [within_prob, across_prob, shift_prob]
        probabilities = [i / sum(probabilities) for i in probabilities]  # normalize probabilities
        augmentation_num = np.random.choice(3, p=probabilities)
        return augment_dict[augmentation_num]

    def mark_crop_on_video(self, corner, crop_size):
        assert self.video_tensor.shape[0] >= corner[0] + crop_size[0] and self.video_tensor.shape[
            1] >= corner[1] + crop_size[1] and self.video_tensor.shape[2] >= corner[2] + \
               crop_size[2], f'assertion error in mark_crop_on_video. crop outside video'

        cur_vals = self.crops_on_video_tensor[corner[0]:corner[0] + crop_size[0], corner[1]:corner[1] + crop_size[1],
                   corner[2]:corner[2] + crop_size[2], :]
        cur_vals = cur_vals + np.ones_like(cur_vals)
        self.crops_on_video_tensor[corner[0]:corner[0] + crop_size[0], corner[1]:corner[1] + crop_size[1],
        corner[2]:corner[2] + crop_size[2], :] = cur_vals

    def across_dims_decide_new_z_axis(self):
        prob_ver_new_z = self.config['data']['params']['augmentation_params']['across']['prob_ver_new_z']
        prob_hor_new_z = self.config['data']['params']['augmentation_params']['across']['prob_hor_new_z']
        probabilities = [prob_ver_new_z, prob_hor_new_z]
        probabilities = [i / sum(probabilities) for i in probabilities]  # normalize probabilities
        new_z_dict = {1: 'ver', 2: 'hor'}  # 1 and 2 for the axes
        new_z_num = np.random.choice([1, 2], p=probabilities)
        return new_z_dict[new_z_num], new_z_num

    def create_across_training_couple(self):
        """
        create across hr-lr pair with parameters from config
        Assumes internally that order is frames, height, width, channels

        :return: tuple of np tensors of above order, (hr,lr)
        """
        iter = 0
        while (1):  # loop for finding doable crop size. Prints when fails each time but doesn't fail run
            iter = iter + 1
            if iter > 200:
                if not self.printed_across_fail_warning:
                    print(
                        'Notice!\nNotice!\nNotice!\nfailed to create Across training example. Probably requesting larger spatial crop than there are frames.\nReturning Within training example instead.\nNotice!\nNotice!\nNotice!')
                    self.printed_across_fail_warning = True
                return self.create_within_training_couple()
            # step 1: decide which axis is the new z: vertical or horizontal
            new_z, new_z_num = self.across_dims_decide_new_z_axis()  # 'ver' or 'hor'
            # step 2: draw the needed crop sizes in each axis
            crop_size_new_spatial = self.crop_size[1]

            spatial_resize = np.random.choice(self.spatial_resize_options, p=self.spatial_resize_probabilities)

            crop_size_new_z = self.crop_size[0]
            new_z_sample_range = self.across_new_z_sample_range
            if new_z_sample_range[0] == new_z_sample_range[1]:
                new_z_sample = new_z_sample_range[0]
            else:
                new_z_sample = np.random.randint(low=new_z_sample_range[0], high=new_z_sample_range[1] + 1)

            # The final buffered sizes:
            orig_spatial_now_spatial_size = math.ceil(
                crop_size_new_spatial * (1 / spatial_resize))  # The spatial that remains spatial. Undergoes only resize
            orig_spatial_now_z_size = math.ceil(crop_size_new_z * (
                    1 / spatial_resize) * new_z_sample)  # The spatial that is now z. Undergoes both resize and sampling
            orig_temp_now_spatial_size = crop_size_new_spatial  # Axis was originally frames, now spatial. No resize or sampling

            if new_z == 'ver':
                crop_size_orig_layout = [orig_temp_now_spatial_size, orig_spatial_now_z_size,
                                         orig_spatial_now_spatial_size]
                permutation = (1, 0, 2, 3)
            elif new_z == 'hor':
                crop_size_orig_layout = [orig_temp_now_spatial_size, orig_spatial_now_spatial_size,
                                         orig_spatial_now_z_size]
                permutation = (2, 1, 0, 3)
            else:
                assert False, f'assertion error in create_across_training_couple, new_z is {new_z}'

            try:
                # assert resize is in dict
                assert (spatial_resize, 1) in self.resize_probability_maps.keys()
                # assert crop size is doable.
                assert self.video_shape[0] >= crop_size_orig_layout[0] and self.video_shape[1] >= crop_size_orig_layout[
                    1] and self.video_shape[2] >= crop_size_orig_layout[2], \
                    f"assertion error in create_across_training_couple: crop larger than video. Video: {self.video_shape}, new_z: {new_z}, spatial_resize:{spatial_resize}, new_z_sample:{new_z_sample}, Crop: {crop_size_orig_layout}"
            except:
                # already printed details and error, "try again"
                continue
            break

        resized_prob_map = self.resize_probability_maps[(spatial_resize, 1)]
        frame_probability_map = self.frame_probability_maps[(spatial_resize, 1)]
        # step 3: crop, make sure of gradients, augment
        iterator = 0
        while 1:
            iterator = iterator + 1
            assert iterator < 1000, f'assertion error in draw_tensor_crop. iterator for drawing crop > 1000'

            patch_choice_resize = utils.tensor_3d_choice(resized_prob_map, frame_probability_map)

            # "un-resize" to get patch in original video. Order is temporal,spatial,spatial
            patch_choice_orig = np.array([int(patch_choice_resize[0] * 1),
                                          int(patch_choice_resize[1] * (1 / spatial_resize)),
                                          int(patch_choice_resize[2] * (1 / spatial_resize))])
            # To avoid cropping outside tensor:
            orig_patch_size = crop_size_orig_layout
            patch_choice_orig = np.array([min(self.video_shape[0] - crop_size_orig_layout[0], patch_choice_orig[0]),
                                          min(self.video_shape[1] - crop_size_orig_layout[1], patch_choice_orig[1]),
                                          min(self.video_shape[2] - crop_size_orig_layout[2], patch_choice_orig[2])])

            # Check fits in tensor.
            if self.video_tensor.shape[0] >= patch_choice_orig[0] + orig_patch_size[0] and self.video_tensor.shape[
                1] >= patch_choice_orig[1] + orig_patch_size[1] and self.video_tensor.shape[2] >= patch_choice_orig[2] + \
                    orig_patch_size[2]:
                break

        drawn_crop = self.video_tensor[patch_choice_orig[0]:patch_choice_orig[0] + orig_patch_size[0],
                     patch_choice_orig[1]:patch_choice_orig[1] + orig_patch_size[1],
                     patch_choice_orig[2]:patch_choice_orig[2] + orig_patch_size[2], :]
        if self.config['debug']:
            self.mark_crop_on_video(patch_choice_orig, orig_patch_size)
        # Here resize BEFORE sample, as they are same axis. Resize is on ORIGINAL spatial
        resize_scale = np.array([1., spatial_resize, spatial_resize, 1.], dtype=np.float32)
        drawn_crop = augmentations.resize_tensor(drawn_crop, "cubic", resize_scale, device=self.device)

        drawn_crop = augmentations.blur_sample_tensor(drawn_crop, sample_axis=new_z_num, sample_jump=new_z_sample,
                                                      blur_flag=self.blur_flag)

        # step 4: permute, ensure size is right
        drawn_crop = np.transpose(drawn_crop, permutation)
        hr_tensor = drawn_crop[0:crop_size_new_z, 0:crop_size_new_spatial, 0:crop_size_new_spatial,
                    :]  # Since may be larger due to ceil, etc.
        assert hr_tensor.shape[0] == crop_size_new_z and hr_tensor.shape[1] == hr_tensor.shape[
            2] == crop_size_new_spatial, \
            f'assertion error in create_across_training_couple - hr size is {hr_tensor.shape}, not {[crop_size_new_z, crop_size_new_spatial, crop_size_new_spatial]}'

        flip_prob = self.config['data']['params']['augmentation_params']['across']['flip_prob']
        rotation_prob = self.config['data']['params']['augmentation_params']['across']['rotation_prob']
        z_flip_prob = self.config['data']['params']['augmentation_params']['across']['new_z_flip_prob']
        hr_tensor = augmentations.flip_rotate_tensor(hr_tensor, flip_prob, rotation_prob, z_flip_prob)

        lr_tensor = self.hr_to_lr(hr_tensor, self.upsample_scale)
        return (hr_tensor, lr_tensor)

    def create_within_training_couple(self):
        """
        create within hr-lr pair with parameters from config
        Assumes internally that order is frames, height, width, channels

        :return: tuple of np tensors of above order, (hr,lr)
        """
        iterator = 0
        while 1:
            iterator = iterator + 1
            assert iterator < 1000, f'assertion error in create_within_training_couple. iterator for crop size > 1000'

            drawn_spatial_resize = np.random.choice(self.spatial_resize_options, p=self.spatial_resize_probabilities)
            drawn_temporal_jump = np.random.choice(self.temporal_jump_options, p=self.temporal_jump_probabilities)
            # Check drawn tensor large enough for crop size. Needed especially for AcrossDims
            if (drawn_spatial_resize, drawn_temporal_jump) in self.resize_probability_maps.keys():
                resized_tensor_shape = self.resize_probability_maps[(drawn_spatial_resize, drawn_temporal_jump)].shape
                if np.all(np.greater_equal(resized_tensor_shape, self.crop_size[0:3])):  # no channels dim
                    break
        resized_prob_map = self.resize_probability_maps[(drawn_spatial_resize, drawn_temporal_jump)]
        frame_probability_map = self.frame_probability_maps[(drawn_spatial_resize, drawn_temporal_jump)]

        patch_choice_resize = utils.tensor_3d_choice(resized_prob_map, frame_probability_map)

        # "un-resize" to get patch in original video. Order is temporal,spatial,spatial
        patch_choice_orig = np.array([int(patch_choice_resize[0] * drawn_temporal_jump),
                                      int(patch_choice_resize[1] * (1 / drawn_spatial_resize)),
                                      int(patch_choice_resize[2] * (1 / drawn_spatial_resize))], dtype='int32')
        # To enable all subsets, need to "overcome" the sampling in temporal axis:
        patch_choice_orig[0] = patch_choice_orig[0] + np.random.randint(0, drawn_temporal_jump, dtype='int32')
        # To avoid cropping outside tensor:
        # calc size of patch needed from original tensor, so that after resize will be in wanted size, [t,h,w,c]
        orig_patch_size = np.array(np.ceil(np.multiply(self.crop_size,
                                                       [drawn_temporal_jump, (1 / drawn_spatial_resize),
                                                        (1 / drawn_spatial_resize), 1])), dtype='int32')
        # just to be sure:
        patch_choice_orig = np.array([min(self.video_shape[0] - orig_patch_size[0], patch_choice_orig[0]),
                                      min(self.video_shape[1] - orig_patch_size[1], patch_choice_orig[1]),
                                      min(self.video_shape[2] - orig_patch_size[2], patch_choice_orig[2])],
                                     dtype='int32')

        hr_tensor = self.video_tensor[patch_choice_orig[0]:patch_choice_orig[0] + orig_patch_size[0],
                    patch_choice_orig[1]:patch_choice_orig[1] + orig_patch_size[1],
                    patch_choice_orig[2]:patch_choice_orig[2] + orig_patch_size[2], :]
        # apply temporal jump
        hr_tensor = augmentations.blur_sample_tensor(hr_tensor, sample_axis=0,
                                                     sample_jump=drawn_temporal_jump,
                                                     blur_flag=self.blur_flag)
        # apply spatial resize
        hr_tensor = augmentations.resize_tensor(hr_tensor, "cubic",
                                                [1.0, drawn_spatial_resize, drawn_spatial_resize, 1.0], device=self.device)
        # ceil may cause not expected size:
        hr_tensor = hr_tensor[0:self.crop_size[0], 0:self.crop_size[1], 0:self.crop_size[2], :]

        if self.config['debug']:
            self.mark_crop_on_video(patch_choice_orig, orig_patch_size)

        assert np.all(hr_tensor.shape == np.array(
            self.crop_size)), f'assertion error in create_within_training_couple - hr size is {hr_tensor.shape}, not {self.crop_size}'

        flip_prob = self.config['data']['params']['augmentation_params']['within']['flip_prob']
        rotation_prob = self.config['data']['params']['augmentation_params']['within']['rotation_prob']
        z_flip_prob = self.config['data']['params']['augmentation_params']['within']['z_flip_prob']
        hr_tensor = augmentations.flip_rotate_tensor(hr_tensor, flip_prob, rotation_prob, z_flip_prob)
        lr_tensor = self.hr_to_lr(hr_tensor, self.upsample_scale)

        return (hr_tensor, lr_tensor)

    def create_shift_training_couple(self):
        """
        create shift hr-lr pair with parameters from config
        Assumes internally that order is frames, height, width, channels

        :return: tuple of np tensors of above order, (hr,lr)
        """
        shift_hor, shift_ver = self.draw_shift_values()
        valid_flag = self.config['data']['params']['augmentation_params']['shift']['valid']

        prob_for_across = self.config['data']['params']['augmentation_params']['shift']['prob_for_across']
        across_flag = np.random.choice([0, 1], p=[1 - prob_for_across, prob_for_across])
        if not across_flag:  # use as within
            crop_size_hor = self.crop_size[1]
            crop_size_ver = self.crop_size[1]
            crop_size_temporal = self.crop_size[0]
        else:  # use as across. Only enable shift==1
            crop_size_temporal = self.crop_size[1]
            new_z, new_z_num = self.across_dims_decide_new_z_axis()
            if new_z == 'ver':
                crop_size_hor = self.crop_size[1]  # still spatial
                crop_size_ver = self.crop_size[0]  # new_z
                permutation = (1, 0, 2, 3)
                shift_ver = np.sign(shift_ver)  # 1 or -1
                shift_hor = 0
            else:  # new_z == 'hor'
                crop_size_hor = self.crop_size[0]  # new_z
                crop_size_ver = self.crop_size[1]  # still spatial
                permutation = (2, 1, 0, 3)
                shift_ver = 0
                shift_hor = np.sign(shift_hor)  # 1 or -1
        iterator = 0
        while 1:
            iterator = iterator + 1
            assert iterator < 1000, f'assertion error in create_within_training_couple. iterator for crop size > 1000'

            drawn_spatial_resize = 1.0  # shift only on largest spatial scale

            # Check drawn tensor large enough for crop size. Needed especially for AcrossDims
            if (drawn_spatial_resize, 1) in self.resize_probability_maps.keys():  # no need for temporal jump
                resized_tensor_shape = self.resize_probability_maps[(drawn_spatial_resize, 1)].shape
                # Add buffer for resize + valid when needed
                crop_size_hor_buff = math.ceil((1 / drawn_spatial_resize) * (
                        crop_size_hor + valid_flag * math.ceil(crop_size_temporal) * max(abs(shift_ver),
                                                                                         abs(shift_hor))))
                crop_size_ver_buff = math.ceil((1 / drawn_spatial_resize) * (
                        crop_size_ver + valid_flag * math.ceil(crop_size_temporal) * max(abs(shift_ver),
                                                                                         abs(shift_hor))))
                crop_size_buffed = [1, crop_size_ver_buff, crop_size_hor_buff]

                if np.all(np.greater_equal(resized_tensor_shape, crop_size_buffed)):  # no channels dim
                    break
        resized_prob_map = self.resize_probability_maps[(drawn_spatial_resize, 1)]
        frame_probability_map = self.frame_probability_maps[(drawn_spatial_resize, 1)]

        while 1:
            patch_choice_resize = utils.tensor_3d_choice(resized_prob_map, frame_probability_map)

            # "un-resize" to get patch in original video. Order is temporal,spatial,spatial
            patch_choice_orig = np.array([int(patch_choice_resize[0] * 1),
                                          int(patch_choice_resize[1] * (1 / drawn_spatial_resize)),
                                          int(patch_choice_resize[2] * (1 / drawn_spatial_resize))])
            # Check fits in tensor. Needed due to buffer taken
            if self.video_tensor.shape[0] >= patch_choice_orig[0] + crop_size_buffed[0] and self.video_tensor.shape[
                1] >= patch_choice_orig[1] + crop_size_buffed[1] and self.video_tensor.shape[2] >= patch_choice_orig[
                2] + \
                    crop_size_buffed[2]:
                break

        drawn_frame_crop = self.video_tensor[patch_choice_orig[0]:patch_choice_orig[0] + crop_size_buffed[0],
                           patch_choice_orig[1]:patch_choice_orig[1] + crop_size_buffed[1],
                           patch_choice_orig[2]:patch_choice_orig[2] + crop_size_buffed[2], :]

        drawn_frame_crop = np.squeeze(drawn_frame_crop)  # Since temp_len=1
        resize_scale = np.array([drawn_spatial_resize, drawn_spatial_resize, 1.], dtype=np.float32)
        drawn_frame_crop = augmentations.resize_tensor(drawn_frame_crop, "cubic", resize_scale, device=self.device)

        shift_tensor = augmentations.shift_frame(drawn_frame_crop, shift_ver, shift_hor, crop_size_temporal,
                                                 valid_flag=valid_flag)
        hr_tensor = shift_tensor[:, 0:crop_size_ver, 0:crop_size_hor, :]  # Since shift_frame may be larger

        if self.config['debug']:
            self.mark_crop_on_video(patch_choice_orig, crop_size_buffed)

        flip_prob = self.config['data']['params']['augmentation_params']['shift']['flip_prob']
        rotation_prob = self.config['data']['params']['augmentation_params']['shift']['rotation_prob']
        if across_flag:
            rotation_prob = 0.0  # to not mix hor and ver. as one will be temporal
        z_flip_prob = self.config['data']['params']['augmentation_params']['shift']['z_flip_prob']
        hr_tensor = augmentations.flip_rotate_tensor(hr_tensor, flip_prob, rotation_prob, z_flip_prob)

        if across_flag:  # need to permute, to use as across
            hr_tensor = np.transpose(hr_tensor, permutation)

        # ceil may cause not expected size:
        hr_tensor = hr_tensor[0:self.crop_size[0], 0:self.crop_size[1], 0:self.crop_size[2], :]

        assert np.all(hr_tensor.shape == np.array(
            self.crop_size)), f'assertion error in create_shift_training_couple - hr size is {hr_tensor.shape}, not {self.crop_size}'
        lr_tensor = self.hr_to_lr(hr_tensor, self.upsample_scale)

        return (hr_tensor, lr_tensor)

    def draw_shift_values(self):
        shift_range_ver = self.shift_range_ver
        shift_range_hor = self.shift_range_hor
        assert self.shift_range_ver[0] != 0 or self.shift_range_ver[1] != 0 or self.shift_range_hor[0] != 0 or \
               self.shift_range_hor[1] != 0, f'shift has all ranges as 0'
        entire_pixels = self.config['data']['params']['augmentation_params']['shift']['entire_pixels']
        shift_ver = 0
        shift_hor = 0
        while (shift_ver == 0 and shift_hor == 0):  # keep drawing until get a non-zero-shift
            if entire_pixels:  # do not enable sub-pixel shifts
                shift_ver = np.random.randint(low=shift_range_ver[0], high=shift_range_ver[1] + 1)
                shift_hor = np.random.randint(low=shift_range_hor[0], high=shift_range_hor[1] + 1)
            else:  # enable sub-pixel shifts
                shift_ver = np.random.uniform(low=shift_range_ver[0], high=shift_range_ver[1])
                shift_hor = np.random.uniform(low=shift_range_hor[0], high=shift_range_hor[1])
        return shift_hor, shift_ver

    def hr_to_lr(self, hr_tensor, jump=2):
        """
        take a HR tensor and return its LR tensor, in the manner determined in config.
        :param hr_tensor: np array
        :return: none, plots the frames or tensors
        """
        # check that the HR tensor is [F,H,W,C]
        assert len(
            hr_tensor.shape) == 4, f'assert error in hr_to_lr.HR tensor shape len is {len(hr_tensor.shape)},not 4'

        lr_tensor = augmentations.blur_sample_tensor(hr_tensor, 0, sample_jump=jump, blur_flag=self.blur_flag)
        return lr_tensor

    @staticmethod
    def visualize_tuple(hr_lr_tuple, save_to_file=True, save_path='./results/imgs'):
        """
        take a tensor and its low resolution version (lr) and show them side-by-side
        :param hr_lr_tuple: (hr,lr) tuple of np arrays
        :param name: save folder name (selected randomly to allow saving seq.)
        :return: none, plots the frames or tensors
        """

        hr_tensor = hr_lr_tuple[0]
        lr_tensor = hr_lr_tuple[1]
        subsample_ratio = hr_tensor.shape[0] // lr_tensor.shape[0]
        if save_to_file:
            idx = 0
            while (1):
                folder_name = os.path.join(save_path, str(idx))
                if not os.path.isdir(folder_name):
                    break
                idx = idx + 1
            os.makedirs(folder_name, exist_ok=True)

        for i in range(lr_tensor.shape[0]):
            plt.figure(1)
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(lr_tensor[i, :])
            plt.title('LR tensor')
            for j in range(subsample_ratio):
                plt.subplot(1, 2, 2)
                plt.imshow(hr_tensor[subsample_ratio * i + j, :])
                plt.title('HR tensor')
                plt.draw()
                # plt.pause(0.05)

                if save_to_file:
                    plt.savefig(os.path.join(folder_name, f'{subsample_ratio * i + j}.png'))
