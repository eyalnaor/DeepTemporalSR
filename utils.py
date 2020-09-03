import glob
import json
import os
import shutil
import time
import traceback
from multiprocessing import Process, Queue
from queue import Empty
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import numpy as np
import torch_resizer
import torch
from PIL import Image
from imresize import imresize
from argparse import ArgumentParser
from simple_backprojection import space_time_backprojection, temporal_backprojection_np

# ---------------------------------------------------------------------------------------------------
# main sub-functions
# ---------------------------------------------------------------------------------------------------
def create_parser():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='path to json config file', default='config.json')
    parser.add_argument('-t', '--tag', type=str, help='optional tag to override config', default=None)
    parser.add_argument('-d', '--data', type=str, help='optional data path to override config', default=None)
    parser.add_argument('-e', '--eval', type=bool, help='Requires checkpoint - only run evaluation', default=None)
    parser.add_argument('-ck', '--checkpoint', type=str, help='path to checkpoint - fine tune from given weights',
                        default=None)
    parser.add_argument('-n', '--network', type=str, help='optional network type to override config', default=None)
    parser.add_argument('-ep', '--epochs', type=int, help='optional epoch num to override config', default=None)
    parser.add_argument('-r', '--gradcutoff', type=str, help='optional gradient cutoff to override config',
                        default=None)
    parser.add_argument('-p', '--spatialcrop', type=int, help='optional spatial crop size to override config',
                        default=None)
    parser.add_argument('-pm', '--spatialmask', type=int, help='num of pixels to ignore on all sides of crop',
                        default=None)
    parser.add_argument('-m', '--temporalcrop', type=int, help='optional temporal crop size to override config',
                        default=None)
    parser.add_argument('-mm', '--temporalmask', type=int, help='num of frames to ignore on start and end of crop',
                        default=None)
    parser.add_argument('-w', '--withinprob', type=float, help='within augmentation probability', default=None)
    parser.add_argument('-a', '--acrossprob', type=float, help='across augmentation probability', default=None)
    parser.add_argument('-l', '--loss', type=int, help='loss type: 0-L1, 1-MSE, 2-lap1', default=None)
    parser.add_argument('-i', '--initiallr', type=float, help='optional initial LR to override config', default=None)
    parser.add_argument('-o', '--workingdir', type=str, help='optional working dir to override config', default=None)
    return parser


def BP_wrapper(config, cumulative_scale, cumulative_spatial_scales, cur_data_path, cur_spatial_scale, output,
               scale_ind, device):
    print('*************')
    print(f'Entering BP on temporal scale {cumulative_scale}, and spatial scale from {cumulative_spatial_scales[scale_ind]} to {cumulative_spatial_scales[scale_ind + 1]}')
    print('*************')
    lfr_hr_path = os.path.join(config['trainer']['working_dir'],
                               f'T1S{cumulative_spatial_scales[scale_ind + 1]}')  # The lfr_hr is one spatial scale AHEAD of current output
    lfr_hr = np.asarray(read_seq_from_folder(lfr_hr_path, config["prefix"], config["dtype"]))
    permutation_for_bp = (1, 2, 0, 3)  # permute to h,w,t,c, as that is what backprojection expects
    lfr_hr = np.transpose(lfr_hr, permutation_for_bp)
    output = np.transpose(output, permutation_for_bp)

    assert ((output.shape[2] / lfr_hr.shape[2]) % 1.0 == 0)
    scale = int(output.shape[2] / lfr_hr.shape[2])
    hfr_chunk = min(scale*(20//scale), output.shape[2]) #max num of frames to take from hfr
    assert ((hfr_chunk/scale) % 1.0 == 0)
    lfr_chunk = int(hfr_chunk/scale)

    hfr_start_frames = np.arange(0, output.shape[2], hfr_chunk)
    hfr_start_frames[-1] = output.shape[2] - hfr_chunk  # For final crop
    lfr_start_frames = [x//scale for x in hfr_start_frames]

    BP_output_shape = [lfr_hr.shape[0], lfr_hr.shape[1], output.shape[2], output.shape[3]]
    for index, hfr_frame_start in enumerate(hfr_start_frames):
        print(f'BP: HFR frame start:{hfr_frame_start} out of {output.shape[2]}')
        lfr_frame_start = lfr_start_frames[index]
        lfr_hr_segment=lfr_hr[:,:,lfr_frame_start:lfr_frame_start+lfr_chunk,:]
        hfr_chunk_segment = output[:,:,hfr_frame_start:hfr_frame_start+hfr_chunk,:]
        hfr_hr_segment = space_time_backprojection(hfr_hr_pred=None, lfr_hr_in=lfr_hr_segment, hfr_lr_in=hfr_chunk_segment, device=device)
        if index == 0: #Workaround for using the dtype that returns from BP
            hfr_hr_pred = np.zeros(BP_output_shape, dtype=hfr_hr_segment.dtype)
        hfr_hr_pred[:, :, hfr_frame_start:hfr_frame_start + hfr_chunk, :] = hfr_hr_segment

    permutation_for_save = (2, 0, 1, 3)  # permute back to t,h,w,c, as that is what save_output_result expects
    hfr_hr_pred = np.transpose(hfr_hr_pred, permutation_for_save)
    output_afterBP_dir = os.path.join(config['trainer']['working_dir'],
                                      f'T{cumulative_scale}S{cumulative_spatial_scales[scale_ind + 1]}')

    save_output_result(hfr_hr_pred, output_afterBP_dir)
    output = hfr_hr_pred  # For saving final result
    cur_data_path = output_afterBP_dir  # For next step
    cur_spatial_scale = cumulative_spatial_scales[scale_ind + 1]
    return cur_data_path, cur_spatial_scale, output


def temporal_bp_wrapper(LFR, HFR):
    permutation = (1, 2, 0, 3)
    permuted_hfr = np.transpose(HFR, permutation)
    permuted_lfr = np.transpose(LFR, permutation)
    bp_hfr = temporal_backprojection_np(permuted_hfr, permuted_lfr)
    output = np.transpose(bp_hfr, (2, 0, 1, 3))
    return output


# ---------------------------------------------------------------------------------------------------
# video readers
# ---------------------------------------------------------------------------------------------------
img_exts = ['png', 'jpg', 'jpeg']


def read_seq_from_folder(frames_folder, prefix, dtype):
    print('-utils- reading sequence from {}'.format(frames_folder))
    return read_seq_from_folder_single_process(frames_folder, prefix, dtype)

def read_seq_from_folder_single_process(frames_folder, prefix, dtype):
    frames = []
    target_shape = None

    for ext in img_exts:
        filenames = sorted(glob.glob(os.path.join(frames_folder, '{}*.{}'.format(prefix, ext))))
        if len(filenames) > 0:
            break

    for filename in filenames:
        frame = np.array(Image.open(filename).convert('RGB')).astype(dtype) / 255.
        if target_shape is None:
            target_shape = frame.shape[:2]
        else:
            assert (target_shape == frame.shape[:2])
        frames.append(frame.astype(dtype))

    return frames


# ---------------------------------------------------------------------------------------------------
# misc functions
# ---------------------------------------------------------------------------------------------------
def tensor_3d_choice(probability_tensor, summed_1d_probability_vector):
    """
    Selects a random point from the probability_tensor, by using its summed_1d_probability_vector to split the random
    to 2 parts.
     Assumes the summed_1d_probability_vector is a summation on the [1,2] axes of the probability_tensor.
     Can be calculated inside, but for runtime purposes is done once externally for each probability_tensor
     when possible.
    """
    first_dim_chosen = np.random.choice(summed_1d_probability_vector.size, p=summed_1d_probability_vector)
    frame_chosen_normed = probability_tensor[first_dim_chosen, :, :] / np.sum(
        probability_tensor[first_dim_chosen, :, :])
    chosen_2d_flat = np.random.choice(frame_chosen_normed.size, p=np.ndarray.flatten(frame_chosen_normed))
    chosen_2d = np.unravel_index(chosen_2d_flat, frame_chosen_normed.shape)
    return (first_dim_chosen, chosen_2d[0], chosen_2d[1])


def downscale_for_BP(config, device):
    """
    creates the necessary spatial downscale folders to enable backprojection
    returns the smallest spatial scale and the folder path with the smallest spatial scale
    """
    upsample_steps = config['upsamaple_steps']
    if config['final_no_BP']: #The final step is w.o. BP, so no need for it here
        upsample_steps=upsample_steps[:-1]
    downsample_steps = list(reversed(upsample_steps))
    downsample_steps = [1] + downsample_steps #to save also the T1S1 scale
    max_downscale = np.prod(downsample_steps)
    orig_data_path = config['data']['params']['frames_folder']
    orig_tensor = np.asarray(read_seq_from_folder(orig_data_path, config["prefix"], config["dtype"]))
    assert orig_tensor.shape[1] % max_downscale == orig_tensor.shape[
        2] % max_downscale == 0, f'assertion error in downscale_for_BP: video shape not divisible by needed downscale'
    working_dir = config['trainer']['working_dir']
    cumulative_scale = 1
    cumulative_scale_list = []
    for scale_ind, scale in enumerate(downsample_steps):
        cumulative_scale = cumulative_scale * scale
        cumulative_scale_list = [cumulative_scale] + cumulative_scale_list
        scale_name = f'T1S{cumulative_scale}'
        folder_name = os.path.join(working_dir, scale_name)
        os.mkdir(folder_name)

        resizer = torch_resizer.Resizer(orig_tensor.shape,
                                        output_shape=[orig_tensor.shape[0],
                                                      int(orig_tensor.shape[1] / cumulative_scale),
                                                      int(orig_tensor.shape[2] / cumulative_scale),
                                                      orig_tensor.shape[3]],
                                        kernel='cubic', antialiasing=True, device=device, dtype=torch.float16)
        resized_tensor = np.clip(resizer(torch.tensor(orig_tensor, dtype=torch.float16).to(device)).cpu().numpy(), 0., 1.)

        save_output_result(resized_tensor, folder_name)
    return folder_name, cumulative_scale, cumulative_scale_list

# ---------------------------------------------------------------------------------------------------
# json config functions
# ---------------------------------------------------------------------------------------------------
def read_json_with_line_comments(cjson_path):
    with open(cjson_path, 'r') as R:
        valid = []
        for line in R.readlines():
            if line.lstrip().startswith('#') or line.lstrip().startswith('//'):
                continue
            valid.append(line)
    return json.loads(' '.join(valid))


def startup(json_path, args=None, copy_files=True):
    print('-startup- reading config json {}'.format(json_path))
    config = read_json_with_line_comments(json_path)

    # do we override tag with command line tag?
    if args is not None and hasattr(args, 'tag') and args.tag is not None:
        config['tag'] = args.tag

    # do we override data path with command line tag?
    if args is not None and hasattr(args, 'data') and args.data is not None:
        config['data']['params']['frames_folder'] = args.data
        # add mark to tag
        config['tag'] = '{}-{}'.format(config['tag'], os.path.split(args.data)[-1])
    # do we override gt location?
    if args is not None and hasattr(args, 'gt') and args.data is not None:
        config['data']['params']['gt_folder'] = args.gt

    if args is not None and hasattr(args, 'eval') and args.eval is not None:
        config['eval'] = args.eval

    if args is not None and hasattr(args, 'checkpoint') and args.checkpoint is not None:
        config['checkpoint'] = args.checkpoint

    if args is not None and hasattr(args, 'network') and args.network is not None:
        config['network'] = args.network

    if args is not None and hasattr(args, 'epochs') and args.epochs is not None:
        config['num_epochs'] = int(args.epochs)

    if args is not None and hasattr(args, 'gradcutoff') and args.gradcutoff is not None:
        config['data']['params']['gradient_percentile'] = int(args.gradcutoff)

    if args is not None and hasattr(args, 'spatialcrop') and args.spatialcrop is not None:
        config['data']['params']['augmentation_params']['crop_sizes']['crop_size_spatial'] = int(args.spatialcrop)

    if args is not None and hasattr(args, 'spatialmask') and args.spatialmask is not None:
        config['data']['params']['augmentation_params']['crop_sizes']['loss_mask_spatial'] = int(args.spatialmask)

    if args is not None and hasattr(args, 'temporalcrop') and args.temporalcrop is not None:
        config['data']['params']['augmentation_params']['crop_sizes']['crop_size_temporal'] = int(args.temporalcrop)

    if args is not None and hasattr(args, 'temporalmask') and args.temporalmask is not None:
        config['data']['params']['augmentation_params']['crop_sizes']['loss_mask_temporal'] = int(args.temporalmask)

    if args is not None and hasattr(args, 'withinprob') and args.withinprob is not None:
        config['data']['params']['augmentation_params']['within']['probability'] = float(args.withinprob)

    if args is not None and hasattr(args, 'acrossprob') and args.acrossprob is not None:
        config['data']['params']['augmentation_params']['across']['probability'] = float(args.acrossprob)

    if args is not None and hasattr(args, 'loss') and args.loss is not None:
        config['loss']['name'] = str(args.loss)

    if args is not None and hasattr(args, 'initiallr') and args.initiallr is not None:
        config['optimization']['params']['lr'] = float(args.initiallr)

    if args is not None and hasattr(args, 'workingdir') and args.workingdir is not None:
        config['working_dir_base'] = str(args.workingdir)

    if copy_files and ("working_dir" not in config['trainer'] or not os.path.isdir(config['trainer']['working_dir'])):
        # find available working dir
        v = 0
        while True:
            working_dir = os.path.join(config['working_dir_base'], '{}-v{}'.format(config['tag'], v))
            if not os.path.isdir(working_dir):
                break
            v += 1
        os.makedirs(working_dir, exist_ok=False)
        config['trainer']['working_dir'] = working_dir
        print('-startup- working directory is {}'.format(config['trainer']['working_dir']))

    # copy shared parameters
    config['data']['params']['dtype'] = config['dtype']
    config['trainer']['dtype'] = config['dtype']

    # copy files?
    if copy_files:
        for filename in os.listdir('.'):
            if filename.endswith('.py'):
                shutil.copy(filename, config['trainer']['working_dir'])
            shutil.copy(json_path, config['trainer']['working_dir'])
        with open(os.path.join(config['trainer']['working_dir'], '_processed_config.json'), 'w') as W:
            W.write(json.dumps(config, indent=2))

    #assertions and prints
    assert not (config["fix_network"] == True and config[
        "fine_tune"] == True), f'assertion error in config - fine tune and fix_network cannot both be True'
    assert config['checkpoint'] is not '' or config[
        'ckpt_first_trained'] == False, f'No checkpoint but ckpt_first_trained is True'

    if config["debug"] == True:
        print('*********************************************************************************')
        print(f'Debug!\nDebug!\nDebug!\nDebug!\nDebug!\nDebug!\nDebug!\nDebug!\nDebug!\nDebug!\n')
        print('*********************************************************************************')
    return config


def visualize_tuple(hr_lr_tuple, name_hr='HR', name_lr='LR', save_to_file=False, save_path='./results/imgs'):
    """
    take a tensor and its low resolution version (lr) and show them side-by-side
    :param hr_lr_tuple: (hr,lr) tuple of np arrays
    :param name: save folder name (selected randomly to allow saving seq.)
    :return: none, plots the frames or tensors
    """

    hr_tensor = hr_lr_tuple[0]
    lr_tensor = hr_lr_tuple[1]
    normalize = True
    if normalize:
        hr_tensor = hr_tensor / np.max(hr_tensor)
        lr_tensor = lr_tensor / np.max(lr_tensor)
    subsample_ratio = hr_tensor.shape[0] // lr_tensor.shape[0]

    for i in range(lr_tensor.shape[0]):
        plt.figure(1)
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(lr_tensor[i, :])
        plt.title(f'{name_lr} frame {i}')
        for j in range(subsample_ratio):
            plt.subplot(1, 2, 2)
            plt.imshow(hr_tensor[subsample_ratio * i + j, :],vmin=0.0)
            plt.title(f'{name_hr} frame {i*subsample_ratio + j}')
            #plt.draw()
            #plt.pause(0.05)

            if save_to_file:
                folder_name = save_path
                os.makedirs(folder_name, exist_ok=True)
                plt.savefig(f'{folder_name}/{subsample_ratio * i + j}.png')


def save_output_result(vid_tensor, path):
    """
    take a video tensor [f,h,w,c] and save it as frames
    :param vid_tensor: video tensor [f,h,w,c] numpy ndarray
    :param path: folder to save the frames
    :return: none
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    for i, im in enumerate(vid_tensor):
        pltimg.imsave(f'{path}/{i:05d}.png', np.clip(im, 0, 1))

def lin_interpolate_2(data_path, save_path):
    """
    used as baseline for comparison
    """
    video_tensor = np.asarray(read_seq_from_folder(data_path, "", "float32"))
    resized_tensor = np.clip(imresize(video_tensor, scale_factor=[2, 1, 1, 1], kernel="linear"), 0., 1.)
    save_output_result(resized_tensor, save_path)
