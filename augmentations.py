import numpy as np
from scipy import ndimage
import math
import torch_resizer
import torch

def shift_frame(input_frame, shift_per_frame_ver, shift_per_frame_hor, z_len, valid_flag=False):
    """
    take a frame and shift by a given amount per frame, for z frames
    :param input_frame: np array. Order: height,width,channels
    :param shift_per_frame_hor: shift per frame in horizontal (x) direction
    :param shift_per_frame_ver: shift per frame in vertical (y) direction
    :param z_len: number of frames in output
    :param valid_flag: When False: same height\width as input, with zeros where can't put valid input.
                 When True: cuts to get minimal valid tensor
    :return: np array tensor. Order: frames,height,width,channels.
    """
    assert len(
        input_frame.shape) == 3, f"assertion error in shift_frame: len(shape) not 3, shape is {input_frame.shape}"
    output_uncropped = np.zeros((z_len,) + input_frame.shape)

    integer_flag = (not shift_per_frame_ver % 1) and (not shift_per_frame_hor % 1)
    if integer_flag: #can use roll
        for idx in range(z_len):
            output_uncropped[idx, :, :, :] = np.roll(input_frame,
                                                     [idx * shift_per_frame_ver, idx * shift_per_frame_hor],axis=(0, 1))
    else: #need to interpolate
        for idx in range(z_len):
            shifts = (idx * shift_per_frame_ver, idx * shift_per_frame_hor, 0)
            output_uncropped[idx, :, :, :] = np.clip(ndimage.shift(input_frame, shifts, order=3, mode='constant', cval=0.0),
                                                     0., 1.)

    if valid_flag:
        valid_ver_start = max(0, math.ceil(z_len * shift_per_frame_ver))
        valid_ver_end = min(input_frame.shape[0], input_frame.shape[0] + math.floor(z_len * shift_per_frame_ver))
        valid_hor_start = max(0, math.ceil(z_len * shift_per_frame_hor))
        valid_hor_end = min(input_frame.shape[1], input_frame.shape[1] + math.floor(z_len * shift_per_frame_hor))

        assert valid_ver_end > valid_ver_start >= 0 and valid_hor_end > valid_hor_start >= 0, \
            f"assertion error in shift_frame: output size is not valid, is " \
                f"[{valid_ver_start}:{valid_ver_end},{valid_hor_start}:{valid_hor_end}]"

        return output_uncropped[:, valid_ver_start:valid_ver_end, valid_hor_start:valid_hor_end, :]

    else:
        return output_uncropped


def flip_rotate_tensor(input_tensor, flip_prob, rotation_prob, z_flip_prob):
    """
    take a tensor and flip+rotation probabilities, and does so accordingly.
    :param input_tensor: input numpy tensor to rotate/flip. Order: frames,height,width,channels. Rotates/flips height,width
    :flip_prob: probability to flip
    :rotation_prob: probability to rotate
    param rotation_times: 1->90, 2->180, 3->270
    :return: np array, rotated/flipped tensor
    """
    flip_flag = np.random.choice([0, 1], p=[1 - flip_prob, flip_prob])
    rotation_flag = np.random.choice([0, 1], p=[1 - rotation_prob, rotation_prob])
    z_flip_flag = np.random.choice([0, 1], p=[1 - z_flip_prob, z_flip_prob])
    out_tensor = input_tensor
    if flip_flag:
        flip_directions = np.random.randint(1, 4)
        out_tensor = flip_tensor(out_tensor, flip_directions)
    if rotation_flag:
        rotation_times = np.random.randint(1, 4)
        out_tensor = rotate_tensor(out_tensor, rotation_times)
    if z_flip_flag:
        out_tensor = flip_tensor(out_tensor, 0)
    return out_tensor


def rotate_tensor(input_tensor, rotation_times):
    """
    take a tensor and rotate it
    :param input_tensor: input numpy tensor to rotate. Order: frames,height,width,channels. Rotates height,width
    :param rotation_times: 1->90, 2->180, 3->270
    :return: np array, rotated tensor
    """
    assert len(input_tensor.shape) == 4, f"assertion error in rotate_tensor: len(shape) not 4, is {len(input_tensor.shape)}"
    assert 1 <= rotation_times <= 3, f"assertion error in rotate_tensor: rotation_times not in [1,3], is {rotation_times}"
    rotated_tensor = np.rot90(input_tensor, rotation_times, (1, 2))
    return rotated_tensor

def flip_tensor(input_tensor, flip_directions):
    """
    take a tensor and rotate it
    :param input_tensor: input numpy tensor to flip. Order: frames,height,width,channels. Flips height,width
    :param flip_directions: 1->horizontal, 2->vertical, 3->hor+ver
    :return: np array, rotated tensor
    """
    assert len(input_tensor.shape) == 4, f"assertion error in flip_tensor: len(shape) not 4, is {len(input_tensor.shape)}"
    assert flip_directions in [0, 1, 2, 3], f"assertion error in flip_tensor: flip_directions is {flip_directions}"
    if flip_directions == 3:
        flip_directions = (1, 2)
    flipped_tensor = np.flip(input_tensor, flip_directions)
    return flipped_tensor

def resize_tensor(input_tensor, method, scale, device, clip01=True):
    """
    take a tensor and resize, using ZSSR's imresize
    :param input_tensor: np array. Order: frames, height, width, channels. Resizes height, width
    :param method: "cubic", "lanczos2", "lanczos3", "box", "linear"
    :param scale: the resize factor (np array of same shape as input_tensor)
    :param clip01: clip to [0,1]. Usually needed. Example where not: gradients.
    :return: np array
    """

    assert len(input_tensor.shape) == len(
        scale), f"assertion error in resize_tensor: input_tensor is {input_tensor.shape}, scale is {scale}"
    assert method in ["cubic", "lanczos2", "lanczos3", "box",
                      "linear"], f"assertion error in resize_tensor: method is {method}, not supported"

    resizer = torch_resizer.Resizer(input_tensor.shape, scale_factor=scale,
                                         kernel=method, antialiasing=True, device=device)
    if clip01:
        resized_tensor = np.clip(resizer(torch.tensor(input_tensor, dtype=torch.float16).to(device)).cpu().numpy(), 0., 1.)
    else:
        resized_tensor = resizer(torch.tensor(input_tensor, dtype=torch.float16).to(device)).cpu().numpy()
    return resized_tensor

def create_blur_filter(sample_jump, input_tensor_shape, sample_axis):
    blur_filter = np.array([1 / sample_jump for i in range(sample_jump)])
    for i in range(len(input_tensor_shape) - 1):
        blur_filter = np.expand_dims(blur_filter, axis=0)
    permute = [i for i in range(len(input_tensor_shape))]
    permute[sample_axis] = permute[-1]
    permute[-1] = sample_axis
    blur_filter = np.transpose(blur_filter, tuple(permute))
    return blur_filter

def blur_sample_tensor(input_tensor, sample_axis, sample_jump, blur_flag=False):
    """
    samples in specific axis. If blur-True, also blurs, by the same jump (full exposure)
    :param input_tensor: np array. Order: frames, height, width, channels.
    :param sample_axis: the axis in which we sample
    :param sample_jump: the jump
    :param blur_flag: boolean. whether to blur+sample (True), or just sample (False). Full exposure.
    :return: np array
    """
    assert sample_axis <= len(input_tensor.shape),\
        f"assertion error in blur_sample_tensor: sample_axis is {sample_axis}, input shape is {input_tensor.shape}"

    # blur if needed
    if sample_jump == 1:
        return input_tensor

    if blur_flag is True:
        # How we do this and avoid edge issues? split the input tensor to #sample_jump tensors, and sum them
        # This was done like this to make sure we avoid edge issues.
        out_size = list(input_tensor.shape)
        out_size[sample_axis] = math.ceil(out_size[sample_axis] / sample_jump)
        blur_sampled_tensor = np.zeros(out_size, dtype=input_tensor.dtype)
        for i in range(sample_jump):
            sl = [slice(None)] * len(input_tensor.shape)
            sl[sample_axis] = slice(i, None, sample_jump)
            addition = (1 / sample_jump) * input_tensor[tuple(sl)]
            addition_padded = np.zeros(out_size, dtype=input_tensor.dtype)
            addition_padded[:addition.shape[0], :addition.shape[1], :addition.shape[2], :addition.shape[3]] = addition
            blur_sampled_tensor = blur_sampled_tensor + addition_padded
        return np.clip(blur_sampled_tensor, 0., 1.)
    elif blur_flag is False:
        sl = [slice(None)] * len(input_tensor.shape)
        sl[sample_axis] = slice(0, None, sample_jump)
        sampled_tensor = input_tensor[tuple(sl)]
        return sampled_tensor
    else:
        assert False, f'assertion error in blur_sample_tensor, blur_flag not valid: {blur_flag}'