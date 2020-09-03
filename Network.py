import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim.optimizer
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import augmentations
import torch_resizer
import utils


class Network:  # The base network
    def __init__(self, config, device, upsample_scale=2):
        self.config = config
        self.upsample_scale = upsample_scale
        self.channels_in = 3
        self.channels_out = 3
        self.device = device
        self.net = self.build_network()
        self.optimizer = self.define_opt()
        self.loss_mask_spatial = self.config['data']['params']['augmentation_params']['crop_sizes']['loss_mask_spatial']
        self.loss_mask_temporal = self.config['data']['params']['augmentation_params']['crop_sizes']['loss_mask_temporal']
        self.lit_pixels = self.calc_lit_pixels()
        assert self.lit_pixels > 0, f'assertion error: no crop left after masking'
        self.loss_fn = self.define_loss()
        self.writer = SummaryWriter(os.path.join(config['trainer']['working_dir'], 'logs_dir'))

        # total number of epochs
        self.epochs = self.config['num_epochs']
        # current or start epoch number
        self.epoch = 0

        self.iter_per_epoch = self.config['num_iter_per_epoch']
        self.save_every = self.config['save_every']

        self.scheduler = self.define_lr_sched()

    def build_network(self):  # BASE version. Other modes override this function
        """
        take the network flag or parameters from config and create network
        :return: net - a torch class/object that can be trained
        """
        net = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.channels_in, out_channels=128, kernel_size=3, padding=1, stride=(self.upsample_scale, 1, 1),
                               output_padding=(self.upsample_scale - 1, 0, 0)),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=self.channels_out, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.ReLU(),
        ).to(self.device)
        return net

    def define_loss(self):
        loss_name = self.config['loss']['name']
        if loss_name == 'MSE':
            return torch.nn.MSELoss(reduction='sum')
        else:
            assert False, f'assertion error in define_opt(), loss does not exist, is {loss_name}'

    def define_opt(self):
        opt_name = self.config['optimization']['name']
        learning_rate = self.config['optimization']['params']['lr']
        if opt_name == 'SGD':
            momentum = self.config['optimization']['params']['SGD_momentum']
            return torch.optim.SGD(self.net.parameters(), lr=learning_rate, momentum=momentum)
        elif opt_name == 'Adam':
            return torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        else:
            assert False, f'assertion error in define_opt(), optimizer does not exist, is {opt_name}'

    def define_lr_sched(self):
        gamma = self.config['lr_sched']['params']['gamma']
        milestones = self.config['lr_sched']['params']['milestones']
        step_size = self.config['lr_sched']['params']['step_size']

        if self.config['lr_sched']['name'] == 'MultiStepLR':
            return lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)

        elif self.config['lr_sched']['name'] == 'StepLR':
            return lr_scheduler.StepLR(self.optimizer, step_size=int(self.epochs * step_size), gamma=gamma)
        else:
            print('****************** NO LR_SCHED DEFINED SETTING DEFAULT *****************************')
            return lr_scheduler.StepLR(self.optimizer, step_size=self.epochs // 10, gamma=1 / 1.5)

    def calc_lit_pixels(self):
        spatial = self.config['data']['params']['augmentation_params']['crop_sizes']['crop_size_spatial']
        temporal = self.config['data']['params']['augmentation_params']['crop_sizes']['crop_size_temporal']
        lit_mask = [temporal - 2 * self.loss_mask_temporal, spatial - 2 * self.loss_mask_spatial,
                    spatial - 2 * self.loss_mask_spatial, 3]
        return np.prod(lit_mask)

    def forward_zstsr(self, input_tensor):  # BASE version. Other modes override this function
        return self.net(input_tensor)

    def calc_loss(self, output, hr_gt):
        """
        calc loss according to the flags in config
        :param output: the output from the net. May need to add input if residual
        :param hr_gt_torch: the hr gt from the tuple
        :return: the loss
        """

        loss_name = self.config['loss']['name']
        # To remove spatial and temporal masking
        t = self.loss_mask_temporal
        t_end = output.shape[2] - t
        s = self.loss_mask_spatial
        s_end_ver = output.shape[3] - s
        s_end_hor = output.shape[4] - s
        shape_masked = np.prod(
            output[:, :, t:t_end, s:s_end_ver, s:s_end_hor].shape)
        if loss_name == 'MSE':
            return torch.sum(
                (output[:, :, t:t_end, s:s_end_ver, s:s_end_hor].to(self.device) -
                 hr_gt[:, :, t:t_end, s:s_end_ver, s:s_end_hor].to(self.device)) ** 2.0) / shape_masked
        else:
            assert False, f'assertion error in calc_loss(), loss not MSE, is {loss_name}'

    def train(self, data_loader_object, cumulative_scale):
        """
        :param data_loader_object: data_handler object that holds the video tensor and can make all necessary augmentations
        :param cumulative_scale: indicates the current training location in the global config. Needed for saving the model.
        :return: train_logs. loss vectors for each epoch
        """
        # epochs
        for e in range(self.epoch, self.epochs):
            t = time.time()
            np.random.seed()
            self.optimizer.zero_grad()
            if e % self.config['val_every'] == self.config['val_every'] - 1:
                if self.config['debug']:
                    print('Debug!\nDebug!\nNo validation!\nDebug!\nDebug!\n')
                else:
                    print(f'applying val at epoch {e}')
                    self.validation(data_loader_object, cumulative_scale=cumulative_scale, epoch=e)
            if e % self.config['save_every'] == self.config['save_every'] - 1:
                print(f'saved model at epoch {e}')
                self.save_model(epoch=e, overwrite=False, cumulative_scale=cumulative_scale)

            # iterations per epochs
            it = 0
            for (hr_gt, lr) in data_loader_object:
                hr_prediction = self.forward_zstsr(lr.to(self.device))
                loss = self.calc_loss(hr_prediction, hr_gt)
                it += 1
            print(f'epoch:{e}, loss:{loss.item():.7f}. Time: {(time.time() - t):.2f}, lr={self.optimizer.param_groups[0]["lr"]}')
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            self.writer.add_scalars('loss', {'loss': loss.item()})
            self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]["lr"]})

        # save final trained model as well
        self.save_model(epoch=self.epochs, overwrite=False, cumulative_scale=cumulative_scale)
        self.writer.close()
        return

    def validation(self, data_loader_object, cumulative_scale, epoch):
        """
        apply eval on video temporally downscaled by working scale, test return to original video
        :param epoch: to save with curent epoch#
        :return: None, but creates the files in output folder
        """
        HTR_val_tensor = data_loader_object.dataset.video_tensor  # input in this training, but for val it's the HTR
        # clip trailing number of frames, so for instance even (not odd) when upsample_scale==2
        HTR_val_tensor = HTR_val_tensor[:HTR_val_tensor.shape[0] - HTR_val_tensor.shape[0] % self.upsample_scale, ...]
        LTR_val_tensor = augmentations.blur_sample_tensor(HTR_val_tensor, sample_axis=0,
                                                          sample_jump=self.upsample_scale,
                                                          blur_flag=data_loader_object.dataset.blur_flag)
        predicted_val = self.eval(LTR_val_tensor)
        val_loss = self.calc_loss(torch.from_numpy(np.expand_dims(predicted_val, 0)).float(), torch.from_numpy(np.expand_dims(HTR_val_tensor, 0)).float())
        self.writer.add_scalars('val_loss', {'val_loss': val_loss})
        print(f'VALIDATION AFTER epoch:{epoch}, loss:{val_loss:.5f}')

        val_dir = os.path.join(self.config['trainer']['working_dir'], 'validation', f'cumulative_scale_{cumulative_scale}', f'epoch_{epoch}_loss_{val_loss:.5f}')
        utils.save_output_result(predicted_val, val_dir)

    def eval(self, video_tensor):
        """
        take the input video and upscale it
        :param data: data_handler object, contains the whole video, on which we run the network to produce an upsampled video
        :return:
        """
        video_tensor = np.copy(video_tensor)
        # this tensor will be filled with crops and returned
        prediction_video = np.zeros([self.upsample_scale * video_tensor.shape[0], video_tensor.shape[1], video_tensor.shape[2], video_tensor.shape[3]])

        if self.config['debug']:
            prediction_video = self.debug_eval(prediction_video, video_tensor)
            return prediction_video

        # Helper function for calculating the sizes needed for operating in crops
        f_pad, f_pad_output, f_starts_input, f_starts_outputs, h_pad, h_starts, net_f_output, net_h, net_w, \
        size_frames, size_height, size_width, w_pad, w_starts = self.eval_calc_param_sizes(video_tensor)

        # Pad the video on all sides by needed factor
        video_tensor = np.pad(video_tensor, [(f_pad, f_pad), (h_pad, h_pad), (w_pad, w_pad), (0, 0)], 'symmetric')

        # create a [f,h,w,c] block of size defined above
        for f_ind, f_start in enumerate(f_starts_input):
            print(f'EVAL: frame start:{f_start}')
            for h_ind, h_start in enumerate(h_starts):
                for w_ind, w_start in enumerate(w_starts):
                    if (f_start + size_frames - 1) > (video_tensor.shape[0]) or (h_start + size_height - 1) > \
                            video_tensor.shape[1] or (w_start + size_width - 1) > video_tensor.shape[2]:
                        print('eval error: should not reach here - size issue')
                        continue
                    crop = video_tensor[f_start:f_start + size_frames, h_start:h_start + size_height,
                           w_start:w_start + size_width, :]
                    net_output = self.eval_forward_crop(crop)

                    # snip and save in the entire output video
                    try:
                        # snip edges - according to the padding parameter
                        net_output = net_output[f_pad_output:-f_pad_output, h_pad:-h_pad, w_pad:-w_pad, :]
                        # Notice: size in "frames" axis in the output is twice the net_size in the input
                        prediction_video[f_starts_outputs[f_ind]:f_starts_outputs[f_ind] + net_f_output,
                        h_start:h_start + net_h, w_start:w_start + net_w, :] = net_output.detach().cpu().numpy()
                    except:
                        print('eval error: should not reach here - cropping/stitching issue')

        return prediction_video

    def debug_eval(self, prediction_video, video_tensor):
        print(f'Debug!\nDebug!\nDebug!\nDebug!\nDebug!\nDebug!\nDebug!\nDebug!\nDebug!\nDebug!\n')
        debug_method = 'copy_frame'  # 'copy_frame' or 'interpolate'. If neither, returns zeros
        if debug_method == 'copy_frame':
            for frame_up_idx in range(prediction_video.shape[0]):
                prediction_video[frame_up_idx, :, :, :] = video_tensor[int(frame_up_idx / self.upsample_scale), :, :, :]
        elif debug_method == 'interpolate':
            resizer = torch_resizer.Resizer(video_tensor.shape[:], scale_factor=(self.upsample_scale, 1, 1, 1),
                                            output_shape=[video_tensor.shape[0] * self.upsample_scale, video_tensor.shape[1], video_tensor.shape[2], video_tensor.shape[3]],
                                            kernel='cubic', antialiasing=True, device='cuda')
            prediction_video = resizer.forward(torch.tensor(video_tensor).to(self.device)).to(self.device).cpu().numpy()

        return prediction_video.squeeze()

    def eval_calc_param_sizes(self, video_tensor):
        size_frames = self.config['data']['params']['eval_params']['size_frames']
        size_height = self.config['data']['params']['eval_params']['size_height']
        size_width = self.config['data']['params']['eval_params']['size_width']
        f_pad = self.config['data']['params']['eval_params']['pad_frames']
        h_pad = self.config['data']['params']['eval_params']['pad_height']
        w_pad = self.config['data']['params']['eval_params']['pad_width']
        f_pad_output = self.upsample_scale * f_pad
        net_f = size_frames - 2 * f_pad  # The actual size added by each forward, need to remove the padding. 2 because each side
        net_f_output = self.upsample_scale * net_f
        net_h = size_height - 2 * h_pad
        net_w = size_width - 2 * w_pad
        # The start points for crops, advance in each axis by its net_size each crop
        f_starts_input = np.arange(0, video_tensor.shape[0], net_f)
        f_starts_input[-1] = video_tensor.shape[0] - net_f  # For final crop at each dim
        f_starts_outputs = self.upsample_scale * f_starts_input  # output is *scale the frames
        h_starts = np.arange(0, video_tensor.shape[1], net_h)
        h_starts[-1] = video_tensor.shape[1] - net_h
        w_starts = np.arange(0, video_tensor.shape[2], net_w)
        w_starts[-1] = video_tensor.shape[2] - net_w
        return f_pad, f_pad_output, f_starts_input, f_starts_outputs, h_pad, h_starts, \
               net_f_output, net_h, net_w, size_frames, size_height, size_width, w_pad, w_starts

    def eval_forward_crop(self, crop):
        """
        helper function for eval - prepares and forwards the crop
        """
        # prep to send to torch (GPU)
        permutation_np_to_torch = (3, 0, 1, 2)  # move channels to first
        crop = np.transpose(crop, permutation_np_to_torch)
        video_tensor_torch = torch.unsqueeze(torch.from_numpy(crop).float(), dim=0).to(self.device)
        # EVAL current block
        self.net.eval()
        with torch.no_grad():
            # the value is automatically converted to numpy and squeezed to [c,f,h,w]
            net_output = torch.squeeze(self.forward_zstsr(video_tensor_torch).to(self.device))
        # transpose back to [f,h,w,c]
        net_output = net_output.permute((1, 2, 3, 0))
        return net_output

    def save_model(self, epoch=None, scale=None, overwrite=False, cumulative_scale=2):
        """
        Saves the model (state-dict, optimizer and lr_sched
        :return:
        """
        if overwrite:
            checkpoint_list = [i for i in os.listdir(os.path.join(self.config['trainer']['working_dir'])) if i.endswith('.pth.tar')]
            if len(checkpoint_list) != 0:
                os.remove(os.path.join(self.config['trainer']['working_dir'], checkpoint_list[-1]))

        filename = 'checkpoint{}{}.pth.tar'.format('' if epoch is None else '-e{:05d}'.format(epoch),
                                                   '' if scale is None else '-s{:02d}'.format(scale))
        folder = os.path.join(self.config['trainer']['working_dir'], 'saved_models', f'cumulative_scale_{cumulative_scale}')
        os.makedirs(folder, exist_ok=True)
        torch.save({'epoch': epoch,
                    'sd': self.net.state_dict(),
                    'opt': self.optimizer.state_dict()},
                   # 'lr_sched': self.scheduler.state_dict()},
                   os.path.join(folder, filename))

    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.net.load_state_dict(checkpoint['sd'], strict=False)
        self.optimizer.load_state_dict(checkpoint['opt'])
        self.epoch = checkpoint['epoch']

