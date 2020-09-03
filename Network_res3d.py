import torch
import torch.nn as nn
import torch_resizer
from Network import Network
from augmentations import *

class Network_residual(Network):  # Network with residual after bilinear
    def __init__(self, config, device, upsample_scale):
        super().__init__(config, device, upsample_scale)

    def build_network(self):
        """
        take the network flag or parameters from config and create network
        :return: net - a torch class/object that can be trained
        """
        class NeuralNetwork(nn.Module):
            def __init__(self, channels_in, channels_out, config, upsample_scale):
                super().__init__()

                self.config = config
                self.upsample_scale = upsample_scale
                # Inputs to 1st hidden layer linear transformation
                self.L1 = nn.Conv3d(in_channels=channels_in, out_channels=128, kernel_size=3, padding=1, padding_mode='zeros')
                torch.nn.init.normal_(self.L1.weight, mean=0, std=np.sqrt(0.1 / np.prod(self.L1.weight.shape[1:])))
                torch.nn.init.normal_(self.L1.bias, mean=0, std=np.sqrt(0.1))
                self.L1_b = nn.BatchNorm3d(128)

                self.L2 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1, padding_mode='zeros')
                torch.nn.init.normal_(self.L2.weight, mean=0, std=np.sqrt(0.1 / np.prod(self.L2.weight.shape[1:])))
                torch.nn.init.normal_(self.L2.bias, mean=0, std=np.sqrt(0.1))
                self.L2_b = nn.BatchNorm3d(128)

                self.L3 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode='zeros')
                torch.nn.init.normal_(self.L3.weight, mean=0, std=np.sqrt(0.1 / np.prod(self.L3.weight.shape[1:])))
                torch.nn.init.normal_(self.L3.bias, mean=0, std=np.sqrt(0.1))
                self.L3_b = nn.BatchNorm3d(128)

                self.L4 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode='zeros')
                torch.nn.init.normal_(self.L4.weight, mean=0, std=np.sqrt(0.1 / np.prod(self.L4.weight.shape[1:])))
                torch.nn.init.normal_(self.L4.bias, mean=0, std=np.sqrt(0.1))
                self.L4_b = nn.BatchNorm3d(128)

                self.L5 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode='zeros')
                torch.nn.init.normal_(self.L5.weight, mean=0, std=np.sqrt(0.1 / np.prod(self.L5.weight.shape[1:])))
                torch.nn.init.normal_(self.L5.bias, mean=0, std=np.sqrt(0.1))
                self.L5_b = nn.BatchNorm3d(128)

                self.L6 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode='zeros')
                torch.nn.init.normal_(self.L6.weight, mean=0, std=np.sqrt(0.1 / np.prod(self.L6.weight.shape[1:])))
                torch.nn.init.normal_(self.L6.bias, mean=0, std=np.sqrt(0.1))
                self.L6_b = nn.BatchNorm3d(128)

                self.L7 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode='zeros')
                torch.nn.init.normal_(self.L7.weight, mean=0, std=np.sqrt(0.1 / np.prod(self.L7.weight.shape[1:])))
                torch.nn.init.normal_(self.L7.bias, mean=0, std=np.sqrt(0.1))
                self.L7_b = nn.BatchNorm3d(128)

                self.L8 = nn.Conv3d(in_channels=128, out_channels=channels_out, kernel_size=3, padding=1, padding_mode='zeros')
                torch.nn.init.normal_(self.L8.weight, mean=0, std=np.sqrt(0.1 / np.prod(self.L8.weight.shape[1:])))
                torch.nn.init.normal_(self.L8.bias, mean=0, std=np.sqrt(0.1))

                self.activation = nn.ReLU()

            def forward(self, x):
                residual_base = self.config["res3d_up_method"]  # 'resize' 'duplicate' 'zero_gap'
                if residual_base == 'resize':
                    self.resizer = torch_resizer.Resizer(x.shape, scale_factor=(1, 1, self.upsample_scale, 1, 1),
                                                         output_shape=[x.shape[0], x.shape[1], x.shape[2] * self.upsample_scale, x.shape[3], x.shape[4]],
                                                         kernel='cubic', antialiasing=True, device='cuda')
                    x_upsampled = self.resizer(x)
                    x = self.resizer(x)

                elif residual_base == 'duplicate':
                    x_upsampled = torch.nn.functional.interpolate(
                        x, scale_factor=(self.upsample_scale, 1, 1), mode='trilinear', align_corners=False)
                    for frame_up_idx in range(x_upsampled.shape[2]):
                        x_upsampled[:, :, frame_up_idx, :, :] = x[:, :, int(frame_up_idx / self.upsample_scale), :, :]

                    x_temp = x.detach().clone()
                    x = torch.nn.functional.interpolate(
                        x, scale_factor=(self.upsample_scale, 1, 1), mode='trilinear', align_corners=False)
                    for frame_up_idx in range(x.shape[2]):
                        x[:, :, frame_up_idx, :, :] = x_temp[:, :, int(frame_up_idx / self.upsample_scale), :, :]

                elif residual_base == 'zero_gap':
                    zero_frame = torch.zeros_like(x[:, :, 0, :, :])
                    x_upsampled = torch.nn.functional.interpolate(
                        x, scale_factor=(self.upsample_scale, 1, 1), mode='trilinear', align_corners=False)
                    for frame_up_idx in range(x_upsampled.shape[2]):
                        if frame_up_idx % self.upsample_scale == 0:  # insert orig frame
                            x_upsampled[:, :, frame_up_idx, :, :] = x[:, :, int(frame_up_idx / self.upsample_scale), :, :]
                        else:
                            x_upsampled[:, :, frame_up_idx, :, :] = zero_frame

                    x_temp = x.detach().clone()
                    x = torch.nn.functional.interpolate(
                        x, scale_factor=(self.upsample_scale, 1, 1), mode='trilinear', align_corners=False)
                    for frame_up_idx in range(x.shape[2]):
                        if frame_up_idx % self.upsample_scale == 0:  # insert orig frame
                            x[:, :, frame_up_idx, :, :] = x_temp[:, :, int(frame_up_idx / self.upsample_scale), :, :]
                        else:
                            x[:, :, frame_up_idx, :, :] = zero_frame

                else:
                    assert False, f'assertion error in Network_residual forward - residual_base not known: {residual_base}'
                # x -> [Batch, Channel, Time, Height, Width]
                x = torch.nn.functional.pad(x, [1, 1, 1, 1, 1, 1], mode='replicate')
                x1 = self.L1(x)
                x1 = self.L1_b(x1)
                x2 = nn.ReLU()(x1)

                x3 = self.L2(x2)
                x3 = self.L2_b(x3)
                x4 = nn.ReLU()(x3)

                x5 = self.L3(x4)
                x5 = self.L3_b(x5)
                x6 = nn.ReLU()(x5)

                x7 = self.L4(x6)
                x7 = self.L4_b(x7)
                x8 = nn.ReLU()(x7)

                x9 = self.L5(x8)
                x9 = self.L5_b(x9)
                x10 = nn.ReLU()(x9)

                x11 = self.L6(x10)
                x11 = self.L6_b(x11)
                x12 = nn.ReLU()(x11)

                x13 = self.L7(x12)
                x13 = self.L7_b(x13)
                x14 = nn.ReLU()(x13)

                x15 = self.L8(x14)
                return x15[:, :, 1:-1, 1:-1, 1:-1] + x_upsampled

        net = NeuralNetwork(self.channels_in, self.channels_out, self.config, self.upsample_scale).to(self.device)
        return net
