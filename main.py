import torch
import Network
import Network_res3d
from data_handler import *
import cProfile
import io
import pstats

parser = utils.create_parser()
args = parser.parse_args()


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def main():
    # read json return
    config = utils.startup(json_path=args.config, args=args, copy_files=args.eval is None or args.eval == 'empty')

    # get available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = {'batch_size': config["batch_size"],
              'shuffle': True,
              'num_workers': 0,
              'worker_init_fn': worker_init_fn}
    cur_data_path = config['data']['params']['frames_folder']

    cur_spatial_scale = 1
    if config['backprojection'] == True:
        cur_data_path, cur_spatial_scale, cumulative_spatial_scales = utils.downscale_for_BP(config, device)

    upsample_steps = config['upsamaple_steps']
    network = None
    output = None
    cumulative_scale = 1
    for scale_ind, scale in enumerate(upsample_steps):
        cumulative_scale = cumulative_scale * scale
        print('*********************************************************************************')
        print(f'entered temporal iteration {scale_ind}. Upsampling by temporal scale {scale}. Until now (including this): {cumulative_scale}')
        print(f'upscaling from spatial scale {cur_spatial_scale}, FROM path: {cur_data_path}')
        print('*********************************************************************************')

        # Check if this scale will be used for eval only, so no need for entire DataHandler object
        scale_for_eval_only = config['fix_network'] == True and scale_ind != 0

        dataset = DataHandler(data_path=cur_data_path, config=config, upsample_scale=scale, device=device,
                              video_only_flag=scale_for_eval_only)
        assert dataset.crop_size[0] % scale == 0, f'assertion error in main, temporal crop size not divisible by scale'
        data_generator = data.DataLoader(dataset, **params)

        train_from_scratch = scale_ind == 0 or (config['fix_network'] is False and config['fine_tune'] is False)
        if train_from_scratch:
            network_class = config['network']
            if network_class == 'base':
                network = Network.Network(config=config, device=device, upsample_scale=scale)
            elif network_class == 'residual':
                network = Network_res3d.Network_residual(config=config, device=device, upsample_scale=scale)
            else:
                assert False, f'assertion fail at main, not a known "network_class"'
        else:  # In fine tuning/fixed network. Either way - No new network.
            assert len(set(upsample_steps)) <= 1  # Make sure all upsamaple_steps are identical
            network.epochs = network.epochs // config['fine_tuning_epochs_factor']  # if fine_tuning: needed. If fixed: No impact
            assert network.epochs > 0, f'assertion error in main. "fine_tuning_epochs_factor" too large - No epochs left for fine_tuning in training iteration {scale_ind}'

        need_to_train = (config['fix_network'] == False or scale_ind == 0) and (not config['ckpt_first_trained'] or scale_ind != 0)  # Net not fixed or first training

        if config['checkpoint'] is not '' and scale_ind == 0:  # Load model. Only first iteration
            network.load_model(config['checkpoint'])
            print('loaded_ckpt\nloaded_ckpt\nloaded_ckpt\nloaded_ckpt\nloaded_ckpt\nloaded_ckpt\n')

        if need_to_train:
            # call train - provide a data_handler object to provide (lr,hr) tuples
            network.train(data_generator, cumulative_scale)
            if config['debug']:
                utils.visualize_tuple((dataset.video_tensor, dataset.crops_on_video_tensor), name_hr='video', name_lr='selected crops', save_to_file=True,
                                      save_path=os.path.join(config['trainer']['working_dir'], 'visualize_crops', f'scale_{cumulative_scale}'))

        # reset the start epoch value for next training scale
        network.epoch = 0

        # call eval
        output = network.eval(dataset.video_tensor)

        # apply temporal BP
        output = utils.temporal_bp_wrapper(dataset.video_tensor, output)

        # save results to file
        output_dir = os.path.join(config['trainer']['working_dir'], f'T{cumulative_scale}S{cur_spatial_scale}')
        utils.save_output_result(output, output_dir)

        # update data_path for next step
        cur_data_path = output_dir

        # Apply BP if needed
        skip_BP = config['final_no_BP'] and scale_ind == len(upsample_steps) - 1  # Final upscale when BP not wanted - skip BP
        if config['backprojection'] and not skip_BP:
            cur_data_path, cur_spatial_scale, output = utils.BP_wrapper(config, cumulative_scale,
                                                                        cumulative_spatial_scales,
                                                                        cur_data_path, cur_spatial_scale, output,
                                                                        scale_ind, device)

    # save final result in "output" folder
    final_output_dir = os.path.join(config['trainer']['working_dir'], 'output')
    utils.save_output_result(output, final_output_dir)


if __name__ == '__main__':
    # open comment to allow profiling
    # pr = cProfile.Profile()
    # pr.enable()
    # main()
    # pr.disable()
    # pr.print_stats(sort="cumtime")
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    # ps.print_stats()
    # with open('profile.txt', 'w+') as f:
    #     f.write(s.getvalue())

    main()
    print('done.')
