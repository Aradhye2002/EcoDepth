
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class BaseOptions():
    def __init__(self):
        pass

    def initialize(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # base configs
        parser.add_argument('--exp_name',   type=str, default='')
        parser.add_argument('--gpu_or_cpu',   type=str, default='gpu')
        parser.add_argument('--data_path',    type=str, default='/data/ssd1/')
        parser.add_argument('--dataset',      type=str, default='nyudepthv2',
                            choices=['nyudepthv2', 'kitti'])
        parser.add_argument('--batch_size',   type=int, default=8)
        parser.add_argument('--workers',      type=int, default=4)
        
        # depth configs
        parser.add_argument('--max_depth',      type=float, default=10.0)
        parser.add_argument('--min_depth',      type=float, default=1e-3)
        parser.add_argument('--max_depth_eval', type=float, default=10.0)
        parser.add_argument('--min_depth_eval', type=float, default=1e-3)        
        parser.add_argument('--do_kb_crop', type=str2bool, default='True')
        parser.add_argument('--kitti_crop', type=str, default=None,
                            choices=['garg_crop', 'eigen_crop'])


        parser.add_argument('--pretrained',    type=str, default='')
        parser.add_argument('--drop_path_rate',     type=float, default=0.3)
        parser.add_argument('--use_checkpoint',   type=str2bool, default='False')
        parser.add_argument('--num_deconv',     type=int, default=3)
        parser.add_argument('--num_filters', nargs='+', type=int, default=[32,32,32])
        parser.add_argument('--deconv_kernels', nargs='+', type=int, default=[2,2,2])

        parser.add_argument('--flip_test', action='store_true')       
        parser.add_argument('--no_of_classes', type=int, default='100')
        parser.add_argument('--vit_model', type=str, default='google/vit-base-patch16-224')
        parser.add_argument('--kitti_split_to_half', type=str2bool, default='True') 
        parser.add_argument('--eigen_crop_in_dataloader_itself_for_nyu', type=str2bool, default='False') 
        parser.add_argument('--use_right', type=str2bool, default='False')  
        parser.add_argument('--cutflip', type=str2bool, default='False')  
        parser.add_argument('--variance_focus', type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)
        
        return parser
