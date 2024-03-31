import argparse
from configs.base_options import BaseOptions

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class TestOptions(BaseOptions):
    def initialize(self):
        parser = BaseOptions.initialize(self)

        # experiment configs
        parser.add_argument('--ckpt_dir',   type=str,
                    default='./ckpt/best_model_nyu.ckpt', 
                    help='load ckpt path')
        parser.add_argument('--result_dir', type=str, default='./results',
                            help='save result images into result_dir/exp_name')     

        parser.add_argument('--save_eval_pngs', action='store_true',
                            help='save result image into evaluation form')
        parser.add_argument('--save_visualize', action='store_true',
                            help='save result image into visulized form')
        parser.add_argument('--median_scaling', type=str2bool, default='False') 
        parser.add_argument('--kitti_validation_split', type=str2bool, default='False')
        parser.add_argument('--kitti_held_out_test_set', type=str2bool, default='False')
        return parser


