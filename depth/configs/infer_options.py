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


class InferOptions(BaseOptions):
    def initialize(self):
        parser = BaseOptions.initialize(self)

        # experiment configs
        parser.add_argument('--ckpt_dir', type=str, help='load ckpt path')
        parser.add_argument('--img_path', type=str, default=None)
        parser.add_argument('--video_path', type=str, default=None)
        return parser


