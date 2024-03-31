
from configs.base_options import BaseOptions
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


class TrainOptions(BaseOptions):
    def initialize(self):
        parser = BaseOptions.initialize(self)

        # experiment configs
        parser.add_argument('--epochs',      type=int,   default=25)
        parser.add_argument('--max_lr',          type=float, default=5e-4) # original
        parser.add_argument('--min_lr',          type=float, default=3e-5)
        parser.add_argument('--weight_decay',          type=float, default=5e-2)
        parser.add_argument('--layer_decay',          type=float, default=0.9)
        
        parser.add_argument('--log_dir', type=str, default='./logs')

        # logging options
        parser.add_argument('--val_freq', type=int, default=200)
        parser.add_argument('--pro_bar', type=str2bool, default='False')
        parser.add_argument('--model_save_freq', type=int, default=200)
        parser.add_argument('--validate_on_kitti_also', type=str2bool, default='False')
        parser.add_argument('--print_freq', type=int, default=1)
        parser.add_argument('--save_last_model',help="save last model also to resume training if crashed,stopped,etc", 
                            action='store_true')     
        parser.add_argument('--resume_from', help='the checkpoint file to resume from', default=None)
        parser.add_argument('--save_depths_gray', action='store_true')
        parser.add_argument('--save_depths_color', action='store_true') 
        parser.add_argument('--learning_rate_schedule', type=str)
        parser.add_argument('--gradient_accumulation', type=str2bool, default='False')
        parser.add_argument('--log_in_wandb', type=str2bool, default='False')
        parser.add_argument('--log_images_freq', type=int, default=50, help='batch frequency of logging images,depths during training to tensorboard')
        parser.add_argument('--finetune_on_another_dataset', type=str2bool, default='False')
        parser.add_argument('--pretrained_ckpt_path', type=str, default='')  
        
        return parser
