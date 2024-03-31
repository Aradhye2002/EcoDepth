
import os
import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from models.model import EcoDepth
import utils_depth.metrics as metrics
import utils_depth.logging as logging

from dataloaders.base_dataset import get_dataset
from configs.test_options import TestOptions
import utils
from tqdm import tqdm
from utils import colorize_depth, visualize, visualize_garg_crop_rectangle

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']

def get_scale(prediction, target,  min_depth, max_depth):
    """Returns the median scaling factor from gt_depth and pred_depth,
        Tells by what scale factor you should scale up(multipy) your pred_depth.
    """
    mask = np.logical_and(target>min_depth , target<max_depth)
    #edge case: pred_depth's median could be zero only if someone placed hands to block the camera. but gt-depth could be zero for far away objects.
    if np.median(prediction[mask])==0: 
        return 1
    scale = np.median(target[mask]) / np.median(prediction[mask])

    # if gt_depth's median is 0, then set the scale by seeing the 60th, 70th, 80th or 90th percentile
    if np.median(target[mask]) == 0: 
        if np.percentile(target[mask], 60):
            scale = np.percentile(target[mask], 60) / np.percentile(prediction[mask], 60)
        elif np.percentile(target[mask], 70):
            scale = np.percentile(target[mask], 70) / np.percentile(prediction[mask], 70)
        elif np.percentile(target[mask], 80):
            scale = np.percentile(target[mask], 80) / np.percentile(prediction[mask], 80)
        elif np.percentile(target[mask], 90):
            scale = np.percentile(target[mask], 90) / np.percentile(prediction[mask], 90)
        else:
            scale = 1
    return scale

def main():
    opt = TestOptions()
    args = opt.initialize().parse_args()
    # print(args)

    utils.init_distributed_mode_simple(args)
    print(args)
    
    device = torch.device(args.gpu)
    if args.exp_name == "":
        args.exp_name = "exp_name"
        
    model = EcoDepth(args=args)

    # CPU-GPU agnostic settings
    
    cudnn.benchmark = True
    model.to(device)
    
    from collections import OrderedDict
    model_weight = torch.load(args.ckpt_dir)['model']
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight, strict=True)
    model.eval()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)


    # Dataset setting
    dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path, 'args': args}

    val_dataset = get_dataset(**dataset_kwargs, is_train=False)


    sampler_val = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=utils.get_world_size(), rank=args.rank, shuffle=False)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, sampler=sampler_val,
                                             pin_memory=True)

    # Perform experiment
    results_dict = validate(val_loader, model,
                                        device=device, args=args)
    if args.rank == 0:
        result_lines = logging.display_result(results_dict)
        print(result_lines)

def blend_depths(left,right):       
    # the width of the overlapping region = (640-607) * 2 = 33*2 = 66
    overlapping_region_weights_left = torch.zeros((352, 64),device=left.device) + torch.linspace(1, 0, 64).to(device=left.device)
    overlapping_region_weights_right = torch.zeros((352, 64),device=left.device) + torch.linspace(0, 1, 64).to(device=left.device)
    left[:,-640:640] *= overlapping_region_weights_left
    right[:,-640:640] *= overlapping_region_weights_right
    blended_image = left + right
    return blended_image

def validate(val_loader, model, device, args):
    
    if args.save_eval_pngs or args.save_visualize:
        result_path = os.path.join(args.result_dir, args.dataset+"_"+args.exp_name)
        if args.rank == 0:
            logging.check_and_make_dirs(result_path)
        print("Saving result images in to %s" % result_path)
        if args.save_eval_pngs:
            os.makedirs(os.path.join(result_path,"depth_gray"),exist_ok=True)
        if args.save_visualize:
            os.makedirs(os.path.join(result_path, "depth_color"),exist_ok=True)

    if args.rank == 0:
        depth_loss = logging.AverageMeter()
    model.eval()

    ddp_logger = utils.MetricLogger()

    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0
    individual_rmse_errors = []
    logits_save_dir = "./logits/vit_logits_nyu"
    os.makedirs(logits_save_dir, exist_ok=True)

    scales_list = []

    for batch_idx, batch in tqdm(enumerate(val_loader)):
        input_RGB = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        filename = batch['filename'][0]
        # import ipdb;ipdb.set_trace()
        # class_id = batch['class_id']

        with torch.no_grad():
            if (args.dataset=="kitti" or args.dataset=="vkitti2") and args.kitti_split_to_half:
                # print("doing shift window test")
                left_part = input_RGB[:, :, :, :640]
                right_part = input_RGB[:, :, :, -640:]   
                input_RGB = torch.cat((left_part, right_part), dim=0)
            if args.flip_test:
                input_RGB = torch.cat((input_RGB, torch.flip(input_RGB, [3])), dim=0)
            
            #do padding
            desired_multiple_of = 64
            bs, _, old_h, old_w  = input_RGB.shape
            new_w = ((old_w-1)//desired_multiple_of + 1) * desired_multiple_of
            new_h = ((old_h-1)//desired_multiple_of + 1) * desired_multiple_of
            new_input_RGB = torch.zeros(bs, 3, new_h, new_w).to(input_RGB.device)
            new_input_RGB[:, :, :old_h, :old_w] = input_RGB
            input_RGB = new_input_RGB
            with torch.no_grad():
                pred = model(input_RGB)
            pred_d = pred['pred_d']
            #undo padding
            new_pred_d = pred_d[:, :, :old_h, :old_w]
            pred_d = new_pred_d
            input_RGB = input_RGB[:, :, :old_h, :old_w]
            
        if args.flip_test:
            batch_s = pred_d.shape[0]//2
            pred_d = (pred_d[:batch_s] + torch.flip(pred_d[batch_s:], [3]))/2.0 
        
        if (args.dataset=="kitti" or args.dataset=="vkitti2") and args.kitti_split_to_half:
            left_part = torch.zeros((352,1216),device=pred_d.device)
            right_part = torch.zeros((352,1216), device=pred_d.device)
            left_part[:, :640] = pred_d[0,0]
            right_part[:, -640:] = pred_d[1,0]
            pred_d = blend_depths(left_part, right_part)

        pred_d = pred_d.squeeze()
        depth_gt = depth_gt.squeeze()
        pred_d[pred_d < args.min_depth_eval] = args.min_depth_eval
        pred_d[pred_d > args.max_depth_eval] = args.max_depth_eval
        pred_d[torch.isinf(pred_d)] = args.max_depth_eval
        pred_d[torch.isnan(pred_d)] = args.min_depth_eval

        if args.median_scaling:
            scale = get_scale(pred_d.detach().cpu().numpy().squeeze(), depth_gt.detach().cpu().numpy().squeeze(), min_depth=args.min_depth_eval, max_depth=args.max_depth_eval )
            scales_list.append(scale)
            pred_d = pred_d * scale


        pred_crop, gt_crop = metrics.cropping_img(args, pred_d, depth_gt)
        computed_result = metrics.eval_depth(pred_crop, gt_crop)
        
        for key in result_metrics.keys():
            result_metrics[key] += computed_result[key]
        print_string = f"scale = {scale}, " if args.median_scaling else ""
        print_string += get_print_string(result_metrics, batch_idx+1)
        print(print_string)

        individual_rmse_errors.append([filename, computed_result['rmse']])

        if args.dataset == 'nyudepthv2':
            scale_factor = 1000.0
        elif args.dataset == "kitti":
            scale_factor = 256.0
        elif args.dataset == "vkitti2":
            scale_factor = 100.0
        elif args.dataset == "diode_outdoor":
            scale_factor = 100.0
        else:
            raise NotImplementedError
            
        if args.save_eval_pngs:
            save_path = os.path.join(result_path,"depth_gray", filename)
            
            if save_path.split('.')[-1] == 'jpg':
                save_path = save_path.replace('jpg', 'png')
            pred_d = pred_d.squeeze()
            pred_d = pred_d.cpu().numpy() * scale_factor

            cv2.imwrite(save_path, pred_d.astype(np.uint16),
                        [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if args.save_visualize:
            save_path = os.path.join(result_path, "depth_color", filename)
            pred_d_numpy = pred_d/scale_factor # convert back to metres
            if args.dataset=="nyudepthv2":
                eval_mask = np.zeros_like(pred_d_numpy)
                eval_mask[45:471, 41:601] = 1
                pred_d_numpy[eval_mask==0] = 0
                depth_gt = depth_gt.detach().cpu().numpy()
                vmin = min(depth_gt.min(), pred_d_numpy.min())
                # take vmax from percentile to avoid the erroneous max depth pixels.
                vmax = max(np.percentile(depth_gt, 98), np.percentile(pred_d_numpy, 98))
                # vmax = max(depth_gt.max(), pred_d_numpy.max())
                pred_d_color = colorize_depth(pred_d_numpy,vmin=vmin,vmax=vmax)
                depth_gt = colorize_depth(depth_gt, vmin=vmin, vmax=vmax)

                cv2.imwrite(save_path, np.hstack((depth_gt, pred_d_color))[:,:,::-1])
            elif args.dataset=="kitti":
                depth_gt = depth_gt.detach().cpu().numpy()
                vmin = depth_gt.min()
                vmax = depth_gt.max()
                valid_mask = depth_gt>0
                pred_d_color = colorize_depth(pred_d_numpy,vmin=vmin,vmax=vmax)
                # pred_d_color = visualize_garg_crop_rectangle(pred_d_color.copy())
                depth_gt = colorize_depth(depth_gt, vmin=vmin, vmax=vmax)
                depth_gt[valid_mask==0] = 0
                # depth_gt = visualize_garg_crop_rectangle(depth_gt.copy())

                img = batch["image"][0].cpu().numpy().transpose((1,2,0))*255
                # img = visualize_garg_crop_rectangle(img.copy())
                
                # stack all colored images and save
                viz = visualize(img, pred_d_color, depth_gt)
                cv2.imwrite(save_path, viz)
            else:
                raise NotImplementedError

        ddp_logger.update(**computed_result)
        
    if args.median_scaling:
        scale_stats = f"(min: {np.min(scales_list):.2f}, max: {np.max(scales_list):.2f}, mean: {np.mean(scales_list):.2f}, median: {np.median(scales_list):.2f})"
    else:
        scale_stats = "Nil"
    print("Median Scaling stats of scale_factor = ",scale_stats)

    ddp_logger.synchronize_between_processes()

    for key in result_metrics.keys():
        result_metrics[key] = ddp_logger.meters[key].global_avg

    return result_metrics

def get_print_string(computed_result,i):
    print_string = ""
    for key in computed_result.keys():
        print_string += f"{key} = {(computed_result[key]/i):.3f}, "
    return print_string




if __name__ == '__main__':
    main()