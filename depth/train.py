
import os
import cv2
import numpy as np
from datetime import datetime
import wandb
import time
import torch
import math
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from models.model import EcoDepth
from models.optimizer import build_optimizers
import utils_depth.metrics as metrics
from utils_depth.criterion import SiLogLoss
import utils_depth.logging as logging

from dataloaders.base_dataset import get_dataset
from configs.train_options import TrainOptions
import utils
import glob
from utils import colorize_depth, seed_everything, cosine_annealing, visualize_garg_crop_rectangle
import matplotlib.pyplot as plt

os.environ['CURL_CA_BUNDLE'] = ''
metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']

log_dir = ""

def load_model(ckpt, model, optimizer=None):
    ckpt_dict = torch.load(ckpt, map_location='cpu')
    # keep backward compatibility
    if 'model' not in ckpt_dict and 'optimizer' not in ckpt_dict:
        state_dict = ckpt_dict
    else:
        state_dict = ckpt_dict['model']
    weights = {}

    for key, value in state_dict.items():
        if key.startswith('module.'):
            weights[key[len('module.'):]] = value
        else:
            weights[key] = value

    model.load_state_dict(weights)

    if optimizer is not None:
        optimizer_state = ckpt_dict['optimizer']
        optimizer.load_state_dict(optimizer_state)
        
    return ckpt_dict['epoch'], ckpt_dict['global_step'], ckpt_dict['results_dict']

def log_metric(tag, scalar_value, global_step, tboard_writer, args=None):
    if args is not None and args.log_in_wandb:
        try:
            wandb.log({tag: scalar_value}, step=global_step)
        except:
            print("wandb log failed for tag: ", tag)
    tboard_writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)

def train(train_loader, val_loader, model, model_without_ddp,  criterion_d, log_txt, optimizer, device, curr_epoch, args, tboard_writer):    
    global global_step
    model.train()
    if args.rank == 0:
        depth_loss = logging.AverageMeter()
    half_epoch = args.epochs // 2
    iterations_per_epoch = len(train_loader)
    result_lines = []

    grad_accum_iter = 2
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(train_loader):      
        global_step += 1

        if args.learning_rate_schedule == "one_cycle":
            if global_step < iterations_per_epoch * half_epoch:
                current_lr = (args.max_lr - args.min_lr) * (global_step / iterations_per_epoch/half_epoch) ** 0.9 + args.min_lr
            else:
                current_lr = max(args.min_lr, (args.min_lr - args.max_lr) * (global_step / iterations_per_epoch/half_epoch - 1) ** 0.9 + args.max_lr)
        elif args.learning_rate_schedule == "cosine":
            # Calculate the learning rate using a cosine learning rate schedule
            current_lr = args.min_lr + 0.5 * (args.max_lr - args.min_lr) * (1 + math.cos(math.pi * global_step / (iterations_per_epoch*args.epochs)))
        elif args.learning_rate_schedule == "cosine_annealing_then_linear_decay":
            if global_step < iterations_per_epoch * half_epoch:
                # Cyclic learning rate phase (e.g., cosine annealing)
                current_lr = args.min_lr + 0.5 * (args.max_lr - args.min_lr) * (1 + math.cos(math.pi * global_step / iterations_per_epoch))
            else:
                # Annealing phase (gradual decay)
                current_lr = args.max_lr - (args.max_lr - args.min_lr) * (global_step - iterations_per_epoch * half_epoch) / (iterations_per_epoch * half_epoch)
            # For the initial phase, it behaves like a cyclic learning rate (cosine annealing).
            # For the later phase, it gradually reduces the learning rate linearly.
        
        elif args.learning_rate_schedule == "linear_decay":
            # Linearly decay the learning rate from max_lr to min_lr
            current_lr = args.max_lr - (args.max_lr - args.min_lr) * (global_step / (iterations_per_epoch*args.epochs))
        elif args.learning_rate_schedule == "cosine_annealing_with_warm_restarts":
            n_cycles = 4 # for cosine annealing with warm restarts
            current_lr = cosine_annealing(global_step, iterations_per_epoch*args.epochs, n_cycles, args.max_lr)
        else:
            raise NotImplementedError

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr*param_group['lr_scale']


        input_RGB = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)

        # the stable diffusion pipeline requires image size (height and width) in multiples of 64
        # hence do padding
        desired_multiple_of = 64
        bs, _, old_h, old_w  = input_RGB.shape
        new_w = ((old_w-1)//desired_multiple_of + 1) * desired_multiple_of
        new_h = ((old_h-1)//desired_multiple_of + 1) * desired_multiple_of
        new_input_RGB = torch.zeros(bs, 3, new_h, new_w).to(input_RGB.device)
        new_input_RGB[:, :, :old_h, :old_w] = input_RGB
        input_RGB = new_input_RGB

        preds = model(input_RGB)
        
        pred_d = preds['pred_d']

        #undo padding
        new_pred_d = pred_d[:, :, :old_h, :old_w]
        pred_d = new_pred_d

        
        loss_d = criterion_d(pred_d.squeeze(dim=1), depth_gt)
        if args.gradient_accumulation:
            loss_d = loss_d / grad_accum_iter 

        if args.rank == 0:
            depth_loss.update(loss_d.item(), input_RGB.size(0))
        loss_d.backward()

        if args.gradient_accumulation:
            if ((batch_idx + 1) % grad_accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
            else:
                pass # accumulate the gradients.
        else:
            optimizer.step()
            optimizer.zero_grad()

        results_dict = {}

        # Log loss, metrics to wandb and training imgs, depths to Tensorboard.
        if args.rank == 0:
            if args.pro_bar:
                logging.progress_bar(batch_idx, len(train_loader), args.epochs, curr_epoch,
                                    ('Depth Loss: %.4f (%.4f)' %
                                    (depth_loss.val, depth_loss.avg)))
                
            # Log Training logs like train_loss, lr, etc to wandb.
            if batch_idx % args.print_freq == 0:
                result_line = 'Epoch: [{0}][{1}/{2}]\t Loss: {loss}, LR: {lr}\n'.format(
                        curr_epoch, batch_idx, iterations_per_epoch,
                        loss=depth_loss.avg, lr=current_lr
                    )
                result_lines.append(result_line)
                #Also log weights sum and average
                var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)
                var_avg = var_sum.item()/var_cnt;
                var_norm = [var.norm().item() for var in model.parameters() if var.requires_grad]
                log_metric(tag='var sum average', scalar_value=var_avg, global_step=global_step, tboard_writer=tboard_writer, args=args)
                log_metric(tag='var norm average', scalar_value=np.mean(var_norm), global_step=global_step, tboard_writer=tboard_writer, args=args)

                log_metric(tag="Training loss", scalar_value=depth_loss.avg, global_step=global_step, tboard_writer=tboard_writer, args=args)
                log_metric(tag='Learning Rate', scalar_value=current_lr, global_step=global_step, tboard_writer=tboard_writer, args=args)
                
            # Log images to tensorboard after every args.log_images_freq batches
            if batch_idx % args.log_images_freq == 0: 
                # depth_gt = torch.where(depth_gt < 1e-3, depth_gt * 0 + 1e3, depth_gt)
                num_log_images = args.batch_size
                for i in range(num_log_images):
                    try:
                        tboard_writer.add_image('depth_gt/image/{}'.format(i), colorize_depth(depth_gt[i].detach().cpu().numpy()[::2,::2]).transpose((2,0,1)), global_step)
                        tboard_writer.add_image('depth_est/image/{}'.format(i), colorize_depth(pred_d[i].detach().cpu().numpy().squeeze()[::2,::2]).transpose((2,0,1)), global_step)
                        tboard_writer.add_image('image/image/{}'.format(i), input_RGB[i, :, :old_h, :old_w].detach().cpu().numpy()[:,::2,::2], global_step)
                    except IndexError:
                        print("Index error occurred!")
                        with open(f"error_log_{args.model_name}.txt", "a") as file:
                            file.write(f"IndexError occurred in {args.model_name}:")
                            file.write('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}\n'.format(curr_epoch, batch_idx, iterations_per_epoch, global_step, current_lr, depth_loss.avg))
                tboard_writer.flush()
        # Validation 
        if global_step % args.val_freq == 0:
            start_time_valid = time.time()
            if args.rank==0:
                print(f"Doing Validation after global_step= {global_step}")
                print("\n",result_line)
            print(f'Doing validation on {args.dataset}:')
            results_dict, loss_val = validate(val_loader, model, criterion_d, device=device, curr_epoch=curr_epoch, args=args)

            if args.rank == 0: 
                # print validation results in terminal
                result_line = logging.display_result(results_dict)
                print(f"{args.dataset} validation results: \n",result_line)
                

                # log validation results to file
                with open(log_txt, 'a') as txtfile:
                    txtfile.write(f'\nEpoch: {curr_epoch:03d} / {args.epochs:03d} || Step {global_step%iterations_per_epoch} / {iterations_per_epoch}')
                    txtfile.write(result_line)

                # log validation results to wandb
                for each_metric, each_results in results_dict.items():
                    log_metric(tag=each_metric, scalar_value=each_results, global_step=global_step, tboard_writer=tboard_writer, args=args)
                
                log_metric(tag="epochh", scalar_value=curr_epoch, global_step=global_step, tboard_writer=tboard_writer, args=args)
                log_metric(tag="val loss", scalar_value=loss_val, global_step=global_step, tboard_writer=tboard_writer, args=args)


            end_time_valid = time.time()
            print(f"\n\nTime taken for validation: {(end_time_valid-start_time_valid)/60:.2f} minutes")
    
        # Model saving
        if args.rank == 0 and global_step % args.model_save_freq == 0:
            # always save if got best rmse or abs_rel                
            global best_rmse, best_abs_rel, log_dir
            if results_dict['rmse'] < best_rmse:
                model_save_name = f"best_rmse_{results_dict['rmse']:.4f}.ckpt"
                print(f"Saving best model as {model_save_name} in {log_dir}")
                if best_rmse < 1000:
                    os.remove(os.path.join(log_dir, f"best_rmse_{best_rmse:.4f}.ckpt"))
                best_rmse = results_dict['rmse']
                torch.save(
                    {
                        'model': model_without_ddp.state_dict(),
                        'epoch': curr_epoch,
                        'global_step': global_step,
                        'results_dict': results_dict,
                        'args': args
                    }, os.path.join(log_dir, model_save_name))
                
            if results_dict['abs_rel'] < best_abs_rel:
                model_save_name = f"best_abs_rel_{results_dict['abs_rel']:.4f}.ckpt"
                print(f"Saving best model as {model_save_name} in {log_dir}")
                if best_abs_rel < 1000:
                    os.remove(os.path.join(log_dir, f"best_abs_rel_{best_abs_rel:.4f}.ckpt"))
                best_abs_rel = results_dict['abs_rel']
                torch.save(
                    {
                        'model': model_without_ddp.state_dict(),
                        'epoch': curr_epoch,
                        'global_step': global_step,
                        'results_dict': results_dict,
                        'args': args
                    }, os.path.join(log_dir, model_save_name))
       
    #also log loss,etc in logs.txt file for each step.
    if args.rank == 0:
        with open(log_txt, 'a') as txtfile:
            txtfile.write('\nEpoch: %03d / %03d' % (curr_epoch, args.epochs))
            for result_line in result_lines:
                txtfile.write(result_line)   

    return results_dict

# we use blending when doing validation on kitti
def blend_depths(left,right):       
    overlapping_region_weights_left = torch.zeros((352, 64),device=left.device) + torch.linspace(1, 0, 64).to(device=left.device)
    overlapping_region_weights_right = torch.zeros((352, 64),device=left.device) + torch.linspace(0, 1, 64).to(device=left.device)
    left[:,-640:640] *= overlapping_region_weights_left
    right[:,-640:640] *= overlapping_region_weights_right
    blended_image = left + right
    return blended_image

def validate(val_loader, model, criterion_d, device, curr_epoch, args):
    global last_save_epoch
    if args.rank == 0 and args.save_depths_gray and curr_epoch > last_save_epoch:
        global log_dir
        print("log_dir in validate()",log_dir)
        depth_gray_path = os.path.join(log_dir, "depth_gray")
        os.makedirs(depth_gray_path, exist_ok=True)
        print("Will save depth in gray (orig) in to %s" % depth_gray_path)
        
    if args.rank == 0 and args.save_depths_color and curr_epoch > last_save_epoch:
        depth_color_path = os.path.join(log_dir, "depth_color")
        os.makedirs(depth_color_path, exist_ok=True)
        print("Will save depth visualizations in to %s" % depth_color_path)
    
    if args.rank == 0:
        depth_loss = logging.AverageMeter()
    model.eval()

    ddp_logger = utils.MetricLogger()

    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    for batch_idx, batch in enumerate(val_loader):
        input_RGB = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        filename = batch['filename'][0]

        with torch.no_grad():
            bs, _, h, w = input_RGB.shape
            if args.dataset=="kitti" and args.kitti_split_to_half:
                left_part = input_RGB[:, :, :, :640]
                right_part = input_RGB[:, :, :, -640:]   
                input_RGB = torch.cat((left_part, right_part), dim=0)
            if args.flip_test:
                input_RGB = torch.cat((input_RGB, torch.flip(input_RGB, [3])), dim=0)

            # padding
            desired_multiple_of = 64
            bs, _, old_h, old_w  = input_RGB.shape
            new_w = ((old_w-1)//desired_multiple_of + 1) * desired_multiple_of
            new_h = ((old_h-1)//desired_multiple_of + 1) * desired_multiple_of
            new_input_RGB = torch.zeros(bs, 3, new_h, new_w).to(input_RGB.device)
            new_input_RGB[:, :, :old_h, :old_w] = input_RGB
            input_RGB = new_input_RGB

            pred = model(input_RGB)
            pred_d = pred['pred_d']

            #undo padding
            new_pred_d = pred_d[:, :, :old_h, :old_w]
            pred_d = new_pred_d
            input_RGB = input_RGB[:, :, :old_h, :old_w]

        if args.flip_test:
            batch_s = pred_d.shape[0]//2
            pred_d = (pred_d[:batch_s] + torch.flip(pred_d[batch_s:], [3]))/2.0

        if args.dataset=="kitti" and args.kitti_split_to_half:
            left_part = torch.zeros((352,1216),device=pred_d.device)
            right_part = torch.zeros((352,1216), device=pred_d.device)
            left_part[:, :640] = pred_d[0,0]
            right_part[:, -640:] = pred_d[1,0]
            pred_d = blend_depths(left_part, right_part)
            
        pred_d = pred_d.squeeze()
        depth_gt = depth_gt.squeeze()

        loss_d = criterion_d(pred_d.squeeze(), depth_gt)
        ddp_logger.update(loss_d=loss_d.item())
        if args.rank == 0:
            depth_loss.update(loss_d.item(), input_RGB.size(0))

        pred_crop, gt_crop = metrics.cropping_img(args, pred_d, depth_gt)
        computed_result = metrics.eval_depth(pred_crop, gt_crop)

        if args.rank == 0 and curr_epoch > last_save_epoch:
            
            if args.save_depths_gray:
                save_path = os.path.join(depth_gray_path, filename)
                save_path = save_path.replace('jpg', 'png') if save_path.split('.')[-1] == 'jpg' else save_path
                pred_d_numpy = pred_d.cpu().numpy()
                if args.dataset == 'kitti':
                    pred_d_numpy = pred_d_numpy * 256.0
                    cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16),[cv2.IMWRITE_PNG_COMPRESSION, 0])
                else: #NYUv2
                    pred_d_numpy = pred_d_numpy * 1000.0
                    cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    
            if args.save_depths_color:
                save_path = os.path.join(depth_color_path, filename)
                gt = depth_gt.cpu().numpy()
                pred = pred_d.cpu().numpy()
                if args.dataset == 'nyudepthv2':
                    eval_mask = torch.zeros(gt.shape)
                    eval_mask[45:471, 41:601] = 1 # eigen_crop
                    pred[eval_mask==0] = 0
                
                valid_mask = gt>0
                gt = colorize_depth(gt)
                gt[valid_mask==0] = 0
                pred = colorize_depth(pred,vmin=depth_gt.min().item(),vmax=depth_gt.max().item())

                if args.dataset == "kitti":
                    gt = visualize_garg_crop_rectangle(gt.copy())
                    pred = visualize_garg_crop_rectangle(pred.copy())

                img = (batch["image"][0].cpu().numpy().transpose((1,2,0))*255).astype(np.uint8)
                viz = np.vstack((img, gt, pred))
                plt.imsave(save_path[:],viz)
                        
        if args.rank == 0:
            loss_d = depth_loss.avg
            if args.pro_bar:
                logging.progress_bar(batch_idx, len(val_loader), args.epochs, curr_epoch)

        ddp_logger.update(**computed_result)
        for key in result_metrics.keys():
            result_metrics[key] += computed_result[key]

    ddp_logger.synchronize_between_processes()
    last_save_epoch = curr_epoch

    for key in result_metrics.keys():
        result_metrics[key] = ddp_logger.meters[key].global_avg

    loss_d = ddp_logger.meters['loss_d'].global_avg

    return result_metrics, loss_d

def main():
    seed_everything()
    opt = TrainOptions()
    args = opt.initialize().parse_args()
    
    utils.init_distributed_mode_simple(args)
    device = torch.device(args.gpu)
    if args.rank == 0:
        print(f"\n Max_depth = {args.max_depth} meters for {args.dataset}!\n")

    # Keep a name for log directory
    dataset_name = args.dataset[:3]
    name = [dataset_name, "BS-"+str(args.batch_size), "lr-"+str(args.learning_rate_schedule)]
    if args.exp_name != '':
        name.append(args.exp_name)
    exp_name = '_'.join(name)
        
    # Logging   
    if args.rank == 0:

        if args.log_in_wandb:
            wandb.init(project='EcoDepth', name ="")

        print("\nmodel will be saved after every %d steps" % args.model_save_freq)
        print("val will be done after every %d steps" % args.val_freq)

        # make a log directory
        # make logs.txt, log changed_files in "files", and log some depth-samples.
        exp_name = '%s_%s' % (datetime.now().strftime('%m%d%H%M'), exp_name)
        print('This experiment name is : ', exp_name)
        global log_dir
        log_dir = os.path.join(args.log_dir, exp_name)
        os.makedirs(log_dir,exist_ok=True)
        print("log_dir in main",log_dir)

        # make logs.txt
        log_txt = os.path.join(log_dir, 'logs.txt')  
        logging.log_args_to_txt(log_txt, args)
        
        file_dir = os.path.join(log_dir, 'files')
        os.makedirs(file_dir,exist_ok=True)
        os.system("cp *.py "+file_dir)
        os.system("cp *.sh "+file_dir)
        command = "mkdir -p "+os.path.join(file_dir, 'configs') + '&& cp configs/*.py ' + os.path.join(file_dir, 'configs')
        os.system(command)
        command = "mkdir -p "+os.path.join(file_dir, 'dataloaders') + '&& cp dataloaders/*.py ' + os.path.join(file_dir, 'dataloaders')
        os.system(command)
        command = "mkdir -p "+os.path.join(file_dir, 'models') + '&& cp models/*.py ' + os.path.join(file_dir, 'models')
        os.system(command)
        command = "mkdir -p "+os.path.join(file_dir, 'utils_depth') + '&& cp utils_depth/*.py ' + os.path.join(file_dir, 'utils_depth')
        os.system(command)
    else:
        log_txt = None
        log_dir = None  
        
    model = EcoDepth(args=args)

    cudnn.benchmark = True
    model.to(device)
    model_without_ddp = model
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)

    # Dataset setting
    dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path, 'args': args}
    train_dataset = get_dataset(**dataset_kwargs)
    val_dataset = get_dataset(**dataset_kwargs, is_train=False)
    
    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=utils.get_world_size(), rank=args.rank, shuffle=True, 
    )

    sampler_val = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=utils.get_world_size(), rank=args.rank, shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               sampler=sampler_train, num_workers=args.workers, 
                                               pin_memory=True, drop_last=True, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, sampler=sampler_val, num_workers=args.workers,
                                             pin_memory=True)
   
    # Training settings
    criterion_d = SiLogLoss(args=args)

    optimizer = build_optimizers(model, dict(type='AdamW', lr=args.max_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay,
                constructor='LDMOptimizerConstructor',
                paramwise_cfg=dict(layer_decay_rate=args.layer_decay, no_decay_names=['relative_position_bias_table', 'rpe_mlp', 'logit_scale'])))
    start_ep = 1
    global global_step, best_rmse, best_abs_rel
    global_step = 0
    best_rmse = 1000
    best_abs_rel = 1000

    if args.resume_from: # if training crashed in b/w then resume from the last checkpoint.
        resume_ep, global_step, results_dict = load_model(args.resume_from, model.module, optimizer)
        print(f'resumed from epoch {resume_ep}, ckpt {args.resume_from}')
        best_rmse = results_dict['rmse'] 
        best_abs_rel = results_dict['abs_rel']
        start_ep = resume_ep + 1
    
    if args.finetune_on_another_dataset:
        resume_ep, global_step, results_dict = load_model(args.pretrained_ckpt_path, model.module, optimizer=None)
        print(f'Pretrained model stats: \nEpoch={resume_ep}, global_step={global_step}, results_dict={results_dict}, ckpt {args.pretrained_ckpt_path}')
        start_ep = 1
        global_step = 0


    tboard_writer = SummaryWriter(log_dir + '/tboard_logs', flush_secs=30) if args.rank==0 else None
    
    global last_save_epoch
    last_save_epoch = -1
    
    # Perform experiment
    for curr_epoch in range(start_ep, args.epochs + 1):
        print('\nEpoch: %03d / %03d' % (curr_epoch, args.epochs))
        #train for 1 epoch
        start_time_epoch = time.time()
        results_dict = train(train_loader, val_loader, model, model_without_ddp, criterion_d, log_txt, optimizer=optimizer, device=device, curr_epoch=curr_epoch, args=args, tboard_writer=tboard_writer)
        end_time_epoch = time.time()
        print(f"\n\nTime taken for 1 epoch: {(end_time_epoch-start_time_epoch)/60:.2f} minutes")

        if args.rank==0 and args.save_last_model:
            print("Saving latest model as last.ckpt in %s" % (log_dir))
            torch.save(
                {
                    'model': model_without_ddp.state_dict(),
                    'epoch': curr_epoch,
                    'global_step': global_step,
                    'results_dict': results_dict,
                    'optimizer': optimizer.state_dict(),
                    'args': args
                }, os.path.join(log_dir, f'last.ckpt'))

if __name__ == '__main__':
    main()


