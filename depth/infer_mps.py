
import cv2
import numpy as np
import torch
from models.model import EcoDepth
from configs.infer_options import InferOptions
from utils import colorize_depth
import math

def predict(orig_img, model, device):
    # requires a numpy image of shape (h,w,3) with pixel values 0~255, the model and device
    # returns a numpy image representing the depth map with shape h, w
    # resize to a given shape
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR).astype(np.float32) 
    
    orig_img = orig_img/255.0
    orig_h,orig_w,_ = orig_img.shape
    max_area = 1000*720
    area = orig_h*orig_w
    ratio = math.sqrt(area/max_area)
    
    new_h = int(orig_h/ratio)
    new_w = int(orig_w/ratio)
    new_img = cv2.resize(orig_img, (new_w, new_h))
    
    # add padding to ensure img dimensions are multiples of 64
    add_h = 64-new_h%64
    add_w = 64-new_w%64
    
    final_h = new_h+add_h
    final_w = new_w+add_w
    
    final_img = np.zeros((final_h, final_w, 3))
    final_img[:new_h, :new_w, :] = new_img
    
    # convert to pytorch tensor, reshape and send to device
    final_img = torch.from_numpy(final_img)
    final_img = final_img.permute(2,0,1)
    final_img = final_img.unsqueeze(0)
    final_img = final_img.to(torch.float32).to(device)
    
    # flip images
    final_img_flipped = torch.flip(final_img, [3])
    final_img_concat = torch.cat([final_img, final_img_flipped])
    
    # change datatype from torch.float64 to torch.float32
    final_img_concat = final_img_concat.to(torch.float32)
    
    # send depth to model
    with torch.no_grad():
        final_depth_concat = model(final_img_concat)['pred_d']
    
    final_depth = final_depth_concat[0]
    final_depth_flipped = final_depth_concat[1]
    
    # take an average of the two predicted images
    final_depth = (final_depth+torch.flip(final_depth_flipped, [2]))/2
    
    # squeeze out extra batch and channel dimensions
    final_depth = final_depth.squeeze()
    
    # undo padding
    final_depth = final_depth[:new_h, :new_w]
    
    final_depth = final_depth.detach().cpu().numpy()

    # resize to original shape
    final_depth = cv2.resize(final_depth, (orig_w, orig_h))
    
    return final_depth

def visualize(img, depth):
    # requires a numpy array of shape (h,w,3) with pixel values 0~255 representing the RGB image
    # requires a numpy array of shape (h,w) representing the predicted depth
    # returns a side-by-side visualization of the image and depth map
    
    # obtain depth map using colorize_depth
    # take log of depth to put greater focus on nearer objects
    
    # remove the top portion and a little bottom part to get a better visualization 
    
    img = img[60:-20, :-20]

    depth_map = colorize_depth(np.log(depth))
    
    depth_map = depth_map[60:-20, :-20]
    
    # reverse the colour channel to get a better visual effect
    depth_map = depth_map[:, :, ::-1]
    
    # stack the img and depth horizontally with the img coming first
    viz = np.hstack((img, depth_map))
    
    return viz.astype(np.uint8)


def main():
    # set inference arguments and load model
    opt = InferOptions()
    args = opt.initialize().parse_args()
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model_weight = torch.load(args.ckpt_dir, map_location=device)['model']
    model_weight = {k: v.to(torch.float32) for k, v in model_weight.items()}
    model = EcoDepth(args=args)
    model.load_state_dict(model_weight)
    model = model.float()
    model.to(device)
    model.eval()

    # model is ready for inference
    if args.img_path is not None:
        print("Converting {} to a depth map".format(args.img_path))
        img_name = args.img_path[:-4]
        ext = args.img_path[-4:]
        # allow support for png or jpg images only
        assert ext == '.png' or ext == '.jpg'
        depth_name = img_name+'_depth'
        depth_path = depth_name+'.png'
        # read img
        img = cv2.imread(args.img_path)
        
        # get depth
        depth = predict(img, model, device)
        
        # get visualization
        viz = visualize(img, depth)
        
        # write visualization to file
        cv2.imwrite(depth_path, viz)
        
    if args.video_path is not None:
        print("Converting {} to a depth video".format(args.video_path))
        video_name = args.video_path[:-4]
        ext = args.video_path[-4:]
        # allow support for mp4 videos only
        assert ext == '.mp4'
        depth_name = video_name+'_depth'
        depth_path = depth_name+'.avi'
        # read img
        
        vidcap = cv2.VideoCapture(args.video_path)
        # read a frame from the video
        success, img = vidcap.read()
        h, w, _ = img.shape
        frame_rate = 30.0
        video = cv2.VideoWriter(depth_path, cv2.VideoWriter_fourcc(*"MJPG"), frame_rate, (2*w-40, h-80))
        
        while(success):
            # get depth
            depth = predict(img, model, device)
            
            # get visualization
            viz = visualize(img, depth)
            # write visualization to file
            video.write(viz)
        
            # read a frame from the video
            success, img = vidcap.read()

        video.release()
        
if __name__ == '__main__':
    main()