# ------------------------------------------------------------------------------
# The code is from VPD (https://github.com/wl-zhao/VPD).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from dataset.base_dataset import BaseDataset

class sunrgbd(BaseDataset):
    def __init__(self, data_path, filenames_path='./dataset/filenames/',
                 is_train=True, crop_size=None, args=None):
        super().__init__(crop_size, args)

        self.is_train = is_train
        self.data_path = os.path.join(data_path, 'sunrgbd')

        self.image_path_list = []
        self.depth_path_list = []

        txt_path = os.path.join(filenames_path, 'sunrgbd')
        if is_train:
            txt_path += '/train_list.txt'
            self.data_path = self.data_path + '/sync'
        else:
            txt_path += '/sunrgbd.txt'
            self.data_path = self.data_path + '/'
 
        self.filenames_list = self.readTXT(txt_path) # debug
        phase = 'train' if is_train else 'test'
        print("Dataset: SUNRGB-D (whole dataset)")
        print("# of %s images: %d" % (phase, len(self.filenames_list)))

        self.min_depth = 1e-3
        self.max_depth = 10
        self.depth_scale = 1000

    def __len__(self):
        return len(self.filenames_list)

    def eval_mask(self, depth_gt):
        depth_gt = np.squeeze(depth_gt)
        valid_mask = np.logical_and(depth_gt > self.min_depth, depth_gt < self.max_depth)
        
        ##########
        eval_mask = np.zeros(valid_mask.shape)
        eval_mask[45:471, 41:601] = 1
        valid_mask = np.logical_and(valid_mask, eval_mask)
        ###########

        valid_mask = np.expand_dims(valid_mask, axis=0)
        return valid_mask
    def __getitem__(self, idx):
        img_path = self.data_path + self.filenames_list[idx].split(' ')[0]
        gt_path = self.data_path + self.filenames_list[idx].split(' ')[1]
        # filename = img_path.split('/')[-2] + '_' + img_path.split('/')[-1]
        filename = "_".join(img_path.split("/")[-5:])
        # print("Evaluating Image: ", img_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.uint16) # SUNRGBD Depth is stored in png files
        #depth = np.load(gt_path).astype('float32') # DIODE Depth is stored in npy files

        ## RESIZE IMAGE TO NYUv2 shape
        # h, w, _ = image.shape
        # import ipdb;ipdb.set_trace();
        # if (h//3 >= w//4):
        #     new_w = (4*h)//3
        #     new_h = h
        # else:
        #     new_w = w
        #     new_h = (3*w)//4
        # if (new_h < h or new_w < w):
        #     new_h += 3
        #     new_w += 4
        # new_image = np.zeros((new_h, new_w, 3))
        # new_image[:h, :w, :] = image
        # new_depth = np.zeros((new_h, new_w))
        # new_depth[:h, :w] = depth
        # image = cv2.resize(new_image, (640, 480)).astype(np.uint8)
        # depth = cv2.resize(new_depth, (640, 480))
        # image = cv2.resize(image, (640, 480)).astype(np.uint8)
        # depth = cv2.resize(depth, (640, 480)).astype(np.uint16)
        
        # took from MDP toolbox
        depth = np.bitwise_or(np.right_shift(depth, 3), np.left_shift(depth, 16 - 3))
        depthInpaint = depth.astype(np.single) / self.depth_scale
        depthInpaint[depthInpaint > 8] = 8
        depth = depthInpaint.astype(np.float32)
        import ipdb;ipdb.set_trace();
        if self.is_train:
            image, depth = self.augment_training_data(image, depth)
        else:
            image, depth = self.augment_test_data(image, depth)
        #depth = depth / (1000.0)  # convert in meters
        # depth = depth / self.depth_scale # Depth scaling for sunrgbd dataset
        # print("Filename: ", img_path)
        
        return {'image': image, 'depth': depth, 'filename': filename}
