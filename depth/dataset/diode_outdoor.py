# ------------------------------------------------------------------------------
# The code is from VPD (https://github.com/wl-zhao/VPD).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

import os
import cv2
import numpy as np
import torch
from dataset.base_dataset import BaseDataset

class diode_outdoor(BaseDataset):
    def __init__(self, data_path, filenames_path='./dataset/filenames/',
                 is_train=True, crop_size=None, args=None):
        super().__init__(crop_size, args)

        self.is_train = is_train
        self.data_path = os.path.join(data_path, 'diode_outdoor_val')

        self.image_path_list = []
        self.depth_path_list = []

        txt_path = os.path.join(filenames_path, 'diode_outdoor_val')
        if is_train:
            txt_path += '/train_list.txt'
            self.data_path = self.data_path + '/sync'
        else:
            txt_path += '/val_list.txt'
            self.data_path = self.data_path + '/'
 
        self.filenames_list = self.readTXT(txt_path) # debug
        phase = 'train' if is_train else 'test'
        print("Dataset: DIODE OUTDOOR ")
        print("# of %s images: %d" % (phase, len(self.filenames_list)))

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        img_path = self.data_path + self.filenames_list[idx].split(' ')[0]
        gt_path = self.data_path + self.filenames_list[idx].split(' ')[1]
        filename = img_path.split('/')[-2] + '_' + img_path.split('/')[-1]

        #print("Evaluating Image: ", img_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        #depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32') # SUNRGBD Depth is stored in png files
        depth = np.load(gt_path).astype('float32') # DIODE Depth is stored in npy files

        ## RESIZE IMAGE TO NYUv2 shape
        # image = cv2.resize(image, (640, 480))
        # depth = cv2.resize(depth, (640, 480))

        if self.is_train:
            image, depth = self.augment_training_data(image, depth)
        else:
            image, depth = self.augment_test_data(image, depth)
        
        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(352, 640)).squeeze()
        depth = torch.nn.functional.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(352, 640)).squeeze()
        
        #depth = depth / (1000.0)  # convert in meters
        depth = depth # Depth scaling for diode_outdoor_val dataset
        
        return {'image': image, 'depth': depth, 'filename': filename}
