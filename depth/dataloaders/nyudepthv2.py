
import os
import cv2
import numpy as np
from dataloaders.base_dataset import BaseDataset

class nyudepthv2(BaseDataset):
    def __init__(self, data_path, filenames_path='./dataloaders/filenames/',
                 is_train=True, crop_size=None, args=None):
        super().__init__(args)

        self.is_train = is_train
        self.data_path = os.path.join(data_path, 'nyu_depth_v2')
        self.args = args
        
        txt_path = os.path.join(filenames_path, 'nyudepthv2')
        if is_train:
            txt_path += '/train_list.txt'
            self.data_path = self.data_path + '/sync'
        else:
            txt_path += '/test_list.txt'
            self.data_path = self.data_path + '/official_splits/test/'
 
        self.filenames_list = self.readTXT(txt_path)#[:16] # debug
        phase = 'train' if is_train else 'test'
        print("Dataset: NYU Depth V2")
        print("# of %s images: %d" % (phase, len(self.filenames_list)))

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        img_path = self.data_path + self.filenames_list[idx].split(' ')[0]
        gt_path = self.data_path + self.filenames_list[idx].split(' ')[1]
        filename = img_path.split('/')[-2] + '_' + img_path.split('/')[-1]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')
        if self.args.eigen_crop_in_dataloader_itself_for_nyu:
            # taken from pixelformer, the coords are little different from eigen_crop.
            valid_mask = np.zeros_like(depth)
            valid_mask[45:471, 41:601] = 1 # I did it same as eigen_crop, pixelf,newcrf,etc are doing it little more, i.e.,[45:472, 43:608]
            depth[valid_mask==0] = 0    
            
        if self.is_train:
            image, depth = self.augment_training_data(image, depth)
        else:
            image, depth = self.augment_test_data(image, depth)
        depth = depth / (1000.0)  # convert in meters
        return {'image': image, 'depth': depth, 'filename': filename}