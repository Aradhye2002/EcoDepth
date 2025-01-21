import random
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import os

def readTXT(txt_path):
    with open(txt_path, 'r') as f:
        listInTXT = [line.strip() for line in f if "None" not in line]
    return listInTXT

class DepthDataset(Dataset):
    def __init__(self, args, is_train, filenames_path, data_path, depth_factor):
        super().__init__()
        self.is_train = is_train
        self.args = args
        self.data_path = data_path
        self.depth_factor = depth_factor
        
        ##### Transforms ######
        self.train_transforms = [
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue()
        ]
        if args.do_random_crop:
            self.train_transforms.append(
                A.RandomCrop(args.crop_h, args.crop_w)
            )
        # register count for using CutDepth
        if args.use_cut_depth:
            self.count = 0
        self.to_tensor = transforms.ToTensor()
        
        ##### Filenames ######
        self.filenames = readTXT(filenames_path)
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        rgb_path, depth_path = self.filenames[idx].split()[:2]
        rgb_path = os.path.join(self.data_path, rgb_path)
        depth_path = os.path.join(self.data_path, depth_path)
        image = cv2.imread(rgb_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype('float32')/self.depth_factor
        image, depth = self.apply_transforms(image, depth)
        return {"image" : image, "depth" : depth}
        
    def apply_cut_depth(self, image, depth):
        # image must have shape (H, W, C)
        # depth must have shape (H, W)
        H, W, C = image.shape
        if self.count % 4 == 0:
            alpha = random.random()
            beta = random.random()
            p = 0.75

            l = int(alpha * W)
            w = int(max((W - alpha * W) * beta * p, 1))

            image[:, l:l+w, 0] = depth[:, l:l+w]
            image[:, l:l+w, 1] = depth[:, l:l+w]
            image[:, l:l+w, 2] = depth[:, l:l+w]
        self.count += 1
        return image, depth
    
    def apply_transforms(self, image, depth):
        # image must be a numpy array in range [0, 255] and of shape (H, W, C)
        # depth must be a numpy array of shape (H, W)
        if self.is_train:
            image, depth = self.apply_train_transforms(image, depth)
        else:
            image, depth = self.apply_test_transforms(image, depth)
        image = self.to_tensor(image)
        depth = self.to_tensor(depth)
        # image is a torch tensor in range [0, 1] and of shape (C, H, W)
        # depth is a torch tensor and of shape (C, H, W)
        return image, depth
    
    def apply_test_transforms(self, image, depth):
        return image, depth
    
    def apply_train_transforms(self, image, depth):
        H, W, C = image.shape
        # do padding in case image dimensions 
        # are smaller than crop size
        if self.args.do_random_crop:
            if H < self.args.crop_h:
                image = np.concatenate([image, np.zeros((self.args.crop_h-H, W, 3), dtype=np.uint8)], axis=0)
                depth = np.concatenate([depth, np.zeros((self.args.crop_h-H, W),  dtype=np.uint8)], axis=0)
            if W < self.args.crop_w:
                image = np.concatenate([image, np.zeros((self.args.crop_h, self.args.crop_w-W, 3),  dtype=np.uint8)], axis=0)
                depth = np.concatenate([depth, np.zeros((self.args.crop_h, self.args.crop_w-W),  dtype=np.uint8)], axis=0)

        additional_targets = {"depth" : "mask"}
        aug = A.Compose(
            transforms=self.train_transforms,
            additional_targets=additional_targets
        )
        augmented = aug(image=image, depth=depth)
        image, depth = augmented["image"], augmented["depth"]
        if self.args.use_cut_depth:
            image, depth = self.apply_cut_depth(image, depth)
        return image, depth
    