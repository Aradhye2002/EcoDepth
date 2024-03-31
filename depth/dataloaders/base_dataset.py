
import random
import importlib
import albumentations as A
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def get_dataset(dataset_name, **kwargs):
    dataset_name = dataset_name.lower()
    dataset_lib = importlib.import_module('.' + dataset_name, package='dataloaders')
    dataset_abs = getattr(dataset_lib, dataset_name)
    return dataset_abs(**kwargs)


class BaseDataset(Dataset):
    def __init__(self, args=None):
        self.count = 0
        self.args = args
        self.basic_transform = [
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue()    
        ]
        if args.dataset=="kitti" and args.kitti_split_to_half:
            self.basic_transform.append(A.RandomCrop(352, 640))
            
        self.to_tensor = transforms.ToTensor() # convert to tensor and scale img to [0,1]

    def readTXT(self, txt_path):
        with open(txt_path, 'r') as f:
            listInTXT = [line.strip() for line in f if "None" not in line]

        return listInTXT

    def augment_training_data(self, image, depth):
        H, W, C = image.shape
        if self.args.dataset == "nyudepthv2":
            # CutDepth paper: https://arxiv.org/abs/2107.07684
            if self.count % 4 == 0:
                """CutDepth"""
                alpha = random.random()
                beta = random.random()
                p = 0.75

                l = int(alpha * W)
                w = int(max((W - alpha * W) * beta * p, 1))

                image[:, l:l+w, 0] = depth[:, l:l+w]
                image[:, l:l+w, 1] = depth[:, l:l+w]
                image[:, l:l+w, 2] = depth[:, l:l+w]

        additional_targets = {'depth': 'mask'}
        aug = A.Compose(transforms=self.basic_transform,
                        additional_targets=additional_targets)
        augmented = aug(image=image, depth=depth)
        image = augmented['image']
        depth = augmented['depth']

        image = self.to_tensor(image)
        depth = self.to_tensor(depth).squeeze()
        self.count += 1
        return image, depth

    def augment_test_data(self, image, depth):
        image = self.to_tensor(image)
        depth = self.to_tensor(depth).squeeze()

        return image, depth

