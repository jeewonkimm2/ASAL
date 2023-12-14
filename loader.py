import glob
import os
from PIL import Image, ImageFilter

from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import numpy as np
import random
import cv2
import torchvision.transforms.functional as F


# CutPaste Object

# label_dict = {'bottle': {'good': 0, 'anomaly': 1},
#                 'cable': {'good': 0, 'anomaly': 1},
#                 'capsule': {'good': 0, 'anomaly': 1},
#                 'carpet': {'good': 0, 'anomaly': 1},
#                 'grid': {'good': 0, 'anomaly': 1},
#                 'hazelnut': {'good': 0, 'anomaly': 1},
#                 'metal_nut': {'good': 0, 'anomaly': 1},
#                 'screw': {'good': 0, 'anomaly': 1},
#                 'zipper': {'good': 0, 'anomaly': 1},
#                 'leather': {'good': 0, 'anomaly': 1},
#                 'pill': {'good': 0, 'anomaly': 1},
#                 'tile': {'good': 0, 'anomaly': 1},
#                 'toothbrush': {'good': 0, 'anomaly': 1},
#                 'transistor': {'good': 0, 'anomaly': 1},
#                 'wood': {'good': 0, 'anomaly': 1}
#             } 

class RotationLoader(Dataset):
    def __init__(self, is_train=True, transform=None, path='./DATA2', path_list=None, class_name = None, label_dict = None):
        self.is_train = is_train
        self.transform = transform
        self.path_list = path_list
        self.class_name = class_name
        self.label_dict = label_dict
        # self.h_flip = transforms.RandomHorizontalFlip(p=1)
        if self.is_train == 0: # train
            self.img_path = glob.glob(f'./DATA2/{class_name}/train/*/*')
        else:
            self.img_path = glob.glob(f'./DATA2/{class_name}/train/*/*')
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)

        if self.is_train:
            img = self.transform(img)
            img1 = torch.rot90(img, 1, [1,2])
            img2 = torch.rot90(img, 2, [1,2])
            img3 = torch.rot90(img, 3, [1,2])
            imgs = [img, img1, img2, img3]
            rotations = [0,1,2,3]
            random.shuffle(rotations)
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3]
        else:
            img = self.transform(img)
            img1 = torch.rot90(img, 1, [1,2])
            img2 = torch.rot90(img, 2, [1,2])
            img3 = torch.rot90(img, 3, [1,2])
            imgs = [img, img1, img2, img3]
            rotations = [0,1,2,3]
            random.shuffle(rotations)
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3], self.img_path[idx]

class RotationLoader_test(Dataset):
    def __init__(self, is_train=True, transform=None, path='./DATA2', path_list=None, class_name = None, label_dict = None):
        self.is_train = is_train
        self.transform = transform
        self.path_list = path_list
        self.class_name = class_name
        self.label_dict = label_dict
        # self.h_flip = transforms.RandomHorizontalFlip(p=1)
        if self.is_train == 0: # train
            self.img_path = glob.glob(f'./DATA2/{class_name}/test/*/*')
        else:
            self.img_path = glob.glob(f'./DATA2/{class_name}/test/*/*')
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)

        if self.is_train:
            img = self.transform(img)
            img1 = torch.rot90(img, 1, [1,2])
            img2 = torch.rot90(img, 2, [1,2])
            img3 = torch.rot90(img, 3, [1,2])
            imgs = [img, img1, img2, img3]
            rotations = [0,1,2,3]
            random.shuffle(rotations)
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3]
        else:
            img = self.transform(img)
            img1 = torch.rot90(img, 1, [1,2])
            img2 = torch.rot90(img, 2, [1,2])
            img3 = torch.rot90(img, 3, [1,2])
            imgs = [img, img1, img2, img3]
            rotations = [0,1,2,3]
            random.shuffle(rotations)
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3], self.img_path[idx]

class Loader2(Dataset):
    def __init__(self, is_train=True, transform=None, path='./DATA', path_list=None, class_name = None, label_dict = None):
        self.is_train = is_train
        self.transform = transform
        self.path_list = path_list
        self.class_name = class_name
        self.label_dict = label_dict

        if self.is_train: # train
            self.img_path = path_list
        else:
            if path_list is None:
                self.img_path = glob.glob(f'./DATA/{class_name}/train/*/*') # for loss extraction
            else:
                self.img_path = path_list

                
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        if self.is_train:
            img = cv2.imread(self.img_path[idx][:-1])
        else:
            if self.path_list is None:
                img = cv2.imread(self.img_path[idx])
            else:
                img = cv2.imread(self.img_path[idx][:-1])
        img = Image.fromarray(img)
        img = self.transform(img)
        label = self.label_dict[self.img_path[idx].split('/')[-2]]

        return img, label

class Loader(Dataset):
    def __init__(self, is_train=True, transform=None, path='./DATA', class_name=None, label_dict=None):
        

        # self.classes = len(label_dict[class_name])
        self.is_train = is_train
        self.transform = transform
        self.class_name = class_name
        self.label_dict = label_dict

        if self.is_train: # train
            self.img_path = glob.glob(f'./DATA/{class_name}/train/*/*')
        else:
            self.img_path = glob.glob(f'./DATA/{class_name}/test/*/*')

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)
        img = self.transform(img)
        label = self.label_dict[self.img_path[idx].split('/')[-2]] 
        
        return img, label

class MvtecLoader_3way(Dataset):
    def __init__(self, is_train=True, transform=None, path='./DATA', class_name = None):
        self.is_train = is_train
        self.transform = transform
        self.make_cutpaste = CutPaste

        if self.is_train == 0: # train
            self.img_path = glob.glob(f'./DATA/{class_name}/train/*/*')
        else:
            self.img_path = glob.glob(f'./DATA/{class_name}/train/*/*')
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)

        img1 = np.array(make_cutpaste.cutpaste(Image.open(self.img_path[idx])))
        img2 = np.array(make_cutpaste.cutpaste(Image.open(self.img_path[idx])))

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        # cutpaste coded needed here
        if self.is_train:
            img = self.transform(img)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            imgs = [img, img1, img2]
            cutpates = [0,1,2]
            random.shuffle(cutpates)
            return imgs[cutpates[0]], imgs[cutpates[1]], imgs[cutpates[2]], cutpates[0], cutpates[1], cutpates[2]
        
        else:
            img = self.transform(img)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            imgs = [img, img1, img2]
            cutpates = [0,1,2]
            random.shuffle(cutpates)
            return imgs[cutpates[0]], imgs[cutpates[1]], imgs[cutpates[2]], cutpates[0], cutpates[1], cutpates[2],  self.img_path[idx]

class MvtecLoader_3way_test(Dataset):
    def __init__(self, is_train=True, transform=None, path='./DATA2', class_name = None):
        self.is_train = is_train
        self.transform = transform
        self.make_cutpaste = CutPaste

        if self.is_train == 0: # train
            self.img_path = glob.glob(f'./DATA2/{class_name}/test/*/*')
        else:
            self.img_path = glob.glob(f'./DATA2/{class_name}/test/*/*')
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)

        img1 = np.array(make_cutpaste.cutpaste(Image.open(self.img_path[idx])))
        img2 = np.array(make_cutpaste.cutpaste(Image.open(self.img_path[idx])))

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        # cutpaste coded needed here
        if self.is_train:
            img = self.transform(img)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            imgs = [img, img1, img2]
            cutpates = [0,1,2]
            random.shuffle(cutpates)
            return imgs[cutpates[0]], imgs[cutpates[1]], imgs[cutpates[2]], cutpates[0], cutpates[1], cutpates[2]
        
        else:
            img = self.transform(img)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            imgs = [img, img1, img2]
            cutpates = [0,1,2]
            random.shuffle(cutpates)
            return imgs[cutpates[0]], imgs[cutpates[1]], imgs[cutpates[2]], cutpates[0], cutpates[1], cutpates[2],  self.img_path[idx]
        


class MvtecLoader_5(Dataset):
    def __init__(self, is_train=True, transform=None, path='./DATA', class_name = None):
        self.is_train = is_train
        self.transform = transform
        self.make_cutpaste = CutPaste

        if self.is_train == 0: # train
            self.img_path = glob.glob(f'./DATA/{class_name}/train/*/*')
        else:
            self.img_path = glob.glob(f'./DATA/{class_name}/train/*/*')
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)

        img1 = np.array(make_cutpaste.cutpaste(Image.open(self.img_path[idx])))
        img2 = np.array(make_cutpaste.cutpaste(Image.open(self.img_path[idx])))

        img3 = np.array(make_cutpaste.cutpaste_scar(Image.open(self.img_path[idx])))
        img4 = np.array(make_cutpaste.cutpaste_scar(Image.open(self.img_path[idx])))


        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        img3 = Image.fromarray(img3)
        img4 = Image.fromarray(img4)


        # cutpaste coded needed here
        if self.is_train:
            img = self.transform(img)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)
            imgs = [img, img1, img2, img3, img4]
            cutpates = [0,1,2,3,4]
            random.shuffle(cutpates)
            return imgs[cutpates[0]], imgs[cutpates[1]], imgs[cutpates[2]], imgs[cutpates[3]], imgs[cutpates[4]], cutpates[0], cutpates[1], cutpates[2], cutpates[3], cutpates[4]
        
        else:
            img = self.transform(img)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)
            imgs = [img, img1, img2, img3, img4]
            cutpates = [0,1,2,3,4]
            random.shuffle(cutpates)
            return imgs[cutpates[0]], imgs[cutpates[1]], imgs[cutpates[2]], imgs[cutpates[3]], imgs[cutpates[4]], cutpates[0], cutpates[1], cutpates[2], cutpates[3], cutpates[4], self.img_path[idx]

class MvtecLoader_2way(Dataset):
    def __init__(self, is_train=True, transform=None, path='./DATA', class_name = None):
        self.is_train = is_train
        self.transform = transform
        self.make_cutpaste = CutPaste

        if self.is_train == 0: # train
            self.img_path = glob.glob(f'./DATA/{class_name}/train/*/*')
        else:
            self.img_path = glob.glob(f'./DATA/{class_name}/train/*/*')
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)

        img1 = np.array(make_cutpaste.cutpaste(Image.open(self.img_path[idx])))

        img1 = Image.fromarray(img1)

        # cutpaste coded needed here
        if self.is_train:
            img = self.transform(img)
            img1 = self.transform(img1)
            imgs = [img, img1]
            cutpates = [0,1]
            random.shuffle(cutpates)
            return imgs[cutpates[0]], imgs[cutpates[1]], cutpates[0], cutpates[1]
        
        else:
            img = self.transform(img)
            img1 = self.transform(img1)
            imgs = [img, img1]
            cutpates = [0,1]
            random.shuffle(cutpates)
            return imgs[cutpates[0]], imgs[cutpates[1]], cutpates[0], cutpates[1], self.img_path[idx]
        
class MvtecLoader_2way_test(Dataset):
    def __init__(self, is_train=True, transform=None, path='./DATA', class_name = None):
        self.is_train = is_train
        self.transform = transform
        self.make_cutpaste = CutPaste

        if self.is_train == 0: # train
            self.img_path = glob.glob(f'./DATA2/{class_name}/test/*/*')
        else:
            self.img_path = glob.glob(f'./DATA2/{class_name}/test/*/*')
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)

        img1 = np.array(make_cutpaste.cutpaste(Image.open(self.img_path[idx])))

        img1 = Image.fromarray(img1)

        # cutpaste coded needed here
        if self.is_train:
            img = self.transform(img)
            img1 = self.transform(img1)
            imgs = [img, img1]
            cutpates = [0,1]
            random.shuffle(cutpates)
            return imgs[cutpates[0]], imgs[cutpates[1]], cutpates[0], cutpates[1]
        
        else:
            img = self.transform(img)
            img1 = self.transform(img1)
            imgs = [img, img1]
            cutpates = [0,1]
            random.shuffle(cutpates)
            return imgs[cutpates[0]], imgs[cutpates[1]], cutpates[0], cutpates[1], self.img_path[idx]
        

class RandomMaskingLoader(Dataset):
    def __init__(self, is_train=True, transform=None, path='./DATA', class_name = None):
        self.is_train = is_train
        self.transform = transform
        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])
        self.chkpt_dir = './chkpt_dir/mae_visualize_vit_large.pth'
        self.model_mae = RandomMasking.prepare_model(self.chkpt_dir, 'mae_vit_large_patch16') 

        if self.is_train == 0: # train
            self.img_path = glob.glob(f'./DATA/{class_name}/train/*/*')
        else:
            self.img_path = glob.glob(f'./DATA/{class_name}/train/*/*')
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)

        # img1 = img

        img1 = Image.open(self.img_path[idx])
        img1 = img1.resize((224, 224))
        img1 = np.array(img1) / 255.

        assert img1.shape == (224, 224, 3)

        # normalize by ImageNet mean and std
        img1 = img1 - self.imagenet_mean
        img1 = img1 / self.imagenet_std

        _, img1 = RandomMasking.run_one_image(img1, self.model_mae)

        # img1을 NumPy ndarray로 변환 후, 이미지의 범위를 [0, 255]로 스케일링
        img1 = (img1.squeeze().numpy() * 255).astype(np.uint8)
        img1 = Image.fromarray(img1)

        # cutpaste coded needed here
        if self.is_train:
            img = self.transform(img)
            img1 = self.transform(img1)
            imgs = [img, img1]
            cutpates = [0,1]
            random.shuffle(cutpates)
            return imgs[cutpates[0]], imgs[cutpates[1]], cutpates[0], cutpates[1], self.img_path[idx]
        
        else:
            img = self.transform(img)
            img1 = self.transform(img1)
            imgs = [img, img1]
            cutpates = [0,1]
            random.shuffle(cutpates)
            return imgs[cutpates[0]], imgs[cutpates[1]], cutpates[0], cutpates[1], self.img_path[idx]
        

class RandomMaskingLoader_test(Dataset):
    def __init__(self, is_train=True, transform=None, path='./DATA', class_name = None):
        self.is_train = is_train
        self.transform = transform
        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])
        self.chkpt_dir = './chkpt_dir/mae_visualize_vit_large.pth'
        self.model_mae = RandomMasking.prepare_model(self.chkpt_dir, 'mae_vit_large_patch16') 

        if self.is_train == 0: # train
            self.img_path = glob.glob(f'./DATA/{class_name}/test/*/*')
        else:
            self.img_path = glob.glob(f'./DATA/{class_name}/test/*/*')
    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)

        # img1 = img

        img1 = Image.open(self.img_path[idx])
        img1 = img1.resize((224, 224))
        img1 = np.array(img1) / 255.

        assert img1.shape == (224, 224, 3)

        # normalize by ImageNet mean and std
        img1 = img1 - self.imagenet_mean
        img1 = img1 / self.imagenet_std

        _, img1 = RandomMasking.run_one_image(img1, self.model_mae)

        # img1을 NumPy ndarray로 변환 후, 이미지의 범위를 [0, 255]로 스케일링
        img1 = (img1.squeeze().numpy() * 255).astype(np.uint8)
        img1 = Image.fromarray(img1)

        # cutpaste coded needed here
        if self.is_train:
            img = self.transform(img)
            img1 = self.transform(img1)
            imgs = [img, img1]
            cutpates = [0,1]
            random.shuffle(cutpates)
            return imgs[cutpates[0]], imgs[cutpates[1]], cutpates[0], cutpates[1], self.img_path[idx]
        
        else:
            img = self.transform(img)
            img1 = self.transform(img1)
            imgs = [img, img1]
            cutpates = [0,1]
            random.shuffle(cutpates)
            return imgs[cutpates[0]], imgs[cutpates[1]], cutpates[0], cutpates[1], self.img_path[idx]