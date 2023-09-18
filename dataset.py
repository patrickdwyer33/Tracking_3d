from utils import utils
import cv2
import numpy as np
import torch
import os
from torch.utils.data import Dataset
import json

def get_batch(data_loader):
    return next(data_loader.__iter__())

def get_rand_pics(data_loader):
    rand_idx = torch.randint(0,4,(1,))[0].item()
    pics = get_batch(data_loader)[rand_idx,:,:,:]
    assert(pics.size()[0] == 8)
    return pics

class TrainingData(Dataset):
    def __init__(self, transform=None, 
                 target_transform=None, 
                 num_body_parts=8, 
                 image_shape_x=1280, 
                 image_shape_y=1024, 
                 data_dir=os.path.join(os.getcwd(), "Training_Data"), 
                 num_images_per_trial=4):
        trial_dirs = os.listdir(data_dir)
        utils.process_dir_list(trial_dirs)
        trial_paths = list(map(lambda x: os.path.join(data_dir, x), trial_dirs))
        self.data = trial_paths
        self.num_images = num_images_per_trial
        self.image_shape_x = image_shape_x
        self.image_shape_y = image_shape_y
        self.num_body_parts = num_body_parts
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        trial_path = self.data[idx]
        trial_data = os.listdir(trial_path)
        utils.process_dir_list(trial_data)
        images = []
        for i in range(0, self.num_images):
            image_path = os.path.join(trial_path, trial_data[i])
            img = cv2.imread(image_path)
            img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_bw = img_bw.astype("float64", copy=False)
            img_bw /= np.max(img_bw)
            images.append(img_bw)
        out_images = np.concatenate(images, axis=0)
        out = torch.tensor(out_images,dtype=torch.double).reshape(self.num_images,self.image_shape_y,self.image_shape_x)
        if self.transform is not None:
            out = self.transform(out)
        if len(trial_data) != 5:
            print("WARNING! Labels for training not found.")
            labels = torch.rand(self.num_images*2,self.num_body_parts,dtype=torch.double)
            return out, labels
        labels_path = os.path.join(trial_path, "labels.csv")
        labels_numpy = np.genfromtxt(labels_path, delimiter=',')
        labels = torch.tensor(labels_numpy, dtype=torch.double).reshape(self.num_images*2,self.num_body_parts)
        if self.target_transform is not None:
            labels = self.target_transform(labels)
        return out, labels
    
class TestData(Dataset):
    def __init__(self, transform=None, 
                 target_transform=None, 
                 num_body_parts=8, 
                 image_shape_x=1280, 
                 image_shape_y=1024, 
                 data_dir=os.path.join(os.getcwd(), "Training_Data"), 
                 num_images_per_trial=4):
        trial_dirs = os.listdir(data_dir)
        utils.process_dir_list(trial_dirs)
        trial_paths = list(map(lambda x: os.path.join(data_dir, x), trial_dirs))
        self.data = trial_paths
        self.num_images = num_images_per_trial
        self.image_shape_x = image_shape_x
        self.image_shape_y = image_shape_y
        self.num_body_parts = num_body_parts
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        trial_path = self.data[idx]
        trial_data = os.listdir(trial_path)
        utils.process_dir_list(trial_data)
        images = []
        for i in range(0, self.num_images):
            image_path = os.path.join(trial_path, trial_data[i])
            img = cv2.imread(image_path)
            img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_bw = img_bw.astype("float64", copy=False)
            img_bw /= np.max(img_bw)
            images.append(img_bw)
        out_images = np.concatenate(images, axis=0)
        out = torch.tensor(out_images,dtype=torch.double).reshape(self.num_images,self.image_shape_y,self.image_shape_x)
        if self.transform is not None:
            out = self.transform(out)
        return out