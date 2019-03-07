import os
import numpy as np
import json
import torch
from PIL import Image
import torchvision
from torch.utils.data import Dataset as Dataset

FACE_CLFN_TRAIN_DATA = './../Data/train_data/medium'
FACE_CLFN_VAL_DATA = './../Data/validation_classification/medium'
FACE_CLFN_TEST_DATA = './../Data/test_classification/medium'

FACE_VRFN_TRAIN_DATA = './../Data/train_data/'
FACE_VRFN_VAL_DATA = './../Data/validation_verification'
FACE_VRFN_VAL_DATA_LIST = './../Data/validation_trials_verification.txt'
FACE_VRFN_TEST_DATA = './../Data/test_verification_new'
FACE_VRFN_TEST_DATA_LIST = './../Data/test_trials_verification_student_new.txt'

DUMPED_DATA_ROOT = './../Data/dumped'

class FaceClassificationDataset(Dataset):
    def __init__(self, num_classes, mode='train'):
        # Check for valid mode.
        self.mode = mode
        self.num_classes = num_classes
        valid_modes = {'train', 'val', 'test'}
        if self.mode not in valid_modes:
            raise ValueError("FaceClassificationDataset Error: Mode must be one of %r." % valid_modes)
        # Path where data and labels tensor will be dumped/loaded to/from.
        self.dataDumpPath = os.path.join(DUMPED_DATA_ROOT, '{}.json'.format(self.mode))
        self.labelsDumpPath = os.path.join(DUMPED_DATA_ROOT, '{}_labels.json'.format(self.mode))
        if self.mode == 'train':
            self.data_dir = FACE_CLFN_TRAIN_DATA
        elif self.mode =='val':
            self.data_dir = FACE_CLFN_VAL_DATA
        else:
            self.data_dir = FACE_CLFN_TEST_DATA
        # Check if we already have previously dumped tensor data.
        if os.path.isfile(self.dataDumpPath):
            with open(self.dataDumpPath, 'r') as infile:
                self.data = json.load(infile)
            # If data json exists then labels MUST exist for train and val modes.
            if self.mode != 'test':
                if os.path.isfile(self.labelsDumpPath):
                    with open(self.labelsDumpPath, 'r') as infile:
                        self.labels = json.load(infile)
                else:
                    raise ValueError("FaceClassificationDataset Error: Data JSON file found at %s but labels JSON file missing at %s." \
                        % (self.dataDumpPath, self.labelsDumpPath))
        else:
            # Load the data and labels (labels = empty for 'test' mode)
            self.data = []
            self.labels = []
            if self.mode != 'test':
                # In case of train and val, we have class folders and multiple images for each class.
                classes = sorted([int(dir) for dir in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, dir))])
                classes = [str(class_id) for class_id in classes]
                for dir in classes:
                    #print('Loop : %d' % (int(dir)))
                    class_path = os.path.join(self.data_dir, dir)
                    image_files = sorted([os.path.join(dir,file) for file in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, file))])
                    self.data.extend(image_files)
                    class_id = [int(dir)] * len(image_files)
                    self.labels.extend(class_id)
            else:
                # In case of test, there is no class folder, directly image files.
                self.data = sorted([int(file.split('.jpg')[0]) for file in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, file))])
                self.data = [(str(file) + ".jpg") for file in self.data]
            # Dump data and labels tensor so we don't have to create again.
            with open(self.dataDumpPath, 'w') as outfile:
                json.dump(self.data, outfile)
            if self.mode != 'test':
                with open(self.labelsDumpPath, 'w') as outfile:
                    json.dump(self.labels, outfile)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.data[idx])
        img = Image.open(img_path)
        img = torchvision.transforms.ToTensor()(img)
        if self.mode == 'test':
            return img
        else:
            label = self.labels[idx]
            return img, label

class FaceVerificationDataset(Dataset):
    def __init__(self, num_classes, mode='train'):
        # Check for valid mode.
        self.mode = mode
        self.num_classes = num_classes
        valid_modes = {'train', 'val', 'test'}
        if self.mode not in valid_modes:
            raise ValueError("FaceVerificationDataset Error: Mode must be one of %r." % valid_modes)
        # Path where data and labels tensor will be dumped/loaded to/from.
        self.dataDumpPath = os.path.join(DUMPED_DATA_ROOT, '{}_vrfn.json'.format(self.mode))
        self.labelsDumpPath = os.path.join(DUMPED_DATA_ROOT, '{}_vrfn_labels.json'.format(self.mode))
        if self.mode == 'train':
            self.data_dir = FACE_VRFN_TRAIN_DATA
        elif self.mode =='val':
            self.data_dir = FACE_VRFN_VAL_DATA
        else:
            self.data_dir = FACE_VRFN_TEST_DATA
        # Check if we already have previously dumped tensor data.
        if os.path.isfile(self.dataDumpPath):
            with open(self.dataDumpPath, 'r') as infile:
                self.data = json.load(infile)
            # If data json exists then labels MUST exist for train and val modes.
            if self.mode != 'test':
                if os.path.isfile(self.labelsDumpPath):
                    with open(self.labelsDumpPath, 'r') as infile:
                        self.labels = json.load(infile)
                else:
                    raise ValueError("FaceVerificationDataset Error: Data JSON file found at %s but labels JSON file missing at %s." \
                        % (self.dataDumpPath, self.labelsDumpPath))
        else:
            # Load the data and labels (labels = empty for 'test' mode)
            self.data = []
            self.labels = []
            if self.mode == 'val':
                with open(FACE_VRFN_VAL_DATA_LIST) as val_file:
                    val_lines = val_file.readlines()
                val_data = [line.strip('\n').split(' ') for line in val_lines]
                self.labels = [int(data[2]) for data in val_data]
                self.data = [[data[0], data[1]] for data in val_data]
            elif self.mode == 'test':
                with open(FACE_VRFN_TEST_DATA_LIST) as test_file:
                    test_lines = test_file.readlines()
                self.data = [line.strip('\n').split(' ') for line in test_lines]
            else:   #train
                # In case of train, we have class folders and multiple images for each class.
                train_data_folders = ['medium', 'large']
                for train_folder in train_data_folders:
                    data_dir = os.path.join(self.data_dir, train_folder)
                    classes = sorted([int(dir) for dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, dir))])
                    classes = [str(class_id) for class_id in classes]
                    for dir in classes:
                        class_path = os.path.join(data_dir, dir)
                        image_files = sorted([os.path.join(train_folder,dir,file) for file in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, file))])
                        self.data.extend(image_files)
                        class_id = [int(dir)] * len(image_files)
                        self.labels.extend(class_id)
            # Dump data and labels tensor so we don't have to create again.
            with open(self.dataDumpPath, 'w') as outfile:
                json.dump(self.data, outfile)
            if self.mode != 'test':
                with open(self.labelsDumpPath, 'w') as outfile:
                    json.dump(self.labels, outfile)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode != 'train':
            img1_path = os.path.join(self.data_dir, self.data[idx][0])
            img2_path = os.path.join(self.data_dir, self.data[idx][1])
            img1 = torchvision.transforms.ToTensor()(Image.open(img1_path))
            img2 = torchvision.transforms.ToTensor()(Image.open(img2_path))
        else: # train
            img_path = os.path.join(self.data_dir, self.data[idx])
            img = torchvision.transforms.ToTensor()(Image.open(img_path))
        if self.mode == 'test':
            return img1, img2
        elif self.mode == 'val':
            return img1, img2, self.labels[idx]
        else:
            return img, self.labels[idx]
