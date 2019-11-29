import os
import numpy as numpy
from PIL import Image
import csv, json
import pandas as pd
import logging
import torch
import torch.utils.data as data
from torchvision import transforms 
import pdb

class OpenImages2019(data.Dataset):
    def __init__(self, filenames=None, detection=False, validation=True, root='../data/open-images', transform=None): 
        super(OpenImages2019, self).__init__()
        
        # Data Initialization
        self.height = 512
        self.width = 512
        self.detection = detection
        self.data_which = 'validation' if validation else 'train'
        self.dir = os.path.join(root, self.data_which) 
        if filenames is None:
            self.filenames = os.listdir(os.path.join(self.dir, 'images'))
        else:
            self.filenames = filenames
        if transform is None:
            shape = (512, 512)
            self.transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(shape)])
        else:
            self.transform = transform
        self.image_ids = [img.replace('.jpg', '') for img in self.filenames]

        # Labeling
        self.classes = pd.read_csv('/home/mschiappa/Desktop/VisualRelationshipsDetection/data/open-images/class-descriptions-boxable-1.csv', header=None)
        self.classes['int_label'] = self.classes.index.values
        self.classes = self.classes.set_index(0)
        self.classes = self.classes.to_dict(orient='index')
        
        if self.detection:
            self.bbox_labels = self.preprocess_labels(os.path.join(self.dir, self.data_which+'-annotations-bbox.json'))
            self.image_ids = [im for im in self.image_ids if im in self.bbox_labels.keys()]
        else:
            self.rel_labels = self.preprocess_labels(os.path.join(self.dir, self.data_which+'-annotations-vrd.json'))
            self.image_ids = [im for im in self.image_ids if im in self.rel_labels.keys()]
        self.length = len(self.image_ids)

    def __len__(self):
        return self.length

    def preprocess_labels(self, filepath):
        if os.path.isfile(filepath):
            return json.load(open(filepath))
        else:
            print("Pre-processing labels because json file did not exist.")
            df = pd.read_csv(filepath.replace('.json', '.csv')).set_index('ImageID')
            json_obj = df.to_dict(orient='index')
            with open(filepath, 'w') as f:
                json.dump(json_obj, f)
        return json_obj

    def __getitem__(self, index):
        """
        boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, 
                    with values between 0 and H and 0 and W

        labels (Int64Tensor[N]): the class label for each ground-truth box
        """
        image_id = self.image_ids[index]
        image_path = image_id+'.jpg'
        image = Image.open(os.path.join(self.dir, 'images', image_path)).convert('RGB')
        image = self.transform(image)
        transform_dict = transforms.Compose([transforms.ToTensor()])
        labels = dict()
        if self.detection:
            bbox_labels = self.bbox_labels[image_id]
            labels['boxes'] = [float(bbox_labels['XMax']), float(bbox_labels['YMax']), float(bbox_labels['XMin']), float(bbox_labels['YMin'])]
            labels['labels'] = [self.classes[bbox_labels['LabelName']]['int_label']]           
            return image, [labels]
        else:
            rel_labels = self.rel_labels[image_id]
            boxes = torch.tensor([rel_labels['XMax1'], rel_labels['YMax1'], rel_labels['XMin1'], rel_labels['YMin1']])
            boxes = torch.stack([boxes, torch.tensor([rel_labels['XMax2'], rel_labels['YMax2'], rel_labels['XMin2'], rel_labels['YMin2']])])
            try:
                labels = torch.tensor([[self.classes[rel_labels['LabelName1']]['int_label']], [self.classes[rel_labels['LabelName2']]['int_label']]])
            except Exception as e:
                print(e) 
                print(rel_labels)
            