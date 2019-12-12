import os
import numpy as numpy
from PIL import Image
import csv, json
import pandas as pd
import logging
import torch
import torch.utils.data as data
from torchvision import transforms 
from torch.nn import functional as F
import pdb
import struct
import networkx as nx


ROOT = '/home/mschiappa/Desktop/VisualRelationshipsDetection/data/open-images/'
CLASS_PATH = os.path.join(ROOT,'class-descriptions.csv')

VAL_DATA_PATH = os.path.join(ROOT, 'validation/validation_image_data.json')
VAL_LABEL_PATH = os.path.join(ROOT, 'validation/validation_relationships.json')
VAL_IMAGES_PATH = os.path.join(ROOT, 'validation/images')

TRAIN_DATA_PATH = os.path.join(ROOT, 'train/train_image_data.json')
TRAIN_LABEL_PATH = os.path.join(ROOT, 'train/train_relationships.json')
TRAIN_IMAGES_PATH = os.path.join(ROOT, 'train/images')


class OpenImages2019(data.Dataset):
    def __init__(self, image_data_path, label_data_path, images_path, root=ROOT, class_description_path=CLASS_PATH, filenames=None, detection=False, validation=True, transform=None): 
        super(OpenImages2019, self).__init__()
        
        # Data Initialization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detection = detection
        self.data_which = 'validation' if validation else 'train'
        self.image_dir = images_path
        self.dir = os.path.join(root, self.data_which) 
        self.image_data = json.load(open(image_data_path))

        if filenames is None:
            self.filenames = os.listdir(images_path)
        else:
            self.filenames = filenames
        self.image_ids = [img.replace('.jpg', '') for img in self.filenames]
        self.rel_mapping = {'at':1, 'on': 2, 'holds': 3, 'plays': 4, 'interacts_with': 5, 'wears': 6, 'is': -1, 'inside_of': 7, 'under': 8, 'hits': 9}


        # Transformations for images
        if transform is None:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            self.transform=transforms.Compose([transforms.ToTensor(), normalize])
        else:
            self.transform = transform
        shape = (256, 170)
        self.bounding_transform = transforms.Compose([transforms.Resize(shape)])
        
        # Add ground truths for either detection or relationship learning
        self.classes = pd.read_csv(class_description_path, header=None)
        self.classes['int_label'] = self.classes.index.values  # create int class label for objects
        self.classes = self.classes.set_index(0)
        classes_path = os.path.join(root, 'classes_with_int_labels.csv')
        if not os.path.exists(classes_path):
            self.classes.to_csv(classes_path)
        self.classes = self.classes.to_dict(orient='index')
        
        if self.detection:
            self.labels = self.preprocess_labels(os.path.join(self.dir, self.data_which+'-annotations-bbox.json'))
        else:
            self.labels = json.load(open(label_data_path))

        # Get remaining image IDs 
        self.image_ids = [im for im in self.image_ids if im in self.labels.keys()]
        
        # Add length info
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
        If detection: 
            boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, 
                        with values between 0 and H and 0 and W

            labels (Int64Tensor[N]): the class label for each ground-truth box
        Else:
            Objects (IntT64Tensor[N, 3, W, H]): A group of objects present in image. 
                Every 2 objects have a relationshop between them in order Object and Subject. 

            labels (Int64[N/2]): Labels/predicates for each pair of objects in an image
        """
        # Read in Image and transform to a normalized tensor
        image_id = self.image_ids[index]
        image_path = image_id+'.jpg'
        image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
        image = self.transform(image)
        
        # Get height and width of image to extract bounding box, assuming all not the same
        height = self.image_data[image_id]['height']
        width = self.image_data[image_id]['width']

        # Return either ground truth bounding boxes for detection or relationships
        labels = dict()
        if self.detection:
            bbox_labels = self.labels[image_id]
            labels['boxes'] = [float(bbox_labels['XMax']), float(bbox_labels['YMax']), float(bbox_labels['XMin']), float(bbox_labels['YMin'])]
            labels['labels'] = [self.classes[bbox_labels['LabelName']]['int_label']]           
            return image, [labels]
        else:           
            # Initialize graph for adjacency matrix 
            # gt = nx.from_edgelist(self.labels[image_id]['edge_list'])
            gt = nx.Graph()
            # Iterate through the relationships and extract the ROI
            objects = list()
            classes = list()
            nodes = list()
            images = dict()
            rel_dict = self.labels[image_id]['relationships']
            
            for rel in  rel_dict:
                # Get object and subject image regions and make them same dimensions
                images[rel['object']['node']] = F.interpolate(self.extract_ROI(rel['object'], image, width, height).unsqueeze_(0), size=(256, 170))  #.to(self.device)
                images[rel['subject']['node']] = F.interpolate(self.extract_ROI(rel['subject'], image, width, height).unsqueeze_(0), size=(256, 170))  # .to(self.device)
                
                # Add classes
                classes.append(self.classes[rel['object']['name']]['int_label'])
                classes.append(self.classes[rel['subject']['name']]['int_label'])
                
                # Add edge weights representing the predicate
                gt.add_edge(rel['object']['node'], rel['subject']['node'], weight=self.rel_mapping[rel['predicate']])
                # gt[rel['object']['node']][rel['subject']['node']]['weight'] =  self.rel_mapping[rel['predicate']]

            # Get adjacency matrix 
            gt = torch.tensor(nx.adjacency_matrix(gt).todense())

            # Store remaining
            labels['gt'] = gt  # length is number of  relationships
            labels['labels'] = classes # length is number of relationships*2
            
            return images, labels
    
    def extract_ROI(self, bbox, image, im_width, im_height):
        (left, right, bottom, top) = (float(bbox['x']) * im_width, (float(bbox['x'])+bbox['w']) * im_width,
                              float(bbox['y']) * im_height, (float(bbox['y'])+bbox['h']) * im_height)
        obj = image[: , int(bottom):int(top), int(left):int(right)]
        return obj


def test_functionality():
    from torch.utils.data import DataLoader
    from torchvision import models
    from torch import nn
    test = OpenImages2019(validation=True, image_data_path=VAL_DATA_PATH, label_data_path=VAL_LABEL_PATH, 
                            images_path=VAL_IMAGES_PATH, detection=False)
    test_loader = DataLoader(test, shuffle=False, batch_size=1)# , collate_fn=test_collate2)
    n = 10
    for inputs, targets in test:
        backbone = models.resnet18(pretrained=True)
        modules=list(backbone.children())[:-1]
        backbone =nn.Sequential(*modules)
        for p in backbone.parameters():
            p.requires_grad = False
        pdb.set_trace()
        break
        
if __name__=='__main__':
    test_functionality()
