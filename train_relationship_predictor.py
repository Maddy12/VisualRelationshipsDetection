from __future__ import print_function
import sys
import time
import random
import math
import os
import logging
import multiprocessing 
import pdb
import numpy as np

import torch
from torch import nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch._six import container_abcs, string_classes, int_classes
from torch.nn import DataParallel

from myutils.datasets import OpenImages2019
from myutils.model import GraphLabelPredict
from run_exp import run_experiment
from myutils.focal_loss import FocalLoss

ROOT = '/home/mschiappa/Desktop/VisualRelationshipsDetection/data/open-images/'
CLASS_PATH = os.path.join(ROOT,'class-descriptions.csv')

VAL_DATA_PATH = os.path.join(ROOT, 'validation/validation_image_data.json')
VAL_LABEL_PATH = os.path.join(ROOT, 'validation/validation_relationships.json')
VAL_IMAGES_PATH = os.path.join(ROOT, 'validation/images')

TRAIN_DATA_PATH = os.path.join(ROOT, 'train/train_image_data.json')
TRAIN_LABEL_PATH = os.path.join(ROOT, 'train/train_relationships.json')
TRAIN_IMAGES_PATH = os.path.join(ROOT, 'train/images')


def train_openim_reldetect(learning_rate=0.001, momentum=0, shape=(512, 512), decay=0.0005):
    """
    """   
    # Set up model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': multiprocessing.cpu_count(), 'pin_memory': True}
    model = GraphLabelPredict(fine_tune=True).to(device)
    model = DataParallel(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=0, weight_decay=decay)
    criterion = nn.CrossEntropyLoss()

    # Set up datasets
    transform=transforms.Compose([transforms.Resize(shape), transforms.ToTensor()] )
    train = OpenImages2019(images_path=TRAIN_IMAGES_PATH, image_data_path=TRAIN_DATA_PATH, label_data_path=TRAIN_LABEL_PATH,
                            validation=False, root=ROOT, transform=None, detection=False)
    test = OpenImages2019(images_path=VAL_IMAGES_PATH, image_data_path=VAL_DATA_PATH, label_data_path=VAL_LABEL_PATH,
                            root=ROOT, validation=True, transform=None, detection=False)
    data_loaders = {'training': train, 'training_length': train.length, 'testing': test, 'testing_length': test.length}

    # Train
    run_experiment(model, 'relationships_OpenImages', data_loaders, train=True, criterion=criterion, optimizer=optimizer, epochs=1, regularizer=None)


if __name__=='__main__':
    train_openim_reldetect()