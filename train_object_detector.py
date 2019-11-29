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
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch._six import container_abcs, string_classes, int_classes
from torch.nn import DataParallel

from myutils.datasets import OpenImages2019
from run_exp import run_experiment
from myutils.focal_loss import FocalLoss

def train_openim_objdetect(batch_size=2, learning_rate=0.001, momentum=0, shape=(512, 512), decay=0.0005,
                root='/home/mschiappa/Desktop/VisualRelationshipsDetection/data/open-images',
                weightfile='/home/mschiappa/Desktop/VisualRelationshipsDetection/pytorchyolo3/yolov3.weights'):
    """
    """   
    # Set up model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': multiprocessing.cpu_count(), 'pin_memory': True}
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=600, pretrained_backbone=True)
    model = model.to(device)
    model = DataParallel(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(model.parameters(), lr=learning_rate/batch_size, momentum=momentum, dampening=0, weight_decay=decay*batch_size)
    focal_loss = FocalLoss(num_classes=600)

    # Set up datasets
    shape = (800, 800)
    transform=transforms.Compose([transforms.Resize(shape), transforms.ToTensor()] )
    
    train = OpenImages2019(validation=False, root=root, transform=transform, detection=True)
    test = OpenImages2019(validation=True, root=root, transform=transform, detection=True)
    train_loader = DataLoader(train, shuffle=False, batch_size=batch_size, collate_fn=default_collate, **kwargs)
    test_loader = DataLoader(test, shuffle=False, batch_size=batch_size, collate_fn=default_collate, **kwargs)
    data_loaders = {'training': train_loader, 'training_length': train.length, 'testing': test_loader, 'testing_length': test.length}

    # Train
    run_experiment(model, 'detection_OpenImages', data_loaders, train=True, criterion=criterion, optimizer=optimizer, epochs=1, regularizer=None)

def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch).cuda()
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        # print({key: default_collate([d[key] for d in batch]) for key in elem})
        # return [b for b in batch]
        # return [batch]
        return [{'boxes': torch.tensor([d['boxes']]), 'labels': torch.tensor(d['labels'])} for d in batch]
        # return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

def criterion(outputs, targets): 
    """
    Sample output: {'loss_classifier': torch.tensor(6.3690), 
                    'loss_box_reg': torch.tensor(0.), 
                    'loss_objectness': torch.tensor(0.6901), 
                    'loss_rpn_box_reg': torch.tensor(0.)}
    """
    return np.sum(list(outputs.values()))

if __name__=='__main__': 
    train_openim_objdetect(weightfile='/home/mschiappa/Desktop/VisualRelationshipsDetection/models/yolov3.weights')