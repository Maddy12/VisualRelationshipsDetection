import multiprocessing
import time
import os
import shutil
import pdb
from  progress.bar import Bar
from datetime import datetime

# Pytorch
import torch
from torch.nn import CrossEntropyLoss
from torchvision import models

# Local
from myutils.metrics import *



def run_experiment(model, exp_name, dataloader, train, criterion=CrossEntropyLoss(), optimizer=None, epochs=None, regularizer=None): 
    """
    # TODO allow for extra things to be stored at checkpoints
    When regularizer is set to True, it assumes that defined in the model is a module defined as "regularizer" that takes the following input:
        * inputs
        * outputs
        * targets
    The output of this module will then be added to the output of the criterion function, aka global objective function. 
    The loss reported is without the regularizer loss being added regardless if a regularizer is used or not. 
    For localized regularization or loss, you can use this method as long as it still fulfilles the requirements of input and module name. 

    Args: 
        model: The model being run. Must be PyTorch model.
        exp_name (str): The name of the experiment.
        epochs (int): the number of epochs to run for training, default is None
        criterion: the criterion function to use to calculate loss
        dataloader (dict): a dictionary that has a validation data generator and optionally a testing data generator. Keys are 'training' and 'testing'
        train (bool): if the model is undergoes training
        optimizer: the optimization function used for backpropagation 
        regularizer (bool): Whether or not the model uses a regularizer, aka a special loss function defined in the model as "loss = model.regularizer()"
    """
    # assertions 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, epochs)
    best_acc = 0.0

    # Training, and if train is false, loads model through passed model path
    print("Running experiment {} with {} epochs".format(exp_name, epochs))
    if train:
        start = time.time()
        for epoch in range(epochs):
            # epoch += 1
            scheduler.step()
            avg_train_loss, avg_train_top1_prec = run_model(epoch, model, criterion, optimizer, dataloader['training'], dataloader['training_length'],  
                                                            train=True, device=device, regularizer=regularizer)
            
            # save model
            current_date = str(datetime.now().year)+str(datetime.now().month)+str(datetime.now().day)
            checkpoint_path = os.path.join(os.getcwd(), '{}_{}_checkpoints'.format(exp_name, current_date))
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            is_best = avg_train_top1_prec > best_acc
            best_acc = max(avg_train_top1_prec, best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': avg_train_top1_prec, 
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=checkpoint_path)
        end = time.time()
        print("Completed epochs {} | Train Time {} | Acc {} | Loss {} ".format(epochs, round(end-start, 2), avg_train_top1_prec,  avg_train_loss))

    # Validation
    start = time.time()
    avg_test_loss, avg_test_top1_prec = run_model(epoch, model, criterion, optimizer, dataloader['testing'], dataloader['testing_length'], 
                                                                train=False, device=device)
    end = time.time() - start
    print(" Test Time  {} | Acc {} | Loss {}".format(round(end-start, 2), avg_test_top1_prec, avg_test_loss, ))


def run_model(epoch, model, criterion, optimizer, dataloader,  datalength, device, train=True, regularizer=None):
    """
    This function will run the model in either train or test mode returning the overall average loss and the top1 loss.

    Args:
        epoch (int): The epoch that the run is currently on. 
        model: The model being used. 
        criterion: The function that is used to evaluate loss.
        optimizer: The function that is used for backpropagation.
        dataloader: A generator object to iterate through the train/test dataset. 
        datalength (int): The number of batches in the data generator. 
        device (str): The device to run on, either 'cpu' or 'cuda'.
        train (bool): Whether the run is to train or not. Default is True.
        regularizer: A regularizer function if using custom regularizer to add to the global loss function. 
        num_classes (int): The number of classes that are being used for the multi-class problem.

    Returns: 
        Average loss, and top1k loss.
    """
    # Determine if we are in training mode or in validation mode
    if train:
        model.train()
        bar = Bar('Training', max=datalength)
    else:
        model.eval()
        bar = Bar('Validating', max=datalength)

    # Initialize model performance tracking metrics
    losses = Metrics()
    top1 = Metrics()
    top5 = Metrics()

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Send inputs and outputs to device which is either 'cpu' or 'cuda'
        inputs = inputs.to(device)
        inputs = torch.autograd.Variable(inputs)
        #### A FIX BECAUSE COLLATE ISSUES SO ANNOYING I KNOW BUT WHATEV ###
        targets = targets[0]
 
        # Forward Pass,                   ]
        outputs = model(inputs, targets)

        # measure accuracy and record loss
        loss = criterion(outputs, targets)
        if regularizer:
            reg = regularizer(inputs, outputs, targets)
            loss = loss + reg

        if train:
            losses.update_detection(outputs, inputs.size(0))
        else:
            losses.update(loss.data.item(), inputs.size(0))
            prec1, prec5 = accuracy(outputs['loss_classifier'], targets.data, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
        
        # Backward Pass: compute gradient and do SGD step
        if train:   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if not train:
            bar.suffix = '({batch}/{size}) | Total: {total:} | Epoch: {epoch:} | OrigLoss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(dataloader),
                total=bar.elapsed_td,
                epoch=epoch+1,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
        else:
            bar.suffix = '({batch}/{size}) | Total: {total:} | Epoch: {epoch:} | OrigLoss: {loss:.4f}'.format(
                batch=batch_idx + 1,
                size=len(dataloader),
                total=bar.elapsed_td,
                epoch=epoch+1,
                loss=losses.detect_avg,
            )
        bar.next() 
    bar.finish()
    return losses.avg, top1.avg


def save_checkpoint(state, is_best, checkpoint):
    """
    This function will save a checkpoint based off of the passed checkpoint save path.
    If the model is the best so far, it will also save the full state dict of the model.
    The state wil have the keys: 
        * epoch: the epoch the model training is currently on
        * state_dict: the parameters of the model at the current epoch
        * acc: the average testing top1 precision
        * best_acc: the best accuracy the model has produced so far, may be from a different epoch
        * optimizer: the optimizer state dict

    Args: 
        state (dict): THe current state of the model.
        is_best (bool): If the model is the best so far with the current parameters.
        checkpoint (str): The directory to save the current model state to 
    """
    try:
        filename = 'checkpoint.pth.tar'
        filepath = os.path.join(checkpoint, str(state['epoch'])+'_'+str(datetime.now()).replace(' ', '')+'_'+filename)
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
    except Exception as e:
        print("There is an error saving checkpoints: {}".format(e))
        exit()