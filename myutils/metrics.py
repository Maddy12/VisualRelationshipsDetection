from __future__ import print_function, absolute_import
import numpy as np

def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@00k for the specified values of k.
    Adopted from: https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/main.py
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Metrics(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.loss_class=0
        self.loss_box_reg=0
        self.loss_objectness=0
        self.loss_rpn_box_reg=0
        self.detect_sum = 0
        self.detect_avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum/self.count

    def update_detection(self, loss, n):
        self.loss_class+=loss['loss_classifier'] 
        self.loss_box_reg+=loss['loss_box_reg']
        self.loss_objectness+=loss['loss_objectness']
        self.loss_rpn_box_reg+=loss['loss_rpn_box_reg']
        self.count+=n
        self.detect_sum += np.nansum(list(loss.values()))
        self.detect_avg = self.detect_sum/self.count