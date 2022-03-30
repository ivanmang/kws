import os
import sys
import shutil
import numpy as np
import time, datetime
import torch
import logging
import torch.nn as nn
import torch.utils
from PIL import Image


def gen_logger(log_path):
    if len(logging.getLogger().handlers) == 0 or not (os.path.exists(log_path)):
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(level=logging.INFO,
                            format=log_format, datefmt='%Y_%m_%d_%H_%M_%S')

        fh = logging.FileHandler(os.path.join(log_path))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)


# label smooth
class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        # %6.2f Print a float with 2 decimal places. Add spaces if this has less than 6 characters.
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):  # for print
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))  # return 's\tp\td' if ['s','p','d'] is entries

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'  # if num_batch = 1000, fmt = :4d
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
        # return '[{:4d}/1000]',so batch_fmtstr.format(batch) output format sss1/1000,s is space


def save_checkpoint(state, save, is_best, save_interval=10):
    if not os.path.exists(save):
        print('state folder not exist')

    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)

    if is_best:
        print("+++++++++++Best Acc!++++++++++!!!!")
        logging.info("+++++++++++Best Acc!++++++++++")
        best_filename = os.path.join(save, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)  # copy checkpoint to best
    if state['epoch'] % save_interval == 0:
        temp_filename = os.path.join(save, 'epoch_' + str(state['epoch']) + '_state.pth.tar')
        shutil.copyfile(filename, temp_filename)  # copy checkpoint to temp_filename


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def draw_training_curves(log_files):
    pass

