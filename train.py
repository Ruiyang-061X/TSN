import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import time
from dataset import TSNDataset
from torch.utils.data import DataLoader
from tsn import TSN
from transform import *
from torch.backends import cudnn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch import nn
import torch


parser = argparse.ArgumentParser(description='pytorch implementation of temporal segment network')
parser.add_argument('--dataset', default='ucf101')
parser.add_argument('--modality', default='RGB', choices=['RGB', 'RGBDiff', 'Flow'])
parser.add_argument('--trainset')
parser.add_argument('--validationset')
parser.add_argument('--base_model', default='BNInception')
parser.add_argument('--n_segment', default=3, type=int)
parser.add_argument('--consensus_type', default='avg', choices=['avg', 'identity'])
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--epoch', default=45, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--lr_step', default=[20, 40], nargs='+', type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--clip_gradient', default=None, type=float)
parser.add_argument('--print_every', default=20, type=int)
parser.add_argument('--validation_every', default=5, type=int)
args = parser.parse_args()

# set seed for reproducibility
def set_random_seed(random_seed=None):
    """
    Using random seed for numpy and torch
    """
    if(random_seed is None):
        random_seed = 8
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return

set_random_seed(74)

cudnn.benchmark = True

if not os.path.exists('trained_model'):
    os.mkdir('trained_model')

if args.modality == 'RGB':
    new_length = 1
else:
    new_length = 5

if 'vgg' in args.base_model or 'resnet' in args.base_model:
    input_size = 224
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]
    if args.modality == 'Flow':
        input_mean = [0.5]
        input_std = [np.mean(input_std)]
    if args.modality == 'RGBDiff':
        input_mean = input_mean + [0] * 3 * new_length
        input_std = input_std + [np.mean(input_std) * 2] * 3 * new_length
if args.base_model == 'BNInception':
    input_size = 224
    input_mean = [104, 117, 128]
    input_std = [1]
    if args.modality == 'Flow':
        input_mean = [128]
    if args.modality == 'RGBDiff':
        input_mean = input_mean * (new_length + 1)
if 'inception' in args.base_model:
    input_size = 299
    input_mean = [0.5]
    input_std = [0.5]
crop_size = input_size
scale_size = input_size * 256 // 224
if args.modality == 'RGB':
    train_augmentation = transforms.Compose([GroupMultiScaleCrop(input_size, [1, 0.875, 0.75, 0.66]), GroupRandomHorizontalFlip(is_flow=False)])
if args.modality == 'RGBDiff':
    train_augmentation = transforms.Compose([GroupMultiScaleCrop(input_size, [1, 0.875, 0.75]), GroupRandomHorizontalFlip(is_flow=False)])
if args.modality == 'Flow':
    train_augmentation = transforms.Compose([GroupMultiScaleCrop(input_size, [1, 0.875, 0.75]), GroupRandomHorizontalFlip(is_flow=True)])

if args.modality != 'RGBDiff':
    normalize = GroupNormalize(input_mean, input_std)
else:
    normalize = IdentityTransform()
transform_trainset = transforms.Compose([train_augmentation, Stack(roll=args.base_model == 'BNInception'), ToTorchFormatTensor(div= args.base_model != 'BNInception'), normalize, ])
trainset = TSNDataset(video_list_path=args.trainset, modality=args.modality, train=True, n_segment=args.n_segment, new_length=new_length, random_shift=True, tranform=transform_trainset)
transform_validationset = transforms.Compose([GroupScale(scale_size), GroupCenterCrop(crop_size), Stack(roll=args.base_model == 'BNInception'), ToTorchFormatTensor(div= args.base_model != 'BNInception'), normalize, ])
validationset = TSNDataset(video_list_path=args.validationset, modality=args.modality, train=False, n_segment=args.n_segment, new_length=new_length, random_shift=False, tranform=transform_trainset)
trainset_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
validation_loader = DataLoader(validationset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

if args.dataset == 'ucf101':
    n_class = 101
tsn = TSN(base_model=args.base_model, n_class=n_class, consensus_type=args.consensus_type, before_softmax=True, dropout=args.dropout, n_crop=1, modality=args.modality, n_segment=args.n_segment, new_length=new_length)
tsn = tsn.cuda()

loss_function = CrossEntropyLoss()
loss_function = loss_function.cuda()

first_conv_weight = []
first_conv_bias = []
normal_weight = []
normal_bias = []
bn = []
conv_count = 0
bn_count = 0
for m in tsn.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        ps = list(m.parameters())
        conv_count += 1
        if conv_count == 1:
            first_conv_weight += [ps[0]]
            if len(ps) == 2:
                first_conv_bias += [ps[1]]
        else:
            normal_weight += [ps[0]]
            if len(ps) == 2:
                normal_bias += [ps[1]]
    if isinstance(m, nn.Linear):
        ps = list(m.parameters())
        normal_weight += [ps[0]]
        if len(ps) == 2:
            normal_bias += [ps[1]]
    if isinstance(m, nn.BatchNorm1d):
        ps = list(m.parameters())
        bn += ps
    if isinstance(m, nn.BatchNorm2d):
        bn_count += 1
        if bn_count == 1:
            ps = list(m.parameters())
            bn += ps
optimize_policy = [
    {'name': 'first_conv_weight', 'params': first_conv_weight, 'lr_mult': 5 if args.modality == 'Flow' else 1, 'decay_mult': 1},
    {'name': 'first_conv_bias', 'params': first_conv_bias, 'lr_mult': 10 if args.modality == 'Flow' else 2, 'decay_mult': 0},
    {'name': 'normal_weight', 'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1},
    {'name': 'normal_bias', 'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0},
    {'name': 'BN scale/shift', 'params': bn, 'lr_mult': 1, 'decay_mult': 0},
]
for i in optimize_policy:
    print('group {} has {} parameters, lr_mult: {}, decay_mult: {}'.format(i['name'], len(i['params']), i['lr_mult'], i['decay_mult']))
optimizer = SGD(optimize_policy, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

class AverageMeter():

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(predicted_label, label, topk=(1, 5)):
    maxk = max(topk)
    _, predicted_label = predicted_label.topk(maxk, 1, True, True)
    predicted_label = predicted_label.t()
    correct = predicted_label.eq(label.reshape(1, -1).expand_as(predicted_label))

    result = []
    for i in topk:
        correct_i = correct[ : i].reshape(-1).float().sum(0)
        result += [correct_i / args.batch_size * 100.0]

    return result

def adjust_lr(optimizer, epoch, lr_step):
    decay = 0.1 ** (sum(epoch >= np.array(lr_step)))
    lr = args.lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = args.weight_decay * param_group['decay_mult']

def validation():
    print('start validation...')
    loss_avg = AverageMeter()
    accuracy1_avg = AverageMeter()
    accuracy5_avg = AverageMeter()

    tsn.eval()
    for i, (input, label) in enumerate(validation_loader):
        input = input.cuda()
        label = label.cuda()
        predicted_label = tsn(input)
        loss = loss_function(predicted_label, label)
        accuracy1, accuracy5 = accuracy(predicted_label, label, topk=(1, 5))
        loss_avg.update(loss.item(), input.size(0))
        accuracy1_avg.update(accuracy1.item(), input.size(0))
        accuracy5_avg.update(accuracy5.item(), input.size(0))
    print('Validation result: Loss {:.3f} Accuracy@1 {:.3f} Accuracy@5 {:.3f}'.format(loss_avg.avg, accuracy1_avg.avg, accuracy5_avg.avg))
    tsn.train()
    accuracy_ = accuracy1_avg.avg

    return accuracy_

print('start training...')
for i in range(args.epoch):
    adjust_lr(optimizer, i, args.lr_step)

    data_time = AverageMeter()
    batch_time = AverageMeter()
    loss_avg = AverageMeter()
    accuracy1_avg = AverageMeter()
    accuracy5_avg = AverageMeter()

    end = time.time()
    for j, (input, label) in enumerate(trainset_loader):
        data_time.update(time.time() - end)

        tsn.zero_grad()
        input = input.cuda()
        label = label.cuda()
        predicted_label = tsn(input)
        loss = loss_function(predicted_label, label)
        accuracy1, accuracy5 = accuracy(predicted_label, label, topk=(1, 5))
        loss_avg.update(loss.item(), input.size(0))
        accuracy1_avg.update(accuracy1.item(), input.size(0))
        accuracy5_avg.update(accuracy5.item(), input.size(0))
        loss.backward()
        if args.clip_gradient is not None:
            total_norm = nn.utils.clip_grad_norm_(tsn.parameters(), args.clip_gradient)
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if j % args.print_every == 0:
            print('Epoch: [{}/{}][{}/{}], lr {:.5f} Data {:.3f} ({:.3f}) Batch {:.3f} ({:.3f}) Loss {:.3f} ({:.3f}) Accuracy@1 {:.3f} ({:.3f}) Accuracy@5 {:.3f} ({:.3f})'.format(i, args.epoch, j, len(trainset_loader), optimizer.param_groups[-1]['lr'], data_time.val, data_time.avg, batch_time.val, batch_time.avg, loss_avg.val, loss_avg.avg, accuracy1_avg.val, accuracy1_avg.avg, accuracy5_avg.val, accuracy5_avg.avg))

    if (i + 1) % args.validation_every == 0 or i == args.epoch - 1:
        accuracy_ = validation()
        trained_model_name = '{}_{}_{}_{}_{:.3f}.pth'.format(args.base_model, args.dataset, args.modality, i, accuracy_)
        trained_model_path = 'trained_model/' + trained_model_name
        torch.save(tsn.state_dict(), trained_model_path)
