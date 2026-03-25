#!/usr/bin/env python3

import time
import argparse
import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import setproctitle

from ..utils import *        
from ..data_loaders import KneeMRIDataset
from ..models import vnet


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(
        now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min
    )


def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')


def noop(x):
    return x

def multiclass_dice(pred, target, num_classes, ignore_index=0, eps=1e-5):
    """
    Compute mean Dice over all classes except ignore_index.

    pred: LongTensor of shape [N]     (predicted class indices)
    target: LongTensor of shape [N]   (ground-truth class indices)
    num_classes: total number of classes (e.g., 7)
    ignore_index: class to ignore in Dice (usually background=0)
    """
    dice_scores = []

    for c in range(num_classes):
        if c == ignore_index:
            continue

        pred_c = (pred == c).float()
        target_c = (target == c).float()

        # If this class is absent in both pred and target, skip it
        if target_c.sum() == 0 and pred_c.sum() == 0:
            continue

        intersection = (pred_c * target_c).sum()
        denom = pred_c.sum() + target_c.sum()
        dice_c = (2.0 * intersection + eps) / (denom + eps)
        dice_scores.append(dice_c.item())

    if len(dice_scores) == 0:
        # no foreground classes present at all
        return 0.0

    return float(sum(dice_scores) / len(dice_scores))


def train_nll(args, epoch, model, trainLoader, optimizer, trainF, weights):
    model.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)

    for batch_idx, (data, target) in enumerate(trainLoader):
        if batch_idx == 0:
            print("data shape:", data.shape)
            print("target shape:", target.shape)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        # VNet outputs [Nvoxels, num_classes]
        output = model(data)
        # Flatten target to [Nvoxels]
        target_flat = target.view(-1)

        loss = F.nll_loss(output, target_flat, weight=weights)

        loss.backward()
        optimizer.step()

        nProcessed += len(data)
        # hard predictions
        pred = output.data.max(1)[1]          # [Nvoxels]
        incorrect = pred.ne(target_flat.data).cpu().sum()
        err = 100.0 * incorrect / target_flat.numel()

        # Dice over classes 1..(num_classes-1)
        num_classes = output.size(1)
        dice = multiclass_dice(pred.cpu(), target_flat.cpu(), num_classes, ignore_index=0)

        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print(
            'Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\t'
            'Loss: {:.4f}\tError: {:.3f}%\tDice: {:.4f}'.format(
                partialEpoch,
                nProcessed,
                nTrain,
                100.0 * batch_idx / len(trainLoader),
                float(loss.item()),
                float(err),
                float(dice),
            )
        )

        # log: epoch_fraction, loss, error, dice
        trainF.write('{},{},{},{}\n'.format(
            partialEpoch, float(loss.item()), float(err), float(dice)
        ))
        trainF.flush()


def test_nll(args, epoch, model, testLoader, optimizer, testF, weights):
    model.eval()
    test_loss = 0.0
    incorrect = 0
    numel = 0

    # we’ll accumulate Dice over batches and average
    dice_accum = 0.0
    dice_batches = 0

    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        target_flat = target.view(-1)
        numel += target_flat.numel()

        output = model(data)
        test_loss += float(F.nll_loss(output, target_flat, weight=weights).data)

        pred = output.data.max(1)[1]
        incorrect += pred.ne(target_flat.data).cpu().sum()

        # Dice for this batch
        num_classes = output.size(1)
        dice = multiclass_dice(pred.cpu(), target_flat.cpu(), num_classes, ignore_index=0)
        dice_accum += dice
        dice_batches += 1

    test_loss /= len(testLoader)
    err = 100.0 * incorrect / numel
    mean_dice = dice_accum / max(dice_batches, 1)

    print(
        '\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.3f}%), '
        'Mean Dice: {:.4f}\n'.format(
            test_loss, incorrect, numel, float(err), float(mean_dice)
        )
    )

    # log: epoch, loss, error, dice
    testF.write('{},{},{},{}\n'.format(epoch, test_loss, float(err), float(mean_dice)))
    testF.flush()
    return err


def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150:
            lr = 1e-1
        elif epoch == 150:
            lr = 1e-2
        elif epoch == 225:
            lr = 1e-3
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=10)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    # inference path disabled for now (we’re not doing LUNA16-style inference)
    parser.add_argument('--weight-decay', '--wd', default=1e-8, type=float,
                        metavar='W', help='weight decay (default: 1e-8)')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()

    best_prec1 = 100.0
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work/vnet.knee.{}'.format(datestr())
    weight_decay = args.weight_decay

    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # DataLoader kwargs
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # Build VNet for 7 classes (0 = bg, 1..6 = cartilage compartments)
    print("build vnet")
    num_classes = 7
    nll = True  # we only support NLL now
    model = vnet.VNet(elu=False, nll=nll, num_classes=num_classes)

    batch_size = args.ngpu * args.batchSz
    gpu_ids = list(range(args.ngpu))
    if args.ngpu > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    if args.cuda:
        model = model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        model.apply(weights_init)

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()]))
    )

    # Fresh save directory
    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    # Datasets / loaders
    print("loading training set")
    trainSet = KneeMRIDataset(root='data', split='train')
    trainLoader = DataLoader(
        trainSet,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    print("loading validation set")
    testSet = KneeMRIDataset(root='data', split='valid')
    testLoader = DataLoader(
        testSet,
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )

    # 7-class weights (can tune later)
    class_weights = torch.ones(num_classes, dtype=torch.float32)
    if args.cuda:
        class_weights = class_weights.cuda()

    # Optimizer
    if args.opt == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=1e-1,
            momentum=0.99,
            weight_decay=weight_decay
        )
    elif args.opt == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            weight_decay=weight_decay
        )
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            weight_decay=weight_decay
        )

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')

    # Always use NLL training/testing
    train_fn = train_nll
    test_fn = test_nll

    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train_fn(args, epoch, model, trainLoader, optimizer, trainF, class_weights)
        err = test_fn(args, epoch, model, testLoader, optimizer, testF, class_weights)

        is_best = err < best_prec1
        if is_best:
            best_prec1 = err

        save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1
            },
            is_best,
            args.save,
            "vnet"
        )

    trainF.close()
    testF.close()


if __name__ == '__main__':
    main()
