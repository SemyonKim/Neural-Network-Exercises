"""
train_onebranch.py

Training script for CIFAR10/100 using ONE_ResNet with a single branch.
"""

import argparse
import os
import random
import shutil
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

from models.resnet_onebranch import ONE_ResNet, one_resnet
from loss import KLLoss
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig, ramps

parser = argparse.ArgumentParser(description="ONE ResNet (1 branch) Training")
parser.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100"], type=str)
parser.add_argument("--epochs", default=160, type=int)
parser.add_argument("--train-batch", default=128, type=int)
parser.add_argument("--test-batch", default=100, type=int)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight-decay", default=1e-4, type=float)
parser.add_argument("--schedule", type=int, nargs="+", default=[151, 225])
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--depth", default=32, type=int)
parser.add_argument("--consistency_rampup", default=80, type=float)
parser.add_argument("--gpu-id", default="0", type=str)
parser.add_argument("--checkpoint", default="checkpoints/cifar100/ONE-32-rampup", type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_cuda = torch.cuda.is_available()
random.seed(42)
torch.manual_seed(42)
if use_cuda:
    torch.cuda.manual_seed_all(42)

best_acc = 0
state = {"lr": args.lr}

def main():
    global best_acc
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    dataloader = datasets.CIFAR100 if args.dataset == "cifar100" else datasets.CIFAR10
    num_classes = 100 if args.dataset == "cifar100" else 10

    trainset = dataloader(root="./data", train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=2)

    testset = dataloader(root="./data", train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=2)

    # Model
    print("==> Creating ONE_ResNet (1 branch)")
    model = ONE_ResNet(depth=args.depth, num_classes=num_classes)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print(f"    Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()
    criterion_kl = KLLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    logger = Logger(os.path.join(args.checkpoint, "log.txt"), title="cifar_onebranch")
    logger.set_names(["TrainAcc_branch", "TestAcc_branch", "TrainAcc_ensemble", "TestAcc_ensemble"])

    # Training loop
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print(f"\nEpoch: [{epoch+1}/{args.epochs}] LR: {state['lr']:.4f}")

        train_loss, train_acc_branch, train_acc_en = train(trainloader, model, criterion, criterion_kl, optimizer, epoch)
        test_loss, test_acc_branch, test_acc_en = test(testloader, model)

        logger.append([train_acc_branch, test_acc_branch, train_acc_en, test_acc_en])

        # Save checkpoint
        is_best = test_acc_branch > best_acc
        best_acc = max(test_acc_branch, best_acc)
        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "acc": test_acc_branch,
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, "log.eps"))
    print("Best acc:", best_acc)


def train(trainloader, model, criterion, criterion_kl, optimizer, epoch):
    model.train()
    losses, losses_kl = AverageMeter(), AverageMeter()
    top1_branch, top1_en = AverageMeter(), AverageMeter()

    consistency_weight = get_current_consistency_weight(epoch)

    for inputs, targets in trainloader:
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

        outputs_branch, outputs_en = model(inputs)

        loss_cross = criterion(outputs_branch, targets) + criterion(outputs_en, targets)
        loss_kl = consistency_weight * criterion_kl(outputs_branch, outputs_en)
        loss = loss_cross + loss_kl

        prec1_branch, _ = accuracy(outputs_branch.data, targets.data, topk=(1, 5))
        prec1_en, _ = accuracy(outputs_en.data, targets.data, topk=(1, 5))

        top1_branch.update(prec1_branch.item(), inputs.size(0))
        top1_en.update(prec1_en.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))
        losses_kl.update(loss_kl.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg, top1_branch.avg, top1_en.avg


def test(testloader, model):
    model.eval()
    losses = AverageMeter()
    top1_branch, top1_en = AverageMeter(), AverageMeter()

    with torch.no_grad():
        for inputs, targets in testloader:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs_branch, outputs_en = model(inputs)

            prec1_branch, _ = accuracy(outputs_branch.data, targets.data, topk=(1, 5))
            prec1_en, _ = accuracy(outputs_en.data, targets.data, topk=(1, 5))

            top1_branch.update(prec1_branch.item(), inputs.size(0))
            top1_en.update(prec1_en.item(), inputs.size(0))

    return losses.avg, top1_branch.avg, top1_en.avg


def save_checkpoint(state, is_best, checkpoint="checkpoint", filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth.tar"))


def get_current_consistency_weight(epoch):
    return ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def adjust_learning_rate(optimizer, epoch):
    if epoch in args.schedule:
        state["lr"] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group["lr"] = state["lr"]


if __name__ == "__main__":
    main()
