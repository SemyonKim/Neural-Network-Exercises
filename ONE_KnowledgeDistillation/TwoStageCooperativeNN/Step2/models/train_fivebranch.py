"""
cifar_one.py

Training script for CIFAR-10/100 using ONE_ResNet with 5 branches.
"""

import argparse
import os
import shutil
import time
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
from loss import KLLoss
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, ramps

parser = argparse.ArgumentParser(description="PyTorch CIFAR10/100 Training (5 branches)")
parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"], type=str)
parser.add_argument("--workers", default=2, type=int)
parser.add_argument("--epochs", default=300, type=int)
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
parser.add_argument("--resume", default="", type=str, help="path to latest 5-branch checkpoint")
parser.add_argument("--evaluate", action="store_true")
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_cuda = torch.cuda.is_available()

if args.dataset == "cifar10":
    dataloader = datasets.CIFAR10
    num_classes = 10
else:
    dataloader = datasets.CIFAR100
    num_classes = 100

best_acc = 0

def main():
    global best_acc
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
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
    trainset = dataloader(root="./data", train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    testset = dataloader(root="./data", train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print(f"==> creating model '{args.arch}' with 5 branches")
    model = models.__dict__[args.arch](num_classes=num_classes, depth=args.depth)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print("    Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))

    criterion = nn.CrossEntropyLoss()
    criterion_kl = KLLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Logger
    logger = Logger(os.path.join(args.checkpoint, "log.txt"), title="cifar_fivebranch")
    logger.set_names(["TrainAcc_C1", "TestAcc_C1",
                      "TrainAcc_C2", "TestAcc_C2",
                      "TrainAcc_C3", "TestAcc_C3",
                      "TrainAcc_C4", "TestAcc_C4",
                      "TrainAcc_C5", "TestAcc_C5",
                      "TrainAcc_ensemble", "TestAcc_ensemble"])

    # Resume
    if args.resume:
        print("==> Resuming from checkpoint..")
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    if args.evaluate:
        print("\nEvaluation only")
        _, acc1, acc2, acc3, acc4, acc5, acc_en = test(testloader, model)
        print(f"Test Acc1: {acc1:.2f}, Test Acc2: {acc2:.2f}, Test Acc3: {acc3:.2f}, Test Acc4: {acc4:.2f}, Test Acc5: {acc5:.2f}, Ensemble: {acc_en:.2f}")
        return

    # Training loop
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print(f"\nEpoch: [{epoch+1}/{args.epochs}] LR: {state['lr']:.4f}")

        _, acc1, acc2, acc3, acc4, acc5, acc_en = train(trainloader, model, criterion, criterion_kl, optimizer, epoch)
        _, tacc1, tacc2, tacc3, tacc4, tacc5, tacc_en = test(testloader, model)

        logger.append([acc1, tacc1, acc2, tacc2, acc3, tacc3, acc4, tacc4, acc5, tacc5, acc_en, tacc_en])

        is_best = tacc1 > best_acc
        best_acc = max(tacc1, best_acc)
        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "acc": tacc1,
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, "log.eps"))
    print("Best acc:", best_acc)

def train(trainloader, model, criterion, criterion_kl, optimizer, epoch):
    model.train()
    losses = AverageMeter()
    top1_c1, top1_c2, top1_c3, top1_c4, top1_c5, top1_en = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    consistency_weight = get_current_consistency_weight(epoch)

    for inputs, targets in trainloader:
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

        outputs1, outputs2, outputs3, outputs4, outputs5, outputs6 = model(inputs)

        loss_cross = (criterion(outputs1, targets) + criterion(outputs2, targets) +
                      criterion(outputs3, targets) + criterion(outputs4, targets) +
                      criterion(outputs5, targets) + criterion(outputs6, targets))
        loss_kl = consistency_weight * (criterion_kl(outputs1, outputs6) +
                                        criterion_kl(outputs2, outputs6) +
                                        criterion_kl(outputs3, outputs6) +
                                        criterion_kl(outputs4, outputs6) +
                                        criterion_kl(outputs5, outputs6))
        loss = loss_cross + loss_kl

        for out, meter in zip([outputs1, outputs2, outputs3, outputs4, outputs5, outputs6],
                              [top1_c1, top1_c2, top1_c3, top1_c4, top1_c5, top1_en]):
            prec1, _ = accuracy(out.data, targets.data, topk=(1, 5))
            meter.update(prec1.item(), inputs.size(0))

        losses.update(loss.item(), inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg, top1_c1.avg, top1_c2.avg, top1_c3.avg, top1_c4.avg, top1_c5.avg, top1_en.avg

def test(testloader, model):
    model.eval()
    losses = AverageMeter()
    top1_c1, top1_c2, top1_c3, top1_c4, top1_c5, top1_en = (
        AverageMeter(), AverageMeter(), AverageMeter(),
        AverageMeter(), AverageMeter(), AverageMeter()
    )

    with torch.no_grad():
        for inputs, targets in testloader:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs1, outputs2, outputs3, outputs4, outputs5, outputs6 = model(inputs)

            for out, meter in zip(
                [outputs1, outputs2, outputs3, outputs4, outputs5, outputs6],
                [top1_c1, top1_c2, top1_c3, top1_c4, top1_c5, top1_en]
            ):
                prec1, _ = accuracy(out.data, targets.data, topk=(1, 5))
                meter.update(prec1.item(), inputs.size(0))

    return losses.avg, top1_c1.avg, top1_c2.avg, top1_c3.avg, top1_c4.avg, top1_c5.avg, top1_en.avg


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
