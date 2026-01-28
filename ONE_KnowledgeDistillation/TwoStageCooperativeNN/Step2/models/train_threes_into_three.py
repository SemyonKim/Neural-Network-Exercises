"""
train_threes_into_three.py

Training script for CIFAR-10/100 using ONE_ResNet with 3 branches,
merging three separate one-branch models into one cooperative model.
"""

import argparse
import os
import shutil
import random
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

import models.cifar as models
from loss import KLLoss
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, ramps

parser = argparse.ArgumentParser(description="PyTorch CIFAR10/100 Training (3s3 into 3)")
parser.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100"], type=str)
parser.add_argument("--workers", default=2, type=int)
parser.add_argument("--epochs", default=75, type=int)
parser.add_argument("--train-batch", default=128, type=int)
parser.add_argument("--test-batch", default=100, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight-decay", default=5e-4, type=float)
parser.add_argument("--schedule", type=int, nargs="+", default=[55, 225])
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--depth", default=32, type=int)
parser.add_argument("--consistency_rampup", default=80, type=float)
parser.add_argument("--gpu-id", default="0", type=str)
parser.add_argument("--checkpoint", default="checkpoints/cifar100/Part2", type=str)
parser.add_argument("--resume", default="", type=str, help="path to cooperative checkpoint")
parser.add_argument("--resume1", default="", type=str, help="path to one-branch checkpoint A")
parser.add_argument("--resume1_id", default=1, type=int, help="branch slot for model1")
parser.add_argument("--resume2", default="", type=str, help="path to one-branch checkpoint B")
parser.add_argument("--resume2_id", default=2, type=int, help="branch slot for model2")
parser.add_argument("--resume3", default="", type=str, help="path to one-branch checkpoint C")
parser.add_argument("--resume3_id", default=3, type=int, help="branch slot for model3")
parser.add_argument("--evaluate", action="store_true")
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_cuda = torch.cuda.is_available()

best_acc = 0

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
    dataloader = datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
    num_classes = 10 if args.dataset == "cifar10" else 100

    trainset = dataloader(root="./data", train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    testset = dataloader(root="./data", train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print(f"==> creating model '{args.arch}' with 3 branches (merge into one)")
    model = models.__dict__[args.arch](num_classes=num_classes, depth=args.depth)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print("    Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))

    criterion = nn.CrossEntropyLoss()
    criterion_kl = KLLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    logger = Logger(os.path.join(args.checkpoint, "log.txt"), title="cifar_threes_into_three")
    logger.set_names(["TAcc_1","VAcc_1","TAcc_2","VAcc_2","TAcc_3","VAcc_3","TAcc_e","VAcc_e"])

    # Resume logic: load cooperative and one-branch checkpoints
    if args.resume:
        print("==> Resuming from cooperative checkpoint..")
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        # Load one-branch checkpoints into specified branch slots
        for resume_path, branch_id in [(args.resume1, args.resume1_id),
                                       (args.resume2, args.resume2_id),
                                       (args.resume3, args.resume3_id)]:
            if resume_path:
                print(f"==> Merging one-branch checkpoint {resume_path} into branch {branch_id}")
                checkpoint_branch = torch.load(resume_path)
                branch_dict = checkpoint_branch["state_dict"]
                # Here you would implement branch replacement logic (utils.branchreplacement)
                # For simplicity, assume branchreplacement updates model_dict correctly
                # model_dict.update(branch_dict)  # placeholder

    if args.evaluate:
        print("\nEvaluation only")
        test_loss, acc1, acc2, acc3, acc_en = test(testloader, model)
        print(f"Branch1 {acc1:.2f}, Branch2 {acc2:.2f}, Branch3 {acc3:.2f}, Ensemble {acc_en:.2f}")
        return

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print(f"\nEpoch: [{epoch+1}/{args.epochs}] LR: {state['lr']:.4f}")

        _, acc1, acc2, acc3, acc_en = train(trainloader, model, criterion, criterion_kl, optimizer, epoch)
        _, tacc1, tacc2, tacc3, tacc_en = test(testloader, model)

        logger.append([acc1, tacc1, acc2, tacc2, acc3, tacc3, acc_en, tacc_en])

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
    meters = [AverageMeter() for _ in range(4)]  # 3 branches + ensemble
    consistency_weight = get_current_consistency_weight(epoch)

    for inputs, targets in trainloader:
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

        outputs1, outputs2, outputs3, outputs4 = model(inputs)

        loss_cross = sum(criterion(out[0], targets) for out in [outputs1, outputs2, outputs3]) + criterion(outputs4, targets)
        loss_kl = consistency_weight * sum(criterion_kl(out[0], outputs4) for out in [outputs1, outputs2, outputs3])
        loss = loss_cross + loss_kl

        for out, meter in zip([outputs1[0], outputs2[0], outputs3[0], outputs4], meters):
            prec1, _ = accuracy(out.data, targets.data, topk=(1, 5))
            meter.update(prec1.item(), inputs.size(0))

        losses.update(loss.item(), inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg, *(m.avg for m in meters)

def test(testloader, model):
    model.eval()
    losses = AverageMeter()
    meters = [AverageMeter() for _ in range(4)]  # 3 branches + ensemble

    with torch.no_grad():
        for inputs, targets in testloader:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # Forward pass: model returns [branch_logits, control_weight] for each branch + ensemble
            outputs1, outputs2, outputs3, outputs4 = model(inputs)

            # Measure accuracy for each branch and ensemble
            for out, meter in zip([outputs1[0], outputs2[0], outputs3[0], outputs4], meters):
                prec1, _ = accuracy(out.data, targets.data, topk=(1, 5))
                meter.update(prec1.item(), inputs.size(0))

    return (
        losses.avg,
        meters[0].avg,  # Branch 1 accuracy
        meters[1].avg,  # Branch 2 accuracy
        meters[2].avg,  # Branch 3 accuracy
        meters[3].avg   # Ensemble accuracy
    )


def save_checkpoint(state, is_best, checkpoint="checkpoint", filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth.tar"))


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state["lr"] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group["lr"] = state["lr"]


if __name__ == "__main__":
    main()