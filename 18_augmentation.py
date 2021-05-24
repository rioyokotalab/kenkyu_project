import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os
import wandb
from models import *

def print0(message):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", postfix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.postfix = postfix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        entries += self.postfix
        print0('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def train(train_loader,model,criterion,optimizer,epoch,device):
    batch_time = AverageMeter('Time', ':.4f')
    train_loss = AverageMeter('Loss', ':.6f')
    train_acc = AverageMeter('Accuracy', ':.6f')
    progress = ProgressMeter(
        len(train_loader),
        [train_loss, train_acc, batch_time],
        prefix="Epoch: [{}]".format(epoch))
    model.train()
    t = time.perf_counter()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        train_loss.update(loss.item(), data.size(0))
        pred = output.data.max(1)[1]
        acc = 100. * pred.eq(target.data).cpu().sum() / target.size(0)
        train_acc.update(acc, data.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            batch_time.update(time.perf_counter() - t)
            t = time.perf_counter()
            progress.display(batch_idx)
    return train_loss.avg, train_acc.avg

def validate(val_loader,model,criterion,device):
    val_loss = AverageMeter('Loss', ':.6f')
    val_acc = AverageMeter('Accuracy', ':.1f')
    progress = ProgressMeter(
        len(val_loader),
        [val_loss, val_acc],
        prefix='\nValidation: ',
        postfix='\n')
    model.eval()
    for data, target in val_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        val_loss.update(loss.item(), data.size(0))
        pred = output.data.max(1)[1]
        acc = 100. * pred.eq(target.data).cpu().sum() / target.size(0)
        val_acc.update(acc, data.size(0))
    progress.display(len(val_loader))
    return val_loss.avg, val_acc.avg

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--bs', '--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', '--learning_rate', type=float, default=1.0e-02, metavar='LR',
                        help='learning rate (default: 1.0e-02)')
    args = parser.parse_args()

    master_addr = os.getenv("MASTER_ADDR", default="localhost")
    master_port = os.getenv('MASTER_PORT', default='8888')
    method = "tcp://{}:{}".format(master_addr, master_port)
    rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))
    world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', '1'))
    dist.init_process_group("nccl", init_method=method, rank=rank, world_size=world_size)
    ngpus = torch.cuda.device_count()
    device = torch.device('cuda',rank % ngpus)

    if rank==0:
        wandb.init()
        wandb.config.update(args)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10('./data',
                                     train=True,
                                     download=True,
                                     transform=transform_train)
    val_dataset = datasets.CIFAR10('./data',
                                   train=False,
                                   transform=transform_val)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.bs,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.bs,
                                             shuffle=False)
    # model = VGG('VGG19')
    # model = ResNet18()
    # model = PreActResNet18()
    # model = GoogLeNet()
    # model = DenseNet121()
    # model = ResNeXt29_2x64d()
    # model = MobileNet()
    # model = MobileNetV2()
    # model = DPN92()
    # model = ShuffleNetG2()
    # model = SENet18()
    # model = ShuffleNetV2(1)
    # model = EfficientNetB0()
    # model = RegNetX_200MF()
    model = VGG('VGG19').to(device)
    if rank==0:
        wandb.config.update({"model": model.__class__.__name__, "dataset": "CIFAR10"})
    model = DDP(model, device_ids=[rank % ngpus])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_acc = train(train_loader,model,criterion,optimizer,epoch,device)
        val_loss, val_acc = validate(val_loader,model,criterion,device)
        if rank==0:
            wandb.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
                })

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
