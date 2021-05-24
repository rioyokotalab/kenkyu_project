import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os

def print0(message):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

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
        if batch_idx % 200 == 0:
            batch_time.update(time.perf_counter() - t)
            t = time.perf_counter()
            progress.display(batch_idx)

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

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
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

    epochs = args.epochs
    batch_size = args.bs
    learning_rate = args.lr

    train_dataset = datasets.MNIST('./data',
                                   train=True,
                                   download=True,
                                   transform=transforms.ToTensor())
    val_dataset = datasets.MNIST('./data',
                                 train=False,
                                 transform=transforms.ToTensor())
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False)
    model = CNN().to(device)
    model = DDP(model, device_ids=[rank % ngpus])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train(train_loader,model,criterion,optimizer,epoch,device)
        validate(val_loader,model,criterion,device)

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
