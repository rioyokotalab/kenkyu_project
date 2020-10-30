import os
import torch
import torch.distributed as dist

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '8888'
rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))
world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', '1'))
dist.init_process_group("nccl", rank=rank, world_size=world_size)
print('Rank: {}, Size: {}'.format(torch.distributed.get_rank(),torch.distributed.get_world_size()))

ngpus = 4
device = rank % ngpus
x = torch.randn(1).to(device)
print('rank {}: {}'.format(rank, x))
dist.broadcast(x, src=0)
print('rank {}: {}'.format(rank, x))
