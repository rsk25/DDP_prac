from functools import partial

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, size, blocking=True):
    """Distributed function here"""
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # send the tensor to process 1
        if blocking:
            dist.send(tensor=tensor, dst=1)
        else:
            req = dist.isend(tensor=tensor, dst=1)
            print('Rank 0 started sending')
    else:
        # receive tensor from process 0
        if blocking:
            dist.recv(tensor=tensor, src=0)
        else:
            req = dist.irecv(tensor=tensor, src=0)
            print('Rank 1 started receiving')
    req.wait()
    print('Rank', rank, ' has data ', tensor[0])


def init_process(rank, size, fn, backend='gloo'):
    """Initialize the distributed environment"""
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == '__main__':
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, partial(run, blocking=False)))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
