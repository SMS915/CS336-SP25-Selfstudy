import os
import torch
import timeit
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(master_addr, master_port, rank, world_size, use_cuda = False):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    cuda_available = torch.cuda.is_available()
    backend = "nccl" if (use_cuda and cuda_available) else "gloo"

    dist.init_process_group(backend, rank=rank, world_size=world_size)

def clean_up():
    dist.destroy_process_group()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def distributed_demo(rank, master_addr, master_port, world_size, use_cuda, tensor_size_mb, n_iter: int = 30):
    setup(master_addr, master_port, rank, world_size, use_cuda)
    if use_cuda:
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()
    device = f'cuda:{rank}' if use_cuda else 'cpu'
    backend = "nccl" if (use_cuda and torch.cuda.is_available()) else "gloo"
    tensor_size_bytes = tensor_size_mb * 1024 * 1024
    num_elements = tensor_size_bytes // 4
    data = torch.randn(num_elements, device=device)

    # warm-up
    for _ in range(5):
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
        if use_cuda:
            torch.cuda.synchronize()
    
    dist.barrier()
    print(f"rank {rank} data (before all-reduce): {data}")
    start_time = timeit.default_timer()
    for _ in range(n_iter):
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
    
    if use_cuda:
        torch.cuda.synchronize()
    end_time = timeit.default_timer()
    
    duration = end_time - start_time
    avg_time = duration / n_iter
    bandwidth = (tensor_size_bytes / avg_time) / 1e9

    if rank == 0:
        print(f"rank {rank} data (after all-reduce): {data}")
        print(f"backend: {backend}, device: {device}, world size: {world_size}, tensor size: {tensor_size_mb}MB")
        print(f"Average time per all-reduce: {avg_time * 1000:.4f}ms")
        print(f"Bandwidth: {bandwidth}GB/s")

    local_result = {
            'rank': rank,
            'world_size': world_size,
            'device': device,
            'tensor_size_mb': tensor_size_mb,
            'avg_time_ms': avg_time * 1000.0,
            'bandwidth_gbps': bandwidth
        }

    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, local_result)

    clean_up()
    if rank == 0:
        return gathered_results 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    master_addr = "localhost"
    master_port = "29500"
    world_size = args.world_size
    use_cuda = args.cuda == True
    tensor_size = args.size

    if world_size > 1:
        mp.spawn(fn=distributed_demo, args=(master_addr, master_port, world_size, use_cuda, tensor_size), nprocs=world_size, join=True)
    else:
        print("Not enough devices/processes to run distributed training.")

if __name__ == '__main__':
    main()


