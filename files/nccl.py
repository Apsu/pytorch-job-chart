import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import numpy as np
import argparse

def init_process(global_rank, world_size, local_rank, master_addr, master_port, backend='nccl'):
"""Initialize the distributed environment."""
# Set environment variables for distributed communication
os.environ['MASTER_ADDR'] = master_addr
os.environ['MASTER_PORT'] = master_port
os.environ['RANK'] = str(global_rank)
os.environ['WORLD_SIZE'] = str(world_size)

# Initialize the process group for distributed training
dist.init_process_group(backend=backend, rank=global_rank, world_size=world_size)
torch.cuda.set_device(local_rank)

def cleanup():
"""Cleans up the distributed process group."""
dist.destroy_process_group()

def measure_bandwidth(elapsed_times, num_bytes, op_multiplier=2, world_size=1):
"""Calculates and returns the average bandwidth over all iterations."""
avg_elapsed = np.mean(elapsed_times)

# Per-rank bandwidth calculation (average over all iterations)
per_rank_bandwidth = (num_bytes * op_multiplier * (world_size - 1) / world_size) / avg_elapsed / 1e9  # GB/s

# Aggregate bandwidth across all ranks
aggregate_bandwidth = (num_bytes * op_multiplier * (world_size - 1)) / avg_elapsed / 1e9  # GB/s

return per_rank_bandwidth, aggregate_bandwidth

def all_reduce_test(global_rank, local_rank, tensor_size, batch=1, iterations=1):
"""Performs all_reduce test with the given tensor size and repetitions."""
device = torch.device(f'cuda:{local_rank}')
tensor = torch.randn(tensor_size, device=device)
matrix = torch.randn((tensor_size, tensor_size), device=device)

for i in range(batch):
    elapsed_times = []

    for _ in range(iterations):
        start_time = time.time()
        torch.matmul(matrix, matrix)
        dist.all_reduce(tensor)
        torch.cuda.synchronize()  # Ensure all operations are complete before timing
        elapsed_times.append(time.time() - start_time)

    num_bytes = tensor.nelement() * tensor.element_size()  # Total size of the tensor in bytes
    per_rank_bw, aggregate_bw = measure_bandwidth(elapsed_times, num_bytes, world_size=dist.get_world_size())

    # Only rank 0 prints the bandwidth results
    if local_rank == 0:
        print(f"Batch {i+1}/{batch} | Tensor Size: {tensor_size} | "
              f"Avg Bandwidth per Rank: {per_rank_bw:.2f} GB/s | Aggregate: {aggregate_bw:.2f} GB/s", flush=True)

def run_worker(local_rank, world_size, base_rank, node_rank, start, stop, batch, repeat, master_addr, master_port):
"""Worker function to run the all_reduce performance test."""
global_rank = base_rank + local_rank  # Calculate global rank for this process
init_process(global_rank, world_size, local_rank, master_addr, master_port)

if global_rank == 0:
    print(f"Rank 0 (Coordinator) waiting for other processes to finish...", flush=True)
    dist.barrier()  # Rank 0 just waits for other ranks to complete the operations
else:
    # Run all_reduce test for tensor sizes from 2^start to 2^stop
    for size_power in range(start, stop + 1):
        tensor_size = 2 ** size_power  # Tensor size as a power of 2
        all_reduce_test(global_rank, local_rank, tensor_size, batch, repeat)

cleanup()

def worker_fn(local_rank, num_gpus, world_size, node_rank, start, stop, batch, repeat, master_addr, master_port):
"""Top-level worker function to spawn per GPU process."""
base_rank = num_gpus * node_rank  # Calculate base rank for this node
run_worker(local_rank, world_size, base_rank, node_rank, start, stop, batch, repeat, master_addr, master_port)

def spawn_processes_on_node(num_gpus, world_size, node_rank, master_addr, master_port, start, stop, batch, repeat):
"""Spawns one process per GPU on this node."""
# Call the top-level worker function for each process (GPU)
mp.spawn(worker_fn, args=(num_gpus, world_size, node_rank, start, stop, batch, repeat, master_addr, master_port), nprocs=num_gpus)


if __name__ == "__main__":
parser = argparse.ArgumentParser(description="PyTorch Distributed All Reduce Performance Test")
parser.add_argument("--master_addr", type=str, default="127.0.0.1", help="Master node IP address")
parser.add_argument("--master_port", type=str, default="29500", help="Master node port")
parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes in the cluster")
parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs per node")
parser.add_argument("--node_rank", type=int, default=0, help="Rank of this node (starting from 0)")
parser.add_argument("--batch", type=int, default=1, help="Number of batches per test")
parser.add_argument("--repeat", type=int, default=5, help="Number of iterations per batch")
parser.add_argument("--start", type=int, default=16, help="Start power of 2 for tensor size (e.g., 2^16)")
parser.add_argument("--stop", type=int, default=26, help="Stop power of 2 for tensor size (e.g., 2^26)")

args = parser.parse_args()

# Calculate total world size (total processes across all nodes)
world_size = args.num_nodes * args.num_gpus

# Spawn processes per node
spawn_processes_on_node(args.num_gpus, world_size, args.node_rank, args.master_addr, args.master_port,
                        args.start, args.stop, args.batch, args.repeat)

