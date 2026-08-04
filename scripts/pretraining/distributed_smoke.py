"""Minimal CUDA/NCCL rendezvous test for the OptiBERTneo Slurm topology."""

import json
import os
import socket

import torch
import torch.distributed as dist


def main():
    required_world_size = int(os.environ.get("EXPECTED_WORLD_SIZE", "8"))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    try:
        if dist.get_world_size() != required_world_size:
            raise RuntimeError(
                f"expected world size {required_world_size}, "
                f"got {dist.get_world_size()}"
            )

        rank = dist.get_rank()
        value = torch.tensor(float(rank), device="cuda")
        dist.all_reduce(value)
        expected_sum = required_world_size * (required_world_size - 1) / 2
        if value.item() != expected_sum:
            raise RuntimeError(
                f"NCCL all-reduce produced {value.item()}, expected {expected_sum}"
            )

        properties = torch.cuda.get_device_properties(local_rank)
        record = {
            "rank": rank,
            "local_rank": local_rank,
            "host": socket.gethostname(),
            "device": properties.name,
            "capability": f"{properties.major}.{properties.minor}",
            "memory_gib": round(properties.total_memory / 2**30, 2),
        }
        records = [None] * required_world_size
        dist.all_gather_object(records, record)
        if rank == 0:
            print(json.dumps(records, indent=2, sort_keys=True))
            print(
                f"NCCL smoke passed: world_size={required_world_size}, "
                f"rank_sum={expected_sum:g}"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
