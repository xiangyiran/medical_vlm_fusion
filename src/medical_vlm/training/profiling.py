"""Original parameter, GFLOPs, latency, and memory profiling."""

import time

import numpy as np
import torch

from ..config import *
from ..utils import autocast_context

def count_parameters(model):
    """Count total and trainable parameters in millions."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6

def move_profile_batch_to_device(batch, device):
    """Move only model inputs to device for cost profiling."""
    return {
        "pixel_values": batch["pixel_values"].to(device, non_blocking=True),
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
    }

def profile_model_cost(model, loader, device):
    """Measure parameter count, GFLOPs, latency, and peak memory on one fixed batch."""
    model.eval().to(device)
    raw_batch = next(iter(loader))
    profile_batch = move_profile_batch_to_device(raw_batch, device)
    batch_size = profile_batch["pixel_values"].size(0)
    params_m, trainable_params_m = count_parameters(model)

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for _ in range(PROFILE_WARMUP_STEPS):
            with autocast_context():
                _ = model(**profile_batch)

    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.inference_mode():
        for _ in range(PROFILE_LATENCY_REPEAT):
            with autocast_context():
                _ = model(**profile_batch)

    if device == "cuda":
        torch.cuda.synchronize()

    latency_ms_per_batch = (time.time() - start_time) / max(PROFILE_LATENCY_REPEAT, 1) * 1000.0
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device == "cuda" else np.nan

    gflops_per_batch, gflops_per_sample = np.nan, np.nan
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, with_flops=True, record_shapes=False, profile_memory=False) as prof:
            with torch.inference_mode():
                with autocast_context():
                    _ = model(**profile_batch)
        total_flops = sum(getattr(event, "flops", 0) or 0 for event in prof.key_averages())
        if total_flops > 0:
            gflops_per_batch = total_flops / 1e9
            gflops_per_sample = gflops_per_batch / batch_size
    except Exception as e:
        print("GFLOPs profiling failed:", repr(e))

    return {
        "params_M": float(params_m),
        "trainable_params_M": float(trainable_params_m),
        "GFLOPs_per_batch": float(gflops_per_batch),
        "GFLOPs_per_sample": float(gflops_per_sample),
        "latency_ms_per_batch": float(latency_ms_per_batch),
        "peak_memory_MB": float(peak_memory_mb),
        "profile_batch_size": int(batch_size),
    }
