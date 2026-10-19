"""Exact row gathering for a device-resident or CPU-resident token embedding.

The caller chooses weight placement. This module never changes that placement,
pins the table, caches rows, quantizes, or reads environment configuration.
"""
import torch
import torch.nn.functional as F


def _device_route(weight_device, ids_device):
    if weight_device.type == "cpu" and ids_device.type == "cpu":
        return "native"
    if weight_device.type == "cuda" and ids_device.type == "cuda" and weight_device == ids_device:
        return "native"
    if weight_device.type == "cpu" and ids_device.type == "cuda":
        return "cpu_rows"
    raise ValueError(f"unsupported embedding device combination: weight={weight_device}, ids={ids_device}")


def _blocking_to(tensor, device):
    return tensor.to(device=device, non_blocking=False)


def _cpu_rows_to_device(weight, input_ids, output_device):
    # IDs are tiny for decode/verification. Blocking copies deliberately keep
    # both their contents and selected-row lifetime simple before optimization.
    cpu_ids = _blocking_to(input_ids, torch.device("cpu"))
    selected = F.embedding(cpu_ids, weight)
    return _blocking_to(selected, output_device)


def embedding_lookup(weight, input_ids):
    """Gather original weight rows, returning on the IDs' device.

    CPU/CPU and same-device CUDA/CUDA preserve the original F.embedding call.
    CPU weights with CUDA IDs synchronously gather on CPU and transfer only
    selected rows in their original dtype. Other device combinations fail.
    Index dtype/rank/range handling is delegated to F.embedding unchanged.
    """
    route = _device_route(weight.device, input_ids.device)
    if route == "native":
        return F.embedding(input_ids, weight)
    return _cpu_rows_to_device(weight, input_ids, input_ids.device)
