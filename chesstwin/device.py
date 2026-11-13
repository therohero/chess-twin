"""Pick the compute device: use CUDA when it really works, otherwise fall back to the CPU."""
import logging

log = logging.getLogger("chesstwin")


def pick_device(preferred=None):
    import torch
    if preferred == "cpu":
        return "cpu"
    if preferred in (None, "cuda") and torch.cuda.is_available():
        try:   # an old GPU/driver/wheel combination can report "available" but fail on the first kernel
            x = torch.zeros(1, 1, 8, 8, device="cuda")
            torch.nn.functional.conv2d(x, torch.zeros(1, 1, 3, 3, device="cuda"), padding=1).sum().item()
            return "cuda"
        except Exception as e:  # noqa: BLE001
            log.warning("CUDA is present but unusable (%s); falling back to CPU", e)
    return "cpu"


def describe(device):
    import torch
    if device == "cuda":
        return f"GPU ({torch.cuda.get_device_name(0)})"
    return "CPU"
