import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("hip version:", getattr(torch.version, "hip", None))

if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    print("matmul ok:", y.mean().item())
else:
    raise SystemExit("No ROCm/AMD GPU detected by PyTorch.")