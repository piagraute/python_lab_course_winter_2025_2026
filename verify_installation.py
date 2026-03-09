# Run this script with `uv run verify_installation.py`.

import torch

x = torch.rand(5, 3)
print(x)

print(f"{torch.cuda.is_available()=}")
