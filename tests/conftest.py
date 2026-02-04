import os
import torch

# Enable Triton cache (uses ~/.triton/cache by default)
# Set explicit cache dir for reproducibility
os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))

# Enable torch.compile/inductor caching
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torch_inductor"))
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

# Reduce recompilation by setting cache size
torch._dynamo.config.cache_size_limit = 256
