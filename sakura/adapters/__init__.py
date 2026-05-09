"""sakura.adapters — per-framework bridges that translate hooks to runtime events."""
from sakura.adapters.base import Adapter
from sakura.adapters.ddp import DDPAdapter
from sakura.adapters.huggingface import HFAdapter
from sakura.adapters.lightning import LightningAdapter

__all__ = ["Adapter", "DDPAdapter", "HFAdapter", "LightningAdapter"]
