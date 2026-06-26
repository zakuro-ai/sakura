"""ActivationCheckpoint — wrap matching submodules with torch.utils.checkpoint."""
from __future__ import annotations

from typing import Literal, Union

from sakura.events import OnTrainBegin
from sakura.service import BaseService


class ActivationCheckpoint(BaseService):
    name = "activation_checkpoint"
    priority = 15

    def __init__(
        self,
        *,
        target_types: tuple = (),
        selective: Union[bool, int, Literal["auto"]] = True,
        non_reentrant: bool = True,
        preserve_rng_state: bool = True,
    ):
        super().__init__()
        if not target_types:
            raise ValueError("target_types must include at least one nn.Module subclass")
        self._target_types = target_types
        self._selective = selective
        self._non_reentrant = non_reentrant
        self._preserve_rng_state = preserve_rng_state
        self.wrapped_count = 0

    def on_train_begin(self, event: OnTrainBegin):
        import torch.utils.checkpoint as _ck

        target_modules = []
        for module in event.model.modules():
            if isinstance(module, self._target_types):
                target_modules.append(module)

        if self._selective is True:
            wrap_indices = set(range(len(target_modules)))
        elif isinstance(self._selective, int):
            n = max(1, int(self._selective))
            wrap_indices = set(range(0, len(target_modules), n))
        elif self._selective == "auto":
            # Wrap every other one.
            wrap_indices = set(range(0, len(target_modules), 2))
        else:  # False
            wrap_indices = set()

        for i, mod in enumerate(target_modules):
            if i not in wrap_indices:
                continue
            original_forward = mod.forward
            use_reentrant = not self._non_reentrant
            preserve_rng = self._preserve_rng_state

            def make_wrapper(orig):
                def _ckpt_forward(*args, **kwargs):
                    return _ck.checkpoint(
                        orig, *args, use_reentrant=use_reentrant,
                        preserve_rng_state=preserve_rng, **kwargs,
                    )
                return _ckpt_forward

            mod.forward = make_wrapper(original_forward)
            self.wrapped_count += 1


__all__ = ["ActivationCheckpoint"]
