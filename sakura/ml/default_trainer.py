from __future__ import print_function
import torch
from sakura import RecNamespace, RecDict
import json


class DefaultTrainer:
    def __init__(self, epochs, optimizer, scheduler, model, model_path, checkpoint_path, train_loader, test_loader,
                 device="cuda",
                 mode="train"):
        self._store=None
        self.state = RecNamespace({
                "shared":{
                        "epoch":{
                                "current": 1,
                                "best": 1,
                                "total": epochs,
                                "seconds_train": None
                            },
                        "metrics": {
                            "loss": {
                                "current": {
                                    "train": None,
                                    "test": None
                                },
                                "best": {
                                    "train": None,
                                    "test": None
                                }
                            },
                            "accuracy": {
                                "current": {
                                    "train": None,
                                    "test": None
                                },
                                "best": {
                                    "train": None,
                                    "test": None
                                },
                            }
                        },
                    },
                "opt":{
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                },
                "mtime": None
            })
        self._dist = None
        self._rank = None
        self._mode = mode
        self._model_path = model_path
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._model = model.to(device)
        self._train_loader = train_loader
        self._test_loader = test_loader

    def run(self):
        raise NotImplementedError

    def description(self):
        shared = json.loads(self._store.get('shared'))
        _metrics = RecNamespace(shared["metrics"])
        _best_epoch = RecNamespace(shared["epoch"]).best
        suffix = ""
        try:
            assert _metrics.accuracy.best.test is not None
            assert _metrics.accuracy.best.test > 0
            suffix += f" | Acc: {_metrics.accuracy.current.test:.4f} / ({_metrics.accuracy.best.test:.4f})"
            suffix += f" | Loss:{_metrics.loss.current.test:.4f} / ({_metrics.loss.best.test:.4f})"
        except:
            pass
        return f"({_best_epoch}) MNIST | Epoch: {self.state.shared.epoch.current}/{self.state.shared.epoch.total}{suffix}"

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
