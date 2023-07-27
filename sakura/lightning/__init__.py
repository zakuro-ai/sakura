# publisher.py
import os
from typing import Any, Optional

import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
import pandas as pd
import torch
from IPython.display import display
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
from collections import OrderedDict
import pickle

from lightning.pytorch.callbacks import Callback
from mpi4py import MPI
import numpy as np
import redis
import bson
import argparse
import time

TRAINER_RANK = 0
VALIDATION_RANK = 1


def serialized_state_dict(model):
    sd = OrderedDict()
    device = model.device
    for k, v in model.cpu().state_dict().items():
        sd[k] = pickle.dumps(v)
    model.to(device)
    return sd


def deserialized_state_dict(state_dict):
    sd = OrderedDict()
    for k, v in state_dict.items():
        sd[k] = pickle.loads(v)
    return sd


class Comm:
    def __init__(self):
        self.r = redis.Redis(host="localhost", port=6379, db=0)
        p0 = self.r.pubsub()
        p0.subscribe("SakuraLightning-0")
        p1 = self.r.pubsub()
        p1.subscribe("SakuraLightning-1")
        self.p = {0: p0, 1: p1}

    def recv(self, source, blocking=True):
        while True:
            message = self.p[1 - source].get_message()
            if message:
                try:
                    assert type(message["data"]) == bytes
                    d = bson.loads(message["data"])
                    return d
                except:
                    time.sleep(0.01)
            if not blocking:
                return None

    def send(self, msg, dest):
        self.r.publish(f"SakuraLightning-{dest}", bson.dumps(msg))


class SakuraLightning(Callback):
    def __init__(self, rank=0, *args, output_dir="/opt/zakuro/logs", **kwargs):
        super(SakuraLightning, self).__init__(*args, *kwargs)
        self._validation_loss = []
        self._training_loss = []
        self._best_val_loss = None
        self._comm = Comm()
        self._rank = rank
        self.epoch = 0
        self._log_file = f"{output_dir}/sakuraLightning.log"
        if self._rank == VALIDATION_RANK:
            with open(self._log_file, "w") as f:
                f.write("")
            self._comm.send({"message": "ready"}, dest=TRAINER_RANK)

    def on_validation_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        msg = self._comm.recv(source=TRAINER_RANK)
        while not msg["epoch"] == self.epoch:
            time.sleep(0.01)
            msg = self._comm.recv(source=TRAINER_RANK)

        sd = deserialized_state_dict(msg["state_dict"])
        # log(f"@{self._rank} received {len(sd)} dictionary")
        _sd = pl_module.state_dict()
        for (_k, _v), (k, v) in zip(_sd.items(), sd.items()):
            _sd[_k] = v
        pl_module.load_state_dict(_sd)

    def on_train_epoch_end(self, trainer, *args, **kwargs):
        self.__log(
            f"@{self._rank}#on_train_epoch_end"
        ) if trainer.global_rank == 0 else None
        self._comm.send(
            {
                "state_dict": serialized_state_dict(trainer.model),
                "state": "TRAINING",
                "epoch": self.epoch,
            },
            dest=VALIDATION_RANK,
        )
        msg = self._comm.recv(source=VALIDATION_RANK, blocking=False)
        if msg is not None:
            self.__log(
                f"@{self._rank} received {msg} from val"
            ) if trainer.global_rank == 0 else None
        self.epoch += 1

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._validation_loss.append(float(outputs.cpu()))

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        val_loss = np.mean(self._validation_loss)
        try:
            assert val_loss > self._best_val_loss
        except:
            self._best_val_loss = val_loss
            trainer.save_checkpoint(self.model_path)
        self.__log(f"@{self._rank} sent {str(val_loss)} to trainer")
        self._comm.send(
            {"epoch": self.epoch, "avg_val_loss": str(val_loss)}, dest=TRAINER_RANK
        )
        self._validation_loss = []
        self.epoch += 1

    def __log(self, msg):
        with open(self._log_file, "a") as f:
            f.write(f"{msg}\n")


class SakuraTrainer:
    def __init__(
        self,
        *args,
        accelerator="auto",
        kwargs_train=None,
        kwargs_val=None,
        **kwargs,
    ) -> None:
        role = int(os.environ["SAKURA_ROLE"])
        if (kwargs_train is not None) and (role == TRAINER_RANK):
            kwargs = kwargs.update(**kwargs_train)
        elif (kwargs_val is not None) and (role == VALIDATION_RANK):
            kwargs = kwargs.update(**kwargs_val)
        self._special_callbacks = SakuraLightning(rank=role)
        self._trainer = L.Trainer(
            *args,
            accelerator=accelerator if role == TRAINER_RANK else "cpu",
            callbacks=[self._special_callbacks],
            **kwargs,
            enable_progress_bar=role == TRAINER_RANK,
        )
        self._role = role

    def run(
        self,
        model,
        train_loader,
        val_loader,
        *args,
        kwargs_train=None,
        kwargs_val=None,
        model_path=None,
        **kwargs,
    ):
        self._special_callbacks.model_path = model_path
        if (kwargs_train is not None) and (self._role == TRAINER_RANK):
            kwargs = kwargs.update(**kwargs_train)
        elif (kwargs_val is not None) and (self._role == VALIDATION_RANK):
            kwargs = kwargs.update(**kwargs_val)
        if self._role == TRAINER_RANK:
            self._trainer.fit(
                model,
                train_loader,
                *args,
                **kwargs,
            )
        else:
            [
                self._trainer.validate(
                    model,
                    val_loader,
                    *args,
                    **kwargs,
                )
                for _ in range(self._trainer.max_epochs)
            ]
