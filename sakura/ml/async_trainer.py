import os
from sakura import RecDict
from mpi4py import MPI
import logging
from collections import OrderedDict
import pickle
from sakura.functional import RecNamespace


class AsyncTrainer:
    def __init__(self,
                 trainer,
                 ):
        """ Initialize the distributed environment. """
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._trainer = trainer
        self._mode = "train" if self._rank == 0 else "test"

    def run(self, train_loader=None, test_loader=None):
        if self._mode == "train":
            # Run the trainer
            for self._trainer._epoch in self._trainer._epochs:
                req = self._comm.irecv(source=1)
                if req.get_status():
                    data = req.wait()
                    self._trainer._metrics.test = RecNamespace(
                        data["metrics"]["test"])
                else:
                    req.cancel()
                self._trainer.train(train_loader=train_loader)
                self._comm.send(
                    {
                        "state_dict": self._trainer.serialized_state_dict(),
                        "metrics": RecDict(self._trainer._metrics),
                    }, dest=1)
            self._comm.send("ACK", dest=1)

        else:
            # Run the trainer
            for self._trainer._epoch in self._trainer._epochs:
                sd = self._comm.recv(source=0)
                if sd == "ACK":
                    logging.warning("Ends")
                    return
                self._trainer._metrics.train = RecNamespace(
                    sd["metrics"]["train"])
                train_sd = sd["state_dict"]
                self.deserialize(train_sd, self._trainer._model)
                self._trainer.test(test_loader=test_loader)
                self._comm.isend(
                    {"metrics": RecDict(self._trainer._metrics)}, dest=0)

    @staticmethod
    def deserialize(_sd, model):
        sd = OrderedDict()
        for k, v in _sd.items():
            sd[k] = pickle.loads(v)
        model.load_state_dict(sd)
