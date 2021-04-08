import time
import numpy as np
import torch
import json
from sakura import RecDict


def synchronize(func):
    def wrapper(*args, **kwargs):
        if func.__name__ == "train":
            return train(func, *args, **kwargs)
        else:
            return test(func, *args, **kwargs)
    return wrapper


def train(func, *args, **kwargs):
    self  = args[0]
    t0 = time.time()

    func(*args, **kwargs)

    self._model.cpu()
    for tag, v in enumerate(self._model.state_dict().values()):
        self._dist.send(v, 1 - self._rank, tag=tag)
    self._model.cuda()
    self.state.opt.scheduler.step()


    self.state.shared.metrics.accuracy.current.train = 100. * self._correct / len(self._train_loader.dataset)
    self.state.shared.metrics.loss.current.train = np.mean(self._avg_loss)
    self.state.shared.epoch.seconds_train = time.time() - t0
    try:
        assert self.state.shared.metrics.accuracy.best.train is not None
        assert self.state.shared.metrics.accuracy.best.train > self.state.shared.metrics.accuracy.current.train
    except AssertionError:
        self.state.shared.metrics.accuracy.best.train = self.state.shared.metrics.accuracy.current.train
        self.state.shared.metrics.loss.best.train = self.state.shared.metrics.loss.current.train


def test(func, *args, **kwargs):
    self  = args[0]
    # Init the metrics to 0
    metrics = self.state.shared.metrics
    metrics.loss.current.test = 0

    for tag, (k, v) in enumerate(self._model.state_dict().items()):
        self._dist.recv(v, 1 - self._rank, tag=tag)
    func(*args, **kwargs)

    # Update the metrics
    metrics.loss.current.test = self._loss / len(self._test_loader.dataset)
    metrics.accuracy.current.test = 100. * self._correct / len(self._test_loader.dataset)
    try:
        assert metrics.accuracy.best.test is not None
        assert metrics.accuracy.best.test > metrics.accuracy.current.test
    except AssertionError:
        metrics.accuracy.best.test = metrics.accuracy.current.test
        metrics.loss.best.test = metrics.loss.current.test
        self.state.shared.epoch.best = self.state.shared.epoch.current + 1
        torch.save(self._model.state_dict(), self._model_path)
    try:
        self._store.set("shared", json.dumps(RecDict(self.state.shared)))
    except RuntimeError:
        return


