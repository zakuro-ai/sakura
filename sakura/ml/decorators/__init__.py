import json
from sakura import RecDict
from sakura import RecNamespace


def parallel(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        self._metrics.test = RecNamespace(eval(self._store.get("test").decode()))
        mode = self._mode
        if mode =="train":
            if func.__name__ == "train":
                return train(func, *args, **kwargs)
            elif func.__name__ == "description":
                return description(func, *args, **kwargs)
        elif mode == "test":
            if func.__name__ == "test":
                return test(func, *args, **kwargs)
            elif func.__name__ == "checkpoint":
                return checkpoint(func, *args, **kwargs)

    return wrapper


def train(func, *args, **kwargs):
    # Train
    func(*args, **kwargs)
    self = args[0]
    if self._world_size>1:
        self._model.cpu()
        for tag, v in enumerate(self._model.state_dict().values()):
            self._dist.send(v, 1 - self._rank, tag=tag)
        self._model.cuda()


def test(func, *args, **kwargs):
    self  = args[0]
    self._model.cpu()
    if self._world_size>1:
        for tag, (k, v) in enumerate(self._model.state_dict().items()):
            self._dist.recv(v, 1 - self._rank, tag=tag)
    self._model.cuda()
    func(*args, **kwargs)
    self._store.set("test", json.dumps(RecDict(self._metrics.test)))


