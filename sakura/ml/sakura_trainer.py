import pickle
from collections import OrderedDict
from sakura.ml.epoch import range


class SakuraTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 metrics,
                 epochs,
                 model_path,
                 checkpoint_path,
                 device="cpu",
                 device_test="cpu"):
        # self._r = redis.Redis(host='localhost', port=6379, db=0)
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._metrics = metrics
        self._epochs = range(0, epochs)
        self._model_path = model_path
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._device_test = device_test

    def run(self, train_loader, test_loader):
        raise NotImplementedError

    def train(self, loader):
        raise NotImplementedError

    def test(self, loader):
        raise NotImplementedError

    def update(self, current, best, loader):
        raise NotImplemented

    def serialized_state_dict(self):
        sd = OrderedDict()
        for k, v in self._model.cpu().state_dict().items():
            sd[k] = pickle.dumps(v)
        return sd

    def deserialized_state_dict(self, blob: bytes) -> OrderedDict:
        """Deserialize a state_dict produced by ``serialized_state_dict``.

        The MPI+Redis transport is gone — the blob now arrives over Zakuro
        (cloudpickle on the wire), so callers pass the raw pickled bytes
        directly instead of fetching from Redis.
        """
        sd = OrderedDict()
        for k, v in pickle.loads(blob).items():
            sd[k] = pickle.loads(v)
        return sd
