import os
import json
import torch.distributed as dist
from torch.multiprocessing import Process
from sakura import RecDict
from datetime import timedelta


class AsyncTrainer:
    def __init__(self,
                 trainer,
                 device="cuda",
                 device_test="cpu",
                 backend='gloo',
                 host='127.0.0.1',
                 port=58604,
                 store_host='127.0.0.1',
                 store_port=58605):
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = host
        os.environ['MASTER_PORT'] = str(port)
        os.environ['STORE_PORT'] = str(store_port)
        os.environ['STORE_ADDR'] = str(store_host)
        self.__trainer = trainer
        self.backend=backend
        self.device=device
        self.world = [
            ("train", self.device),
            ("test", device_test)
        ]
        self.world_size = len(self.world)

    def run(self, *args, **kwargs):
        processes = []
        for rank, (mode, device) in enumerate(self.world):
            p = Process(target=self.init_process, args=(rank, args, kwargs, mode))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    def init_process(self, rank, args, kwargs, mode):
        dist.init_process_group(self.backend, rank=rank, world_size=self.world_size)
        store = dist.TCPStore(os.environ['STORE_ADDR'],
                              int(os.environ['STORE_PORT']),
                              len(self.world),
                              rank==0,
                              timedelta(seconds=30))
        self.__trainer._dist = dist
        self.__trainer._rank = rank
        self.__trainer._mode = mode
        self.__trainer._world_size=len(self.world)
        store.set("test", str(json.dumps(RecDict(self.__trainer._metrics.test))))
        self.__trainer._store = store
        self.__trainer.run(*args, **kwargs)

