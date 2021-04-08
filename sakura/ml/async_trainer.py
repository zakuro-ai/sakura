from __future__ import print_function
import os
import torch.distributed as dist
from torch.multiprocessing import Process
import json
from sakura import RecDict
from datetime import timedelta


class AsyncTrainer:
    def __init__(self, cls, device="cuda", backend='gloo'):
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        self.cls = cls
        self.backend=backend
        self.device=device
        self.world = [
            ("train", self.device),
            ("test", "cpu")
        ]
        self.world_size = len(self.world)

    def run(self, *args, **kwargs):
        processes = []
        for rank, (mode, device) in enumerate(self.world):
            kwargs.update({"mode": mode, "device":device})
            p = Process(target=self.init_process, args=(rank, args, kwargs))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    def init_process(self, rank, args, kwargs):
        dist.init_process_group(self.backend, rank=rank, world_size=self.world_size)
        store =  dist.TCPStore("127.0.0.1", 1234, 2, rank==0, timedelta(seconds=30))
        trainer = self.cls(*args, **kwargs)
        store.set("shared", str(json.dumps(RecDict(trainer.state.shared))))
        trainer._store = store
        trainer._dist = dist
        trainer._rank = rank
        trainer.run()

