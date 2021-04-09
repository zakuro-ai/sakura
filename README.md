![sakura Logo](imgs/sakura.png)

--------------------------------------------------------------------------------

Sakura is a simple but powerfull tool to reduce training time by running the train/test asynchronously. It provides two features:
- A simple ML framework for asynchronous training.
- An integration with PyTorch. 


You can reuse your favorite Python framework such as Pytorch, Tensorflow or PaddlePaddle.


## Sakura modules

At a granular level, Sakura is a library that consists of the following components:

| Component | Description |
| ---- | --- |
| **sakura** | Contains the sakura modules. |
| **sakura.ml** | Contains the code related to ml processing |
| **sakura.decorators** | Decorators used to synchronize the train/test.|

## Code design
If you worked with PyTorch in your project your would find a common structure. 
Simply change the `test` and `train` in your trainer as shown in `mnist_demo`. 
```python
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sakura.ml import AsyncTrainer
from sakura import defaultMetrics
from .trainer import Trainer
from .model import Net
from .utils import init_loaders, load_config

if __name__ == "__main__":
    # Load the config
    sakura = load_config()

    # Initialize
    model = Net()
    optimizer = optim.Adadelta(model.parameters(), lr=sakura.optim.lr)
    scheduler = StepLR(optimizer, step_size=sakura.optim.step, gamma=sakura.optim.gamma)

    # Build the trainer
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      metrics=defaultMetrics,
                      epochs=sakura.trainer.epochs,
                      model_path=sakura.trainer.model_path,
                      checkpoint_path=sakura.trainer.checkpoint_path,
                      device=sakura.trainer.device)

    # Comment the following line to disable to async trainer
    trainer = AsyncTrainer(trainer=trainer,
                           device_test=sakura.trainer.device_test)

    # Init the loaders
    train_loader, test_loader = init_loaders(seed=sakura.loader.seed,
                                             batch_size=sakura.loader.batch_size,
                                             test_batch_size=sakura.loader.test_batch_size,
                                             device=sakura.loader.device)

    # Run the trainer
    trainer.run(train_loader=train_loader,
                test_loader=test_loader)
```



## Installation
### Local
```
python setup.py install
```

### Pypi
```
pip install sakura-ml  --no-cache-dir
```

### Docker
To build the image and launch a container to run a test demo on MNIST.
```
docker pull zakuroai/sakura
sh docker.sh
```

### Try it!

```python
python -m mnist_demo
```
You should be able to see this output with no delay between epochs (asynchronous testing).
```
(1) MNIST | Epoch: 1/10: 100%||███████████████████████████████████████████████████████| 938/938 [00:12<00:00, 76.14it/s]
(1) MNIST | Epoch: 2/10: 100%||███████████████████████████████████████████████████████| 938/938 [00:12<00:00, 76.43it/s]
(2) MNIST | Epoch: 3/10 | Acc: 98.5000 / (98.5000) | Loss:0.0474 / (0.0474): 100%|███████████████████████████████████████████████████████| 938/938 [00:12<00:00, 76.13it/s]
(3) MNIST | Epoch: 4/10 | Acc: 98.7900 / (98.7900) | Loss:0.0376 / (0.0376): 100%|███████████████████████████████████████████████████████| 938/938 [00:12<00:00, 75.53it/s]
(4) MNIST | Epoch: 5/10 | Acc: 98.7900 / (98.7900) | Loss:0.0337 / (0.0337): 100%|███████████████████████████████████████████████████████| 938/938 [00:12<00:00, 74.20it/s]
(5) MNIST | Epoch: 6/10 | Acc: 98.9700 / (98.9700) | Loss:0.0310 / (0.0310): 100%|███████████████████████████████████████████████████████| 938/938 [00:12<00:00, 73.46it/s]
(6) MNIST | Epoch: 7/10 | Acc: 99.0000 / (99.0000) | Loss:0.0290 / (0.0290): 100%|███████████████████████████████████████████████████████| 938/938 [00:13<00:00, 71.63it/s]
(7) MNIST | Epoch: 8/10 | Acc: 99.1000 / (99.1000) | Loss:0.0273 / (0.0273): 100%|███████████████████████████████████████████████████████| 938/938 [00:12<00:00, 73.11it/s]
(7) MNIST | Epoch: 9/10 | Acc: 99.0900 / (99.1000) | Loss:0.0285 / (0.0273): 100%|███████████████████████████████████████████████████████| 938/938 [00:12<00:00, 75.10it/s]
(9) MNIST | Epoch: 10/10 | Acc: 99.1700 / (99.1700) | Loss:0.0267 / (0.0267): 100%|███████████████████████████████████████████████████████| 938/938 [00:12<00:00, 73.22it/s]
```

FYI the meaning of the above notation is:
```
([best_epoch]) [name_exp] | Epoch: [current]/[total] | Acc: [current_test_acc] / ([best_test_acc]) | Loss:[current_test_loss] / ([best_test_loss]): 100%|███| [batch_k]/[batch_n] [[time_train]<[time_left], [it/s]]
```