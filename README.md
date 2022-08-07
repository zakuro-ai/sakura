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

## Code design
If you worked with PyTorch in your project your would find a common structure. 
Simply change the `test` and `train` in your trainer as shown in `mnist_demo`. 
```python
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sakura.ml import AsyncTrainer
from sakura import defaultMetrics
from mnist_demo.trainer import Trainer
from mnist_demo.model import Net
from mnist_demo.utils import init_loaders, load_config

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
                      device=sakura.trainer.device,
                      device_test=sakura.trainer.device_test)

    # Comment the following line to disable to async trainer
    trainer = AsyncTrainer(trainer=trainer)

    # Init the loaders
    train_loader, test_loader = init_loaders(seed=sakura.loader.seed,
                                             batch_size=sakura.loader.batch_size,
                                             test_batch_size=sakura.loader.test_batch_size)

    # Run the trainer
    trainer.run(train_loader=train_loader,
                test_loader=test_loader)


```



## Installation (CPU/GPU)
### Local
Move the `config_template.yml` file into `$HOME/.config/sakura/config.yml` and install the wheel file. 
```
pip install dist/*.whl
pip install dist/*.whl  --extra-index-url https://download.pytorch.org/whl/cu116
```

### Pypi (CPU/GPU)
```
pip install sakura-ml 
pip install sakura-ml --extra-index-url https://download.pytorch.org/whl/cu116
```

### Docker
To build the image and launch a container to run a test demo on MNIST.
```
docker pull zakuroai/sakura
sh docker.sh
```

### Try it!

```python
sakura -m mnist_demo
```
You should be able to see this output with no delay between epochs (asynchronous testing).
```
   _____           _                               __  __   _      
  / ____|         | |                             |  \/  | | |     
 | (___     __ _  | | __  _   _   _ __    __ _    | \  / | | |     
  \___ \   / _` | | |/ / | | | | | '__|  / _` |   | |\/| | | |     
  ____) | | (_| | |   <  | |_| | | |    | (_| |   | |  | | | |____ 
 |_____/   \__,_| |_|\_\  \__,_| |_|     \__,_|   |_|  |_| |______|

(0) MNIST | Epoch: 1/10 | Acc: 0.0000 / (0.0000) | Loss:0.0000 / (0.0000): 100%|██████████| 18/18 [00:06<00:00,  2.69it/s]
(1) MNIST | Epoch: 2/10 | Acc: 0.0000 / (0.0000) | Loss:0.0000 / (0.0000): 100%|██████████| 18/18 [00:05<00:00,  3.36it/s]
(2) MNIST | Epoch: 3/10 | Acc: 90.4600 / (90.4600) | Loss:0.4034 / (0.4034): 100%|██████████| 18/18 [00:05<00:00,  3.42it/s]
(3) MNIST | Epoch: 4/10 | Acc: 95.3246 / (95.3246) | Loss:0.1907 / (0.1907): 100%|██████████| 18/18 [00:05<00:00,  3.43it/s]
(4) MNIST | Epoch: 5/10 | Acc: 96.9332 / (96.9332) | Loss:0.1379 / (0.1379): 100%|██████████| 18/18 [00:05<00:00,  3.38it/s]
(5) MNIST | Epoch: 6/10 | Acc: 97.3693 / (97.3693) | Loss:0.1167 / (0.1167): 100%|██████████| 18/18 [00:05<00:00,  3.42it/s]
(6) MNIST | Epoch: 7/10 | Acc: 97.7237 / (97.7237) | Loss:0.1040 / (0.1040): 100%|██████████| 18/18 [00:05<00:00,  3.41it/s]
(7) MNIST | Epoch: 8/10 | Acc: 98.0172 / (98.0172) | Loss:0.0938 / (0.0938): 100%|██████████| 18/18 [00:05<00:00,  3.31it/s]
(8) MNIST | Epoch: 9/10 | Acc: 98.2402 / (98.2402) | Loss:0.0886 / (0.0886): 100%|██████████| 18/18 [00:05<00:00,  3.41it/s]
```

FYI the meaning of the above notation is:
```
([best_epoch]) [name_exp] | Epoch: [current]/[total] | Acc: [current_test_acc] / ([best_test_acc]) | Loss:[current_test_loss] / ([best_test_loss]): 100%|███| [batch_k]/[batch_n] [[time_train]<[time_left], [it/s]]
```