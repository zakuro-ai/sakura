<h1 align="center">
  <br>
  <img src="https://drive.google.com/uc?id=1Mz2WqXHrwEOjwtWfJVHV7NiRwC_64Shh">
</h1>
<p align="center">
  <a href="#modules">Modules</a> •
  <a href="#code-structure">Code structure</a> •
  <a href="#code-design">Code design</a> •
  <a href="#installing-the-application">Installing the application</a> •
  <a href="#makefile-commands">Makefile commands</a> •
  <a href="#environments">Environments</a> •
  <a href="#running-the-application">Running the application</a>
</p>


--------------------------------------------------------------------------------

Sakura is a simple but powerfull tool to reduce training time by running the train/test asynchronously. It provides two features:
- A simple ML framework for asynchronous training.
- An integration with PyTorch. 


You can reuse your favorite Python framework such as Pytorch, Tensorflow or PaddlePaddle.


# Modules

At a granular level, Sakura is a library that consists of the following components:

| Component | Description |
| ---- | --- |
| **sakura** | Contains the sakura modules. |
| **sakura.ml** | Contains the code related to ml processing |



# Code structure
```python
from setuptools import setup
from sakura import __version__

setup(
    name="sakura-ml",
    version=__version__,
    short_description="Sakura provides asynchronous training for DNN.",
    long_description="Sakura provides asynchronous training for DNN.",
    url='https://zakuro.ai',
    packages=[
        "sakura",
        "sakura.ml",
        "sakura.ml.epoch",
    ],
    entry_points={
        "console_scripts": [
            "sakura=sakura:main"
        ]
    },
    include_package_data=True,
    package_data={"": ["*.yml"]},
    install_requires=[r.rsplit()[0] for r in open("requirements.txt")],
    license='MIT',
    author='ZakuroAI',
    python_requires='>=3.6',
    author_email='git@zakuro.ai',
    description='Sakura provides asynchronous training for DNN.',
    platforms="linux_debian_10_x86_64",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ]
)
```
# Code design
If you worked with PyTorch in your project your would find a common structure. 
Simply change the `test` and `train` in your trainer as shown in `mnist_demo`. 
```python
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sakura.ml import AsyncTrainer
from sakura import defaultMetrics
from mnist_demo.trainer import Trainer
from mnist_demo.model import Net
from mnist_demo.utils import init_loaders
from sakura import cfg

if __name__ == "__main__":
    # Initialize
    model = Net()
    optimizer = optim.Adadelta(model.parameters(), lr=cfg.optim.lr)
    scheduler = StepLR(optimizer, step_size=cfg.optim.step,
                       gamma=cfg.optim.gamma)

    # Build the trainer
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      metrics=defaultMetrics,
                      epochs=cfg.trainer.epochs,
                      model_path=cfg.trainer.model_path,
                      checkpoint_path=cfg.trainer.checkpoint_path,
                      device=cfg.trainer.device,
                      device_test=cfg.trainer.device_test)

    # # Comment the following line to disable to async trainer
    trainer = AsyncTrainer(trainer=trainer)

    # Init the loaders
    train_loader, test_loader = init_loaders(seed=cfg.loader.seed,
                                             batch_size=cfg.loader.batch_size,
                                             test_batch_size=cfg.loader.test_batch_size)

    # # Run the rainer
    trainer.run(train_loader=train_loader,
                test_loader=test_loader)
```

# Installing the application
To clone and run this application, you'll need the following installed on your computer:
- [Git](https://git-scm.com)
- Docker Desktop
   - [Install Docker Desktop on Mac](https://docs.docker.com/docker-for-mac/install/)
   - [Install Docker Desktop on Windows](https://docs.docker.com/desktop/install/windows-install/)
   - [Install Docker Desktop on Linux](https://docs.docker.com/desktop/install/linux-install/)
- [Python](https://www.python.org/downloads/)

Install the package:
```bash
# Clone this repository and install the code
git clone https://github.com/zakuro-ai/sakura

# Go into the repository
cd sakura
```

# Makefile commands
Exhaustive list of make commands:
```
build_wheel
install_wheels
build_dockers
sandbox
```
# Environments

## Docker

> **Note**
> 
> Running this application by using Docker is recommended.

Launch a docker image
```
make sandbox
```

## PythonEnv

> **Warning**
> 
> Running this application by using PythonEnv is possible but *not* recommended.
```
sudo apt install libopenmpi-dev && \
    pip install dist/*.whl  --extra-index-url https://download.pytorch.org/whl/cu116 && \
    make install_wheels
```

# Running the application

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