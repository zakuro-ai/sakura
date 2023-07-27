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
        "sakura.lightning",
    ],
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
import os
import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import argparse
from sakura.lightning import SakuraTrainer


class MNISTModel(L.LightningModule):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def validation_step(self, batch, batch_nb):
        with torch.no_grad():
            x, y = batch
            loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


if __name__ == "__main__":
    PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
    BATCH_SIZE = 2000 if torch.cuda.is_available() else 64
    # Init our model
    mnist_model = MNISTModel()

    # Init DataLoader from MNIST Dataset
    train_ds = MNIST(
        PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor()
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

    # Init DataLoader from MNIST Dataset
    val_ds = MNIST(
        PATH_DATASETS, train=False, download=True, transform=transforms.ToTensor()
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    trainer = SakuraTrainer(
        accelerator="auto",
        max_epochs=10,
    )

    trainer.run(
        mnist_model, train_loader, val_loader, model_path="models/best_model.pth"
    )

```

# Installing the application
To clone and run this application, you'll need the following installed on your computer:
- [Git](https://git-scm.com)
- Docker Desktop
   - [Install Docker Desktop on Mac](https://docs.docker.com/docker-for-mac/install/)
   - [Install Docker Desktop on Windows](https://docs.docker.com/desktop/install/windows-install/)
   - [Install Docker Desktop on Linux](https://docs.docker.com/desktop/install/linux-install/)
- [Python](https://www.python.org/downloads/)

### Clone the code and install the binary
```bash
# Clone this repository and install the code
git clone https://github.com/zakuro-ai/sakura

# Go into the repository
cd sakura

# Install sakura
curl https://get.zakuro.ai/sakura/install | sh
```

### Check that the binary has been downloaded
```bash
PATH=$PATH:~/.zakuro/bin which sakura
```

# Running the application

```bash
PATH=$PATH:~/.zakuro/bin sakura main.py
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