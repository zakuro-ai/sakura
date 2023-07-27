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
