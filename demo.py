from __future__ import print_function
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch
from sakura.ml import AsyncTrainer, DefaultTrainer
from sakura.ml.decorators import synchronize


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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


class Trainer(DefaultTrainer):
    def __init__(self,
                 model,
                 epochs,
                 train_loader,
                 test_loader,
                 optimizer,
                 scheduler,
                 model_path="mnist_cnn.pt",
                 checkpoint_path="mnist_cnn.ckpt.pt",
                 *args,
                 **kwargs):
        super(Trainer, self).__init__(epochs,
                                      optimizer,
                                      scheduler,
                                      model,
                                      model_path,
                                      checkpoint_path,
                                      train_loader,
                                      test_loader,
                                      *args,
                                      **kwargs)
        self._model = model
        self._epochs = epochs
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._model_path = model_path
        self._checkpoint_path = checkpoint_path
        self._avg_loss = []
        self._correct = 0
        self._loss = 0
        self._epoch_start = 1

    @synchronize
    def train(self):
        self._model.train()
        self._avg_loss = []
        self._correct=0
        for batch_idx, (data, target) in tqdm(
                enumerate(self._train_loader),
                total=len(self._train_loader),
                desc=self.description()):
            data, target = data.to(self._device), target.to(self._device)
            self._optimizer.zero_grad()
            output = self._model(data)
            loss= F.nll_loss(output, target)
            loss.backward()
            self._avg_loss.append(loss.item())
            self._optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            self._correct += pred.eq(target.view_as(pred)).sum().item()

    @synchronize
    def test(self):
        self._correct = 0
        self._loss = 0
        # Test
        self._model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self._test_loader):
                data, target = data.to(self._device), target.to(self._device)
                output = self._model(data)
                self._loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                self._correct += pred.eq(target.view_as(pred)).sum().item()

    def run(self, *args, **kwargs):
        for self.state.shared.epoch.current in range(self._epoch_start, self._epochs):
            if self._mode == "train":
                self.train()
                self._scheduler.step()
            else:
                self.test()


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()


    # Instantiate
    torch.manual_seed(args.seed)
    use_cuda = "cuda" # use_cuda = not args.no_cuda and torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)

    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1, dataset2 = datasets.MNIST('data',
                                        train=True,
                                        download=True,
                                        transform=transform), \
                         datasets.MNIST('data',
                                        train=False,
                                        transform=transform)
    train_loader, test_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs), \
                                torch.utils.data.DataLoader(dataset2, **test_kwargs)
    epochs = args.epochs
    # Instantiate
    model = Net()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    trainer = AsyncTrainer(cls=Trainer, device="cuda")
    trainer.run(model,
                epochs,
                train_loader,
                test_loader,
                optimizer,
                scheduler,
                "mnist_cnn.pt",
                "mnist_cnn.ckpt.pt")

