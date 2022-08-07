from torchvision import datasets, transforms
import torch
import yaml
import os
from gnutools.fs import load_config as _load_config
from gnutools.utils import RecNamespace


def load_config():
    config_path = f"{os.environ['HOME']}/.config/sakura/config.yml"
    cfg = _load_config(config_path).sakura
    return RecNamespace(cfg)


def init_loaders(seed, batch_size, test_batch_size):
    # Instantiate
    torch.manual_seed(seed)
    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1, dataset2 = (
        datasets.MNIST('data',
                       train=True,
                       download=True,
                       transform=transform),
        datasets.MNIST('data',
                       train=False,
                       transform=transform)
    )
    train_loader, test_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs), \
        torch.utils.data.DataLoader(dataset2, **test_kwargs)
    return train_loader, test_loader


if __name__ == "__main__":
    from mnist_demo import cfg

    # Init the loaders
    train_loader, test_loader = init_loaders(seed=cfg.loader.seed,
                                             batch_size=cfg.loader.batch_size,
                                             test_batch_size=cfg.loader.test_batch_size)
