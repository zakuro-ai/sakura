from torchvision import datasets, transforms
import torch
import yaml
from sakura import RecNamespace
import os


def load_config():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    return RecNamespace(yaml.load(open(f"{cur_dir}/config.yml"), Loader=yaml.FullLoader)["sakura"])


def init_loaders(seed, batch_size, test_batch_size, device):
    # Instantiate
    torch.manual_seed(seed)
    train_kwargs = {'batch_size':batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if device=="cuda":
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
                                        train=False,
                                        download=True,
                                        transform=transform), \
                         datasets.MNIST('data',
                                        train=False,
                                        transform=transform)
    train_loader, test_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs), \
                                torch.utils.data.DataLoader(dataset2, **test_kwargs)
    return train_loader, test_loader