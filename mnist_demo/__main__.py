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
