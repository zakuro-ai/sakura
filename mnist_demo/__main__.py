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
