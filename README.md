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
If you worked with PyTorch in your project your would find a common structure. Simply change the `test` and `train` in your trainer as shown in the demo file. 
```python
class Trainer(DefaultTrainer):
   ...
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