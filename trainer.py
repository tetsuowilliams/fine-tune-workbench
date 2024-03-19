import torch
from torch.utils.data import DataLoader
import torchvision.datasets as ds
from torchvision.transforms import ToTensor
from model import MnistModule

class Trainer:
    def __init__(self, device: str, 
                 model: torch.nn.Module, 
                 train_dataloader: DataLoader, 
                 test_dataloader: DataLoader, 
                 epoch: int, 
                 loss_fn: torch.nn.CrossEntropyLoss, 
                 optimizer: torch.optim.SGD) -> None:
        self.device = device
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.epoch = epoch
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_and_test(self):
        for i in range(self.epoch):
            print(f"Epoch: {i+1} ===================")
            self.train()
            self.test()

    def train(self):
        self.model.train()
        size = len(self.train_dataloader.dataset)

        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)

            prediction = self.model(X)
            loss = self.loss_fn(prediction, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    def test(self):
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        self.model.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size

        print(f"Test error: \n Accuracy: {(100*correct):>0.1f}% Avg loss {test_loss:>8f} \n")