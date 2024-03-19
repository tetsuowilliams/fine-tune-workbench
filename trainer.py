import torch
from torch.utils.data import DataLoader
import torchvision.datasets as ds
from torchvision.transforms import ToTensor
from model import MnistModule

class Trainer:
    def __init__(self, device: str, 
                 model: torch.nn.Module, 
                 train_dataloader: DataLoader, 
                 test_dataloader: DataLoader) -> None:
        self.device = device
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def train_for(self, epoch: int, loss_fn: torch.nn.CrossEntropyLoss, optimizer: torch.optim.SGD):
        for i in range(epoch):
            print(f"Epoch: {i+1} ===================")
            self.train(self.train_dataloader, self.model, loss_fn, optimizer)
            self.test(self.test_dataloader, self.model, loss_fn)

    def train(self, dataloader, model, loss_fn, optimizer):
        model.train()
        size = len(dataloader.dataset)

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            prediction = model(X)
            loss = loss_fn(prediction, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    def test(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()

        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size

        print(f"Test error: \n Accuracy: {(100*correct):>0.1f}% Avg loss {test_loss:>8f} \n")