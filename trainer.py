import torch
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, device: str, 
                 epoch: int, 
                 loss_fn: torch.nn.CrossEntropyLoss, 
                 optimizer: torch.optim.SGD) -> None:
        self.device = device
        self.epoch = epoch
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_and_test(self, description:str, 
                       model: torch.nn.Module, 
                       train_dl: DataLoader, 
                       test_dl: DataLoader):
        print(description)

        for i in range(self.epoch):
            print(f"Epoch: {i+1} ===================")
            self.train(model, train_dl)
            self.test(model, test_dl)

    def train(self, model: torch.nn.Module, train_dl: DataLoader):
        model.train()
        size = len(train_dl.dataset)
        
        for batch, (X, y) in enumerate(train_dl):
            X, y = X.to(self.device), y.to(self.device)

            prediction = model(X)
            loss = self.loss_fn(prediction, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    def test(self, model: torch.nn.Module, test_dl: DataLoader):
        size = len(test_dl.dataset)
        num_batches = len(test_dl)
        model.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in test_dl:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size

        print(f"Test error: \n Accuracy: {(100*correct):>0.1f}% Avg loss {test_loss:>8f} \n")
