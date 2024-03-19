import torch

class MnistModule(torch.nn.Module):
    def __init__(self ) -> None:
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(28*28, 1512),
            torch.nn.ReLU(),
            torch.nn.Linear(1512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.seq(x)
        return logits
