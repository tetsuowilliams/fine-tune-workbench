import torch
from torch.utils.data import DataLoader
import torchvision.datasets as ds
from torchvision.transforms import ToTensor
from model import MnistModule
from trainer import Trainer


train_data = ds.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = ds.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

batch_size = 64 
device = "mps"

train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

model = MnistModule().to(device)
trainer = Trainer(device, model, train_dataloader, test_dataloader)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD( model.parameters(), lr=1e-3)
trainer.train_for(6, loss_fn, optimizer)

