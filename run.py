import torch
from torch.utils.data import DataLoader
import torchvision.datasets as ds
from torchvision.transforms import ToTensor
from model import MnistModule
from trainer import Trainer
from tuners.peft_tuner import PEFTTuner

train_data = ds.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = ds.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

batch_size = 64 
device = "mps"

model = MnistModule().to(device)
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD( model.parameters(), lr=1e-3)

trainer = Trainer(
    device=device, 
    model = model, 
    train_dataloader = train_dataloader, 
    test_dataloader=test_dataloader, 
    epoch = 6, 
    loss_fn=loss_fn, 
    optimizer=optimizer
)

#trainer.train_and_test()

peft_tuner = PEFTTuner(device)
peft_tuner.train_model(trainer)
