import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as ds
from torchvision.transforms import ToTensor
from model import MnistModule
from trainer import Trainer
from tuners.peft_builder import PeftBuilder
from data_prepper import DataPrepper

batch_size = 64 
device = "mps"
data_prepper = DataPrepper()

train_data_unfiltered = ds.FashionMNIST(
    root="data", 
    train=True, 
    download=True, 
    transform=ToTensor()
)

test_data_unfiltered = ds.FashionMNIST(
    root="data", 
    train=False, 
    download=True, 
    transform=ToTensor()
)

train_dataloader_all = data_prepper.build_dataset(train_data_unfiltered, [])
test_dataloader_all = data_prepper.build_dataset(test_data_unfiltered, [])

train_dataloader_no_7s = data_prepper.build_dataset(train_data_unfiltered, [7])
test_dataloader_no_7s = data_prepper.build_dataset(test_data_unfiltered, [7])

train_dataloader_only_7s = data_prepper.build_dataset(train_data_unfiltered, [0,1,2,3,4,5,6,8,9])

base_model = MnistModule().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(base_model.parameters(), lr=1e-3)

trainer = Trainer(
    device=device, 
    epoch = 6,
    loss_fn=loss_fn, 
    optimizer=optimizer
)

# Build our base model on the whole dataset except 7's.
trainer.train_and_test("Building base model on no 7's", base_model, train_dataloader_no_7s, test_dataloader_no_7s)

peft_builder = PeftBuilder(device)
peft_model = peft_builder.get_model(base_model)
trainer.train_and_test("PEFT fine tuning on only 7's", peft_model, train_dataloader_only_7s, test_dataloader_all)

trainer.train_and_test("Vanilla tuning on only 7's", base_model, train_dataloader_only_7s, test_dataloader_all)
