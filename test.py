import os

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = torch.cuda.device_count()
print(AVAIL_GPUS)
BATCH_SIZE = 256 if AVAIL_GPUS else 64

class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


mnist_model = MNISTModel()

# Init DataLoader from MNIST Dataset
train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=4)
#def run():
#    torch.multiprocessing.freeze_support()
#    print('loop')

#if __name__ == '__main__':
#    run()
# Initialize a trainer
trainer = Trainer(
    gpus=2,    
    max_epochs=10,
    strategy='dp'
)

# Train the model ⚡
trainer.fit(mnist_model, train_loader)