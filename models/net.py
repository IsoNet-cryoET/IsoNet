from IsoNet.models.unet.model import Unet
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.callbacks import RichProgressBar
from .data_sequence import get_datasets

class Net:
    def __init__(self, settings = None):
        self.model = Unet()

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint)
    
    def save(self, path):
        state = self.model.state_dict()
        torch.save(state, path)

    def train(self, data_path):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

        train_dataset, val_dataset = get_datasets(data_path)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True,persistent_workers=True,
                                                num_workers=4, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False,persistent_workers=True,
                                                pin_memory=True, num_workers=4)
        self.model.train()
        trainer = pl.Trainer(
            gpus=[0,1,2,3],
            max_epochs=3,
            strategy = 'dp',
            #enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
            callbacks=RichProgressBar(),
            num_sanity_val_steps=0
        )
        trainer.fit(self.model, train_loader, val_loader)
