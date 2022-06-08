from re import I
from IsoNet.models.unet.model import Unet
import torch
import os
import pytorch_lightning as pl  
from .data_sequence import get_datasets
from pytorch_lightning.callbacks import RichProgressBar

def prepare_first_model(settings):
    model = Unet() 
    state = model.state_dict()
    settings.init_model = settings.result_dir + '/model_iter00.h5'
    torch.save(state, settings.init_model)
    return settings

def reload_ckpt(ckpt, model, device=torch.device("cuda:0")):
    if os.path.isfile(ckpt):
        print(f"=> loading checkpoint {ckpt}")
        checkpoint = torch.load(ckpt, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"=> no checkpoint found at '{ckpt}'")

def train_data(settings):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    train_dataset, val_dataset = get_datasets(settings)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True,persistent_workers=True,
                                                num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False,persistent_workers=True,
                                                pin_memory=True, num_workers=4)
    model = Unet()
    reload_ckpt(settings.init_model, model)
    model.train()

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
    trainer.fit(model, train_loader, val_loader)
    state_dict=model.state_dict()
    model_filename = '{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count)
    torch.save(state_dict, model_filename)