from IsoNet.models.unet.model import Unet, Context_encoder
import torch
import os
import pytorch_lightning as pl  
from .data_sequence import get_datasets
from pytorch_lightning.loggers import TensorBoardLogger
from .keras_progressbar import KerasProgressBar

def prepare_first_model(settings):
    model = Context_encoder() 

    state = dict(
        epoch = 0,
        state_dict = model.state_dict(),
    )

    settings.init_model = settings.result_dir + '/model_iter00.h5'
    torch.save(state, settings.init_model)
    return settings

def reload_ckpt_bis(ckpt, model, device=torch.device("cuda:0")):
    if os.path.isfile(ckpt):
        print(f"--> loading checkpoint {ckpt}")
        #try:
        checkpoint = torch.load(ckpt, map_location = device)
        #start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        return 0#start_epoch
        #except RuntimeError:
        #    model.load_state_dict(ckpt, map_location = 'cpu')
    else:
        raise ValueError(f"--> no checkpoint found at {ckpt}")

def reload_ckpt(ckpt, model, device=torch.device("cuda:0")):
    if os.path.isfile(ckpt):
        print(f"=> loading checkpoint {ckpt}")
        
        checkpoint = torch.load(ckpt, map_location=device)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        return start_epoch

    else:
        raise ValueError(f"=> no checkpoint found at '{ckpt}'")

def train_data(settings):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    model_1 = Context_encoder()
    #model_1.cuda()
    #model_1 = torch.nn.parallel.DataParallel(model_1, find_unused_parameters=False)
    
    #need to check init_model is apporate
    reload_ckpt_bis(settings.init_model, model_1)
    #model_1 = model_1.cuda()
    train_dataset, val_dataset = get_datasets(settings)

    #need search for drop last
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True,persistent_workers=True,
                                                num_workers=4, pin_memory=True, drop_last = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False,persistent_workers=True,
                                                pin_memory=True, num_workers=4)

    #model_1 = torch.nn.DataParallel(model_1)
    print("Train dataset number of batches", len(train_loader))
    print("Val dataset number of batches", len(val_loader))
    print("start training now")

    from pytorch_lightning.callbacks import RichProgressBar
    trainer = pl.Trainer(
        gpus=[0,1,2,3],
        max_epochs=3,
        strategy = 'dp',
        #enable_progress_bar=False,
        logger=False,
        #enable_checkpointing=False,
        #callbacks=RichProgressBar(),
        num_sanity_val_steps=0
    )
    trainer.fit(model_1, train_loader, val_loader)
    state = dict(
        epoch=1,
        state_dict=model_1.state_dict(),
    )
    best_filename = '{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count)
    torch.save(state, best_filename)