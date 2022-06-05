from time import perf_counter
from IsoNet.models.unet.model import Unet
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from torch.nn import L1Loss
from torch.autograd import Variable
import numpy as np


from .utils import AverageMeter, ProgressMeter
from .data_sequence import get_datasets

def prepare_first_model(settings):
    model = Unet() 
    params = model.parameters()
    optimiser = torch.optim.Adam(params, lr=1e-4, weight_decay=0)

    state = dict(
        epoch = 0,
        state_dict = model.state_dict(),
        optimiser = optimiser.state_dict()
    )

    settings.init_model = settings.result_dir + '/model_iter00.h5'
    torch.save(state, settings.init_model)
    return settings

def reload_ckpt_bis(ckpt, model, device=torch.device("cuda:0")):
    if os.path.isfile(ckpt):
        print(f"--> loading checkpoint {ckpt}")
        try:
            checkpoint = torch.load(ckpt, map_location = device)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            return start_epoch
        except RuntimeError:
            model.load_state_dict(ckpt, map_location = 'cpu')
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
    t_writer_1 = SummaryWriter(str(settings.result_dir))
    model_1 = Unet()
    
    #need to check init_model is apporate
    reload_ckpt_bis(settings.init_model, model_1)
    model_1 = model_1.cuda()
    criterion = L1Loss().cuda()
    criterion_val = L1Loss().cuda()
    params = model_1.parameters()
    optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=0)
    train_dataset, val_dataset = get_datasets(settings)

    #need search for drop last
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True,
                                                num_workers=4, pin_memory=False, drop_last = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False,
                                                pin_memory=False, num_workers=8)

    #model_1 = torch.nn.DataParallel(model_1)
    print("Train dataset number of batches", len(train_loader))
    print("Val dataset number of batches", len(val_loader))
    print("start training now")

    for epoch in range(settings.epochs):
        ts = perf_counter()

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data',':6.3f')
        losses = AverageMeter('loss',':.4e')

        batch_per_epoch = len(train_loader)
        progress = ProgressMeter( batch_per_epoch, [batch_time, data_time, losses], prefix=f"train Epoch:[{epoch}]" )

        end = perf_counter()
        
        for i, batch in enumerate(train_loader):
            # empty cache takes long time, if not empty cache the tensor.cuda() takes long time
            torch.cuda.empty_cache()
            inputs_S1, labels_S1 = batch["image"], batch["label"]
            #inputs_S1, labels_S1 = batch[0]["image"].float(), batch[0]["label"].float()
            #inputs_S1, labels_S1 = torch.from_numpy(batch[0]["image"]),torch.from_numpy(batch[0]["label"])

            #inputs_S1, labels_S1 = Variable(inputs_S1), Variable(labels_S1)
            inputs_S1, labels_S1 = inputs_S1.cuda(), labels_S1.cuda()

            data_time.update(perf_counter()-end)

            optimizer.zero_grad()

            segs_S1 = model_1(inputs_S1)

            loss_ = criterion(segs_S1, labels_S1)

            t_writer_1.add_scalar(f"Loss{''}",
                                    loss_.item(),
                                    global_step=batch_per_epoch * epoch + i)

            # measure accuracy and record loss_
            if not np.isnan(loss_.item()):
                losses.update(loss_.item())
            else:
                print("NaN in model loss!!")

            # compute gradient and do SGD step
            loss_.backward()
            optimizer.step()

            t_writer_1.add_scalar("lr", optimizer.param_groups[0]['lr'],
                                    global_step=epoch * batch_per_epoch + i)

            # measure elapsed time
            batch_time.update(perf_counter() - end)
            end = perf_counter()
            # Display progress
            progress.display(i)
        t_writer_1.add_scalar(f"SummaryLoss/train", losses.avg, epoch)

        te = perf_counter()
        print(f"Train Epoch done in {te - ts} s")
        torch.cuda.empty_cache()

        # Validate at the end of epoch every val step
        if (epoch + 1) % 1 == 0:
            validation_loss_1 = step(val_loader, model_1, criterion_val, epoch, t_writer_1)
            t_writer_1.add_scalar(f"SummaryLoss", validation_loss_1, epoch)

            if True:#validation_dice > best_1:
                #save_folder = settings.results_dir
                state = dict(
                        epoch=epoch,
                        state_dict=model_1.module.state_dict(),
                        optimizer=optimizer.state_dict(),
                    )
                best_filename = '{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count)
                torch.save(state, best_filename)

            ts = perf_counter()
            print(f"Val epoch done in {ts - te} s")
            torch.cuda.empty_cache()



def step(data_loader, model, criterion, epoch, writer):
    # Setup
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    mode = "val"
    batch_per_epoch = len(data_loader)
    progress = ProgressMeter(
        batch_per_epoch,
        [batch_time, data_time, losses],
        prefix=f"{mode} Epoch: [{epoch}]")

    end = perf_counter()

    for i, val_data in enumerate(data_loader):
        # measure data loading time
        data_time.update(perf_counter() - end)

        model.eval()
        with torch.no_grad():
            val_inputs, val_labels = (
                val_data["image"].cuda(),
                val_data["label"].cuda(),
            )
            #val_outputs = inference(val_inputs, model)
            val_outputs = model(val_inputs.type(torch.cuda.FloatTensor))

            segs = val_outputs
            targets = val_labels
            loss_ = criterion(segs, targets)

        writer.add_scalar(f"Loss/{mode}{''}",
                          loss_.item(),
                          global_step=batch_per_epoch * epoch + i)

        # measure accuracy and record loss_
        if not np.isnan(loss_.item()):
            losses.update(loss_.item())
        else:
            print("NaN in model loss!!")


        # measure elapsed time
        batch_time.update(perf_counter() - end)
        end = perf_counter()
        # Display progress
        progress.display(i)

    return losses.avg 