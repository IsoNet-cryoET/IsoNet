from lib2to3.pytree import Leaf
from re import I
from typing import List
import torch
import torch.nn as nn
import pytorch_lightning as pl
import logging
class ConvBlock(pl.LightningModule):
    # conv_per_depth fixed to 2
    def __init__(self, in_channels, out_channels, n_conv, kernel_size =3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding, bias=False), 
            nn.BatchNorm3d(num_features=out_channels),
            #nn.InstanceNorm3d(num_features = out_channels),
            nn.LeakyReLU()
        ]
        for _ in range(max(n_conv-1,0)):
            layers.append(nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
            layers.append(nn.BatchNorm3d(num_features=out_channels))
            #layers.append(nn.InstanceNorm3d(num_features=out_channels))
            layers.append(nn.LeakyReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class EncoderBlock(pl.LightningModule):
    def __init__(self, filter_base, unet_depth, n_conv):
        super(EncoderBlock, self).__init__()
        self.module_dict = nn.ModuleDict()
        self.module_dict['first_conv'] = nn.Conv3d(in_channels=1, out_channels=filter_base[0], kernel_size=3, stride=1, padding=1)

        for n in range(unet_depth):
            self.module_dict["conv_stack_{}".format(n)] = ConvBlock(in_channels=filter_base[n], out_channels=filter_base[n], n_conv=n_conv)
            self.module_dict["stride_conv_{}".format(n)] = ConvBlock(in_channels=filter_base[n], out_channels=filter_base[n+1], n_conv=1, kernel_size=2, stride=2, padding=0)
        
        self.module_dict["bottleneck"] = ConvBlock(in_channels=filter_base[n+1], out_channels=filter_base[n+1], n_conv=n_conv-1)
    
    def forward(self, x):
        down_sampling_features = []
        for k, op in self.module_dict.items():
            x = op(x)
            if k.startswith('conv'):
                down_sampling_features.append(x)
        return x, down_sampling_features

class DecoderBlock(pl.LightningModule):
    def __init__(self, filter_base, unet_depth, n_conv):
        super(DecoderBlock, self).__init__()
        self.module_dict = nn.ModuleDict()
        for n in reversed(range(unet_depth)):
            self.module_dict["deconv_{}".format(n)] = nn.ConvTranspose3d(in_channels=filter_base[n+1],
                                                                         out_channels=filter_base[n],
                                                                         kernel_size=2,
                                                                         stride=2,
                                                                         padding=0)
            self.module_dict["activation_{}".format(n)] = nn.LeakyReLU()
            self.module_dict["conv_stack_{}".format(n)] = ConvBlock(filter_base[n]*2, filter_base[n],n_conv=n_conv)
        
    def forward(self, x,
        down_sampling_features: List[torch.Tensor]):
        for k, op in self.module_dict.items():
            x=op(x)
            if k.startswith("deconv"):
                x = torch.cat((down_sampling_features[int(k[-1])], x), dim=1)
        return x

class Unet(pl.LightningModule):
    def __init__(self,learning_rate=3e-4):
        super(Unet, self).__init__()
        filter_base = [64,128,256,320,320,320]
        #filter_base = [32,64,128,256,320,320]
        #filter_base = [1,1,1,1,1]
        unet_depth = 3
        n_conv = 3
        self.encoder = EncoderBlock(filter_base=filter_base, unet_depth=unet_depth, n_conv=n_conv)
        self.decoder = DecoderBlock(filter_base=filter_base, unet_depth=unet_depth, n_conv=n_conv)
        self.final = nn.Conv3d(in_channels=filter_base[0], out_channels=1, kernel_size=3, stride=1, padding=1)
        self.mse_layer = nn.Sequential(
            nn.Conv3d(in_channels=filter_base[0], out_channels=filter_base[0]//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=filter_base[0]//2, out_channels=filter_base[0]//4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=filter_base[0]//4, out_channels=filter_base[0]//8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=filter_base[0]//8, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Softplus()
        )
        self.variance_out = False
        self.learning_rate = learning_rate
        self.val_loss = 0
    
    def forward(self, x):
        if self.variance_out:
            with torch.no_grad():
                x, down_sampling_features = self.encoder(x)
                x = self.decoder(x, down_sampling_features)
                y_hat = self.final(x)
            mse_map = self.mse_layer(x) + 10**-3
            return [y_hat,mse_map]
        else:
            x, down_sampling_features = self.encoder(x)
            x = self.decoder(x, down_sampling_features)
            y_hat = self.final(x)
            return y_hat
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        if self.variance_out:
            #loss = nn.L1Loss()(out[1], torch.abs(out[0]-y))
            c = 0.6931471805599453 # log(2)
            loss = torch.mean(torch.div(torch.abs(out[0]-y), out[1]) + torch.log(out[1]) + c)
        else:
            loss = nn.L1Loss()(out, y)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer 

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch
            out = self(x)
            if self.variance_out:
                #loss = nn.L1Loss()(out[1], torch.abs(out[0]-y))
                c = 0.6931471805599453 # log(2)
                loss = torch.mean(torch.div(torch.abs(out[0]-y), out[1]) + torch.log(out[1]) + c)
            else:
                loss = nn.L1Loss()(out, y)
            return loss

    def training_epoch_end(self, outputs):
        #print(outputs)
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        #avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        #self.trainer.progress_bar_callback.main_progress_bar.write(
        #    f"test")        
        #loss = torch.stack(outputs).mean()
        pb = self.trainer.progress_bar_callback.main_progress_bar
        print(pb.format_dict["elapsed"])
        pb.write(
            "Epoch {}: train_loss:{:.5f}, val_loss:{:.5f}".format(self.trainer.current_epoch,loss, self.val_loss))
        #self.trainer.progress_bar_callback.main_progress_bar.set_description(
        #    f"Epoch {self.trainer.current_epoch} Training loss:{loss}")
        #self.trainer.progress_bar_callback.main_progress_bar.refresh()
    #    logger = logging.getLogger('lightning')
    #    logger.info(f"Training loss: {loss} \n")
        #self.trainer.progress_bar_callback.main_progress_bar.write(
        #    f"Epoch {self.trainer.current_epoch} validation loss={loss.item()}")
    #    self.log("my_loss", loss, prog_bar=True)
    #    pass        
        #self.trainer.progress_bar_callback.main_progress_bar.refresh()

    def validation_epoch_end(self, outputs):
        #avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        #self.trainer.progress_bar_callback.main_progress_bar.write(
        #    f"test")        
        loss = torch.stack(outputs).mean()
        self.val_loss = loss
        #log = {'val_loss': avg_loss}
        #self.log("\n val_loss", avg_loss)
#        print("\n")
        #print(outputs)
        #loss = torch.stack(outputs).mean()
    #    logger = logging.getLogger('lightning')
    #    logger.info(f"Validation loss: {loss} \n")
        #print("\n")
        #self.log("Validation_loss asdfasf", loss, prog_bar=True, on_step=False, on_epoch=True)
    #    self.log("Validation Loss:",loss,prog_bar=True)
        #self.trainer.progress_bar_callback.main_progress_bar.write(
        #    f"Epoch {self.trainer.current_epoch} validation loss={loss.item()}")
        
