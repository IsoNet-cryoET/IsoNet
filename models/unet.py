import torch
import torch.nn as nn
import pytorch_lightning as pl
import logging
class ConvBlock(pl.LightningModule):
    # conv_per_depth fixed to 2
    def __init__(self, in_channels, out_channels, n_conv = 2, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding), 
            nn.BatchNorm3d(num_features=out_channels),
            #nn.InstanceNorm3d(num_features = out_channels),
            nn.LeakyReLU()
        ]
        for _ in range(max(n_conv-1,0)):
            layers.append(nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding))
            layers.append(nn.BatchNorm3d(num_features=out_channels))
            #layers.append(nn.InstanceNorm3d(num_features=out_channels))
            layers.append(nn.LeakyReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class EncoderBlock(pl.LightningModule):
    def __init__(self, filter_base, unet_depth = 3, n_conv = 2):
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
    def __init__(self, filter_base, unet_depth = 3, n_conv=2):
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
        
        self.module_dict["final"] = nn.Conv3d(in_channels=filter_base[0], out_channels=1, kernel_size=1, stride=1, padding=0 )
    
    def forward(self, x, down_sampling_features):
        for k, op in self.module_dict.items():
            x=op(x)
            if k.startswith("deconv"):
                x = torch.cat((down_sampling_features[int(k[-1])], x), dim=1)
        return x

class Unet(pl.LightningModule):
    def __init__(self):
        super(Unet, self).__init__()
        filter_base = [64,128,256,320,320,320]
        #filter_base = [32,64,128,256,320,320]
        #filter_base = [1,1,1,1,1]
        unet_depth =3
        n_conv = 2
        self.encoder = EncoderBlock(filter_base=filter_base, unet_depth=unet_depth, n_conv=n_conv)
        self.decoder = DecoderBlock(filter_base=filter_base, unet_depth=unet_depth, n_conv=n_conv)
    
    def forward(self, x):
        x, down_sampling_features = self.encoder(x)
        x = self.decoder(x, down_sampling_features)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.L1Loss()(self(x), y)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer 

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.L1Loss()(y_hat, y)
        return loss

    '''
    def training_epoch_end(self, outputs):
        #print(outputs)
        #loss = torch.stack([x['loss'] for x in outputs]).mean()
        #avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        #self.trainer.progress_bar_callback.main_progress_bar.write(
        #    f"test")        
        print("\n")
        #loss = torch.stack(outputs).mean()
        #logger = logging.getLogger('lightning')
        #logger.info(f"Training loss: {loss} \n")
        #self.trainer.progress_bar_callback.main_progress_bar.write(
        #    f"Epoch {self.trainer.current_epoch} validation loss={loss.item()}")
        
        #self.trainer.progress_bar_callback.main_progress_bar.refresh()

    def validation_epoch_end(self, outputs):
        #avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        #self.trainer.progress_bar_callback.main_progress_bar.write(
        #    f"test")        
        #rint("asdfasfasf")
        loss = torch.stack(outputs).mean()
        logger = logging.getLogger('lightning')
        logger.info(f"Validation loss: {loss} \n")
        #self.trainer.progress_bar_callback.main_progress_bar.write(
        #    f"Epoch {self.trainer.current_epoch} validation loss={loss.item()}")
        
        #self.trainer.progress_bar_callback.main_progress_bar.refresh()
        '''