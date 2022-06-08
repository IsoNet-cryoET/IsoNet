import torch
import torch.nn as nn
import pytorch_lightning as pl
import logging
class ConvBlock(pl.LightningModule):
    # conv_per_depth fixed to 2
    def __init__(self, in_channels, out_channels, n_conv = 2, kernal_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernal_size, stride=stride, padding=padding), 
            nn.BatchNorm3d(num_features=out_channels),
            nn.LeakyReLU()
        ]
        for _ in range(n_conv-1):
            layers.append(nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                    kernel_size=kernal_size, stride=stride, padding=padding))
            layers.append(nn.BatchNorm3d(num_features=out_channels))
            layers.append(nn.LeakyReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class EncoderBlock(pl.LightningModule):
    def __init__(self, filter_base, unet_depth = 3):
        super(EncoderBlock, self).__init__()
        self.module_dict = nn.ModuleDict()
        self.module_dict['first_conv'] = nn.Conv3d(in_channels=1, out_channels=filter_base[0], kernel_size=3, stride=1, padding=1)

        for n in range(unet_depth):
            self.module_dict["conv_stack_{}".format(n)] = ConvBlock(in_channels=filter_base[n], out_channels=filter_base[n])
            self.module_dict["stride_conv_{}".format(n)] = nn.Conv3d(in_channels=filter_base[n], out_channels=filter_base[n+1], kernel_size=2, stride=2, padding=0)
            self.module_dict["activation_{}".format(n)] = nn.LeakyReLU()
        
        self.module_dict["bottleneck"] = ConvBlock(in_channels=filter_base[n+1], out_channels=filter_base[n+1])
    
    def forward(self, x):
        down_sampling_features = []
        for k, op in self.module_dict.items():
            x = op(x)
            if k.startswith('conv'):
                down_sampling_features.append(x)
        return x, down_sampling_features

class DecoderBlock(pl.LightningModule):
    def __init__(self, filter_base, unet_depth = 3):
        super(DecoderBlock, self).__init__()
        self.module_dict = nn.ModuleDict()
        for n in reversed(range(unet_depth)):
            self.module_dict["deconv_{}".format(n)] = nn.ConvTranspose3d(in_channels=filter_base[n+1],
                                                                         out_channels=filter_base[n],
                                                                         kernel_size=2,
                                                                         stride=2,
                                                                         padding=0)
            self.module_dict["activation_{}".format(n)] = nn.LeakyReLU()
            self.module_dict["conv_stack_{}".format(n)] = ConvBlock(filter_base[n]*2, filter_base[n])
        
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
        #filter_base = [64,128,256,320,320]
        filter_base = [32,64,128,256,320]
        #filter_base = [1,1,1,1,1]
        self.encoder = EncoderBlock(filter_base=filter_base)
        self.decoder = DecoderBlock(filter_base=filter_base)
    
    def forward(self, x):
        x, down_sampling_features = self.encoder(x)
        x = self.decoder(x, down_sampling_features)
        return x

class Discriminator(pl.LightningModule):
    def __init__(self, filter_base = [32,64,128,256]):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(

            nn.Conv3d(1, filter_base[0], kernel_size=2, stride=2, padding=0), #64->32
            nn.LeakyReLU(),
            ConvBlock(filter_base[0],filter_base[1]),

            nn.Conv3d(filter_base[1], filter_base[1], kernel_size=2, stride=2, padding=0), #32->16
            nn.LeakyReLU(),
            ConvBlock(filter_base[1],filter_base[2]),

            nn.Conv3d(filter_base[2], filter_base[2], kernel_size=2, stride=2, padding=0), #16->8
            nn.LeakyReLU(),
            ConvBlock(filter_base[2],filter_base[3]),

            nn.Conv3d(filter_base[3], filter_base[3], kernel_size=2, stride=2, padding=0), #8->4
            nn.LeakyReLU(),

            nn.Flatten(),
            nn.Linear(filter_base[3]*64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        validity = self.model(x)
        return validity

class Context_encoder(pl.LightningModule):
    def __init__(self):
        super(Context_encoder, self).__init__()
        self.generator = Unet()
        self.discriminator = Discriminator()

    def forward(self, x):
        return self.generator(x)

    #def adv_loss(self, y_hat, y):
    #    return torch.binary_cross_entropy_with_logits(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        criterionMAE = nn.L1Loss()
        #remove sigmoid
        #criterion = nn.BCEWithLogitsLoss()
        criterion = nn.L1Loss()

        x,y = batch
        ones = torch.ones(x.size(0), 1)
        ones = ones.type_as(x)

        zeros = torch.zeros(x.size(0),1)
        zeros = zeros.type_as(x)

        y_pred = self(x)
        y_pred_disc = self.discriminator(y_pred)
        
        if optimizer_idx == 0:
            #discriminator
            #print("dis")
            y_disc = self.discriminator(y)
            #print(y_disc.size())
            #print(y_pred_disc.size())
            #print(ones.size())
            #print(zeros.size())
            loss = criterion(y_disc, ones) + criterion(y_pred_disc, zeros)
            loss = loss*0.5
            self.log("d_loss", loss, prog_bar=True)
            return loss

        if optimizer_idx == 1:
            #generator
            #print("gen")
            mae_loss = criterionMAE(y_pred,y)
            adv_loss = criterion(y_pred_disc, ones)
            loss = mae_loss * 0.95 + adv_loss * 0.05
            self.log("g_loss", loss, prog_bar=True)
            self.log("mae", mae_loss, prog_bar=True)
            return loss

        return loss

    
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=1e-2)
        return [opt_d, opt_g], []
        #return opt_g

    '''
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = L1Loss()(y_hat, y)
        return loss
        
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