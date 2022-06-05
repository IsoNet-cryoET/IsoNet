from turtle import forward
import torch
import torch.nn as nn
class ConvBlock(nn.Module):
    # conv_per_depth fixed to 2
    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernal_size, stride=stride, padding=padding), 
            nn.BatchNorm3d(num_features=out_channels),
            nn.LeakyReLU()
        )
        self.net2 = nn.Sequential(
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                    kernel_size=kernal_size, stride=stride, padding=padding), 
            nn.BatchNorm3d(num_features=out_channels),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.net2(self.net1(x))

class EncoderBlock(nn.Module):
    def __init__(self, filter_base, unet_depth = 3):
        super(EncoderBlock, self).__init__()
        self.module_dict = nn.ModuleDict()
        self.module_dict['first_conv'] = nn.Conv3d(in_channels=1, out_channels=filter_base[0], kernel_size=3, stride=1, padding=1)

        for n in range(unet_depth):
            self.module_dict["conv_stack_{}".format(n)] = ConvBlock(in_channels=filter_base[n], out_channels=filter_base[n])
            self.module_dict["stride_conv_{}".format(n)] = nn.Conv3d(in_channels=filter_base[n], out_channels=filter_base[n+1], kernel_size=2, stride=2, padding=0)
            #maybe need activation here
        
        self.module_dict["bottleneck"] = ConvBlock(in_channels=filter_base[n+1], out_channels=filter_base[n+1])
    
    def forward(self, x):
        down_sampling_features = []
        for k, op in self.module_dict.items():
            x = op(x)
            if k.startswith('conv'):
                down_sampling_features.append(x)
        return x, down_sampling_features

class DecoderBlock(nn.Module):
    def __init__(self, filter_base, unet_depth = 3):
        super(DecoderBlock, self).__init__()
        self.module_dict = nn.ModuleDict()
        for n in reversed(range(unet_depth)):
            self.module_dict["deconv_{}".format(n)] = nn.ConvTranspose3d(in_channels=filter_base[n+1],
                                                                         out_channels=filter_base[n],
                                                                         kernel_size=2,
                                                                         stride=2,
                                                                         padding=0)
            ## maybe need activation here
            self.module_dict["conv_stack_{}".format(n)] = ConvBlock(filter_base[n]*2, filter_base[n])
        
        self.module_dict["final"] = nn.Conv3d(in_channels=filter_base[0], out_channels=1, kernel_size=1, stride=1, padding=0 )
    
    def forward(self, x, down_sampling_features):
        for k, op in self.module_dict.items():
            x=op(x)
            if k.startswith("deconv"):
                x = torch.cat((down_sampling_features[int(k[-1])], x), dim=1)
        return x


class Unet(nn.Module):
    def __init__(self):
        #filter_base = [64,128,256,320,320]
        #filter_base = [32,64,128,256,320]
        filter_base = [1,1,1,1,1]
        super(Unet, self).__init__()
        self.encoder = EncoderBlock(filter_base=filter_base)
        self.decoder = DecoderBlock(filter_base=filter_base)
    
    def forward(self, x):
        x, down_sampling_features = self.encoder(x)
        x = self.decoder(x, down_sampling_features)
        return x
                