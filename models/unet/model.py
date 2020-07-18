from models.unet import builder

def Unet(filter_base=32,
        depth=2,
        convs_per_depth=2,
        kernel=(3,3),
        batch_norm=False,
        dropout=0.0,
        pool=(2,2)):
    model = builder.build_unet(filter_base,depth,convs_per_depth,
               kernel,
               batch_norm,
               dropout,
               pool)
    return model