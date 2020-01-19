class Setting():
    def __init__(self):


        # train settings
        self.epoch = 40
        self.batch_size = 256
        self.steps_per_epoch = 70
        # prepare settings
        self.ncube = 640
        self.cube_sidelen = 64
        self.npatchesper = 30
        self.patches_sidelen =64
        self.rotate = True
