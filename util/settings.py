
class Settings3D():
    def __init__(self):
        # computation settings
        self.gpuID = "0,1,2,3"

        # file settings
        import os
        directory = 'particles-bin4'
        self.mrc_list = os.listdir(directory)
        self.mrc_list = ['{}/{}'.format(directory,i) for i in self.mrc_list]
        #self.mrc_list = ['pp109_bin2.rec', 'pp127_bin2.rec']

        #preprocessing settings
        self.do_preprocess = True
        self.mask = None#'mask.tif'
        self.preprocessing_ncpus = 40

        self.cube_sidelen = 96
        self.cropsize = 128
        self.ncube = 1

        #self.cubes_file = 'pp676_2d_patches.npz'


        # train settings
        self.do_training = True
        self.epochs = 10
        self.batch_size = 8
        self.steps_per_epoch = 100

        self.drop_out = 0.1
        self.convs_per_depth = 3
        self.kernel = (3,3,3)
        self.unet_depth = 4

        self.continue_training =  True
        self.continue_training_iter = 37
        self.reload_weight = True



        #predict settings
        self.do_predict = True

        self.predict_cubesize = 96
        self.predict_cropsize = 128

        self.predict_batch_size = 4
        '''
        if crop size is 128 for 1080ti bin2 max batch size is 2
        if cube size 64 for 1080ti max batch size is 20
        '''

        self.iterations = 100








        #self.ncube = 640
        #self.cube_sidelen = 64
        #self.npatchesper = 30
        #self.patches_sidelen =64
        #self.rotate = True
