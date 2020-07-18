from keras.utils import Sequence
import numpy as np
import mrcfile
import os
# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class dataSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.perm = np.random.permutation(len(self.x))

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.x))

    def __getitem__(self, i):
        idx = slice(i*self.batch_size,(i+1)*self.batch_size)
        idx = self.perm[idx]
        rx = np.array([mrcfile.open(self.x[j]).data[:,:,:,np.newaxis] for j in idx])
        ry = np.array([mrcfile.open(self.y[j]).data[:,:,:] for j in idx])
        sp=ry.shape
        ry_final=np.zeros((sp[0],sp[1],sp[2],sp[3],21))
        for i in range(0,21):
            ry_final[:,:,:,:,i] = (ry==i).astype(int)
        #    print(i,np.count_nonzero((ry==i).astype(int)))
        #print(ry_final[0,:,:,:,10].shape)
        #with mrcfile.new("1.mrc", overwrite=True) as output_mrc:
        #    output_mrc.set_data(ry_final[0,:,:,:,10].astype(np.float32))

        #imsave("1.tif",ry_final[0,:,:,:,10].astype(np.uint8))
        return rx,ry_final

def prepare_dataseq(data_folder, batch_size):

    dirs_tomake = ['train_x','train_y', 'test_x', 'test_y']
    path_all = []
    for d in dirs_tomake:
        p = '{}/{}/'.format(data_folder, d)
        path_all.append(sorted([p+f for f in os.listdir(p)]))
    train_data = dataSequence(path_all[0], path_all[1], batch_size)
    test_data = dataSequence(path_all[2], path_all[3], batch_size)
    return train_data, test_data
