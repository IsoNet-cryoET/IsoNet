import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import mrcfile

class Brats(Dataset):
    def __init__(self, data_dir, prefix = "train"):
        super(Brats, self).__init__()
        self.path_all = []
        for d in  [prefix+"_x", prefix+"_y"]:
            p = '{}/{}/'.format(data_dir, d)
            self.path_all.append(sorted([p+f for f in os.listdir(p)]))

    def __getitem__(self, idx):
        with mrcfile.open(self.path_all[0][idx]) as mrc:
            rx = mrc.data[np.newaxis,:,:,:]
        with mrcfile.open(self.path_all[1][idx]) as mrc:
            ry = mrc.data[np.newaxis,:,:,:]
        #rx = mrcfile.open(self.path_all[0][idx]).data[np.newaxis,:,:,:]
        #ry = mrcfile.open(self.path_all[1][idx]).data[np.newaxis,:,:,:]
        rx = torch.from_numpy(rx.copy())
        ry = torch.from_numpy(ry.copy())

        return rx, ry

    def __len__(self):
        return len(self.path_all[0])

from IsoNet.preprocessing.img_processing import normalize
class Predict_sets(Dataset):
    def __init__(self, mrc_list):
        super(Predict_sets, self).__init__()
        self.mrc_list=mrc_list

    def __getitem__(self, idx):
        rx = mrcfile.open(self.mrc_list[idx]).data[np.newaxis,:,:,:]
        rx=normalize(-rx, percentile = True)

        return dict(image=rx)

    def __len__(self):
        return len(self.mrc_list)



def get_datasets(settings):
    base_folder = settings.data_dir#pathlib.Path(get_brats_folder(on)).resolve()
    print(base_folder)

    train_dataset = Brats(base_folder, prefix="train")
    val_dataset = Brats(base_folder, prefix="test")
    return train_dataset, val_dataset#, bench_dataset