#!/usr/bin/env python3
import fire 

class MWR:
    """
    MWR: Train on tomograms and Predict to restore missing-wedge
    """
    def train(self, tomo_path: str, binv: int=4, if_mask: bool=True, gpuID: str = '0,1,2,3'):
        """
        Preprocess tomogram and train u-net model on generated subtomos
        :param tomo_path: path to tomogram from which subtomos are sampled; format: .mwr or .rec
        :param binv: binvolume of tomogram, will be used to determine the subtomos' size
        :param if_mask: if sample subtomos with a mask to exclude background region.

        """
        print('training')

    def predict(self, tomo_path: str, output: str, binv: int, model_path: str):
        """
        Predict tomograms using trained model including model.json and weight(xxx.h5)
        :param tomo_path: path to tomogram format: .mwr or .rec
        :param binv: binvolume of tomogram, will be used to determine the subtomos' size
        :param model_path: path to trained model
        :param output: 
        """
        print('successfully predicted')

    def makemask(self,tomo_path,mask_name):
        """
        generate a mask to constrain sampling area of the tomogram
        :param tomo_path: path to the tomogram
        :param mask_name: path and name of the mask to save as
        """
        print('mask generated')
    def genenoise(self,outfolder: str,num: int, cubesize: int, ncpus: int=20):
        """
        generate training noise to accelerate the missing wedge information retrieval
        :param outfolder: path to folder for saving noises
        :param num: number of noise cubes to generate
        :param cubesize: side length of the noise cubes, usually 64 or 96
        :param ncpus: number of cpus to use
        """
        print('noise generated ')
            
    def setting(self):
        """
        set required training parameters
        """
        pass
    
    
if __name__ == "__main__":
    fire.Fire(MWR)
