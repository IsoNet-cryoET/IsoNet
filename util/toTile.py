import numpy as np

class reform3D:
    def __init__(self,data3D):
        self._sp = data3D.shape
        self._orig_data = data3D

    def pad_and_crop_new(self, cubesize=32, cropsize = 64):
        self._cropsize = cropsize
        sp = np.array(self._sp)
        self._sidelen = sp//cubesize+1
        padi = np.int((cropsize - cubesize)/2)
        padsize = (self._sidelen*cubesize + padi - sp).astype(int)
        data = np.pad(self._orig_data,((padi,padsize[0]),(padi,padsize[1]),(padi,padsize[2]),(0,0)),'symmetric')
        outdata=[]

        for i in range(self._sidelen[0]):
            for j in range(self._sidelen[1]):
                for k in range(self._sidelen[2]):
                    cube = data[i*cubesize:i*cubesize+cropsize,
                            j*cubesize:j*cubesize+cropsize,
                            k*cubesize:k*cubesize+cropsize]
                    outdata.append(cube)
        outdata=np.array(outdata)
        return outdata




    def pad_and_crop(self,cropsize=(64,64,64)):
        self._cropsize = cropsize
        sp = np.array(self._sp)
        padsize = (sp//64+1)*64-sp
        data = np.pad(self._orig_data,((0,padsize[0]),(0,padsize[1]),(0,padsize[2]),(0,0)),'edge')
        self._sidelen = (padsize+sp)//64

        outdata=[]
        for i in range(self._sidelen[0]):
            for j in range(self._sidelen[1]):
                for k in range(self._sidelen[2]):
                    cube = data[i*cropsize[0]:(i+1)*cropsize[0],j*cropsize[0]:(j+1)*cropsize[0],k*cropsize[0]:(k+1)*cropsize[0]]
                    outdata.append(cube)
        outdata=np.array(outdata)
        return outdata

    def restore_from_cubes(self,cubes):
        if len(cubes.shape)==5 and cubes.shape[-1]==1:
            cubes = cubes.reshape(cubes.shape[0:-1])
        new = np.zeros((self._sidelen[0]*64,self._sidelen[1]*64,self._sidelen[2]*64))
        for i in range(self._sidelen[0]):
            for j in range(self._sidelen[1]):
                for k in range(self._sidelen[2]):
                    new[i*self._cropsize[0]:(i+1)*self._cropsize[0],j*self._cropsize[0]:(j+1)*self._cropsize[0],k*self._cropsize[0]:(k+1)*self._cropsize[0]] \
                     = cubes[i*self._sidelen[1]*self._sidelen[2]+j*self._sidelen[1]+k]
        return new[0:self._sp[0],0:self._sp[1],0:self._sp[2]]

    def restore_from_cubes_new(self,cubes, cubesize = 32, cropsize = 64):
        if len(cubes.shape)==5 and cubes.shape[-1]==1:
            cubes = cubes.reshape(cubes.shape[0:-1])

        new = np.zeros((self._sidelen[0]*cubesize,self._sidelen[1]*cubesize,self._sidelen[2]*cubesize))
        start=int((cropsize-cubesize)/2)
        end=int((cropsize+cubesize)/2)
        
        for i in range(self._sidelen[0]):
            for j in range(self._sidelen[1]):
                for k in range(self._sidelen[2]):
                    new[i*cubesize:(i+1)*cubesize,j*cubesize:(j+1)*cubesize,k*cubesize:(k+1)*cubesize] \
                            = cubes[i*self._sidelen[1]*self._sidelen[2]+j*self._sidelen[2]+k][start:end,start:end,start:end]
        return new[0:self._sp[0],0:self._sp[1],0:self._sp[2]]


    def pad4times(self,time=4):
        sp = np.array(self._orig_data.shape)
        sp = np.expand_dims(sp,axis=0)
        padsize = (sp // time + 1) * time - sp
        self._padsize =padsize
        print(padsize, np.zeros((len(self._orig_data.shape),1)))
        width = np.concatenate((np.zeros((len(self._orig_data.shape),1),int),padsize.T),axis=1)
        return np.pad(self._orig_data,width,'edge')
    def cropback(self,padded):
        sp = padded.shape
        ps = self._padsize
        orig_sp = sp - ps
        print ('orig_sp',orig_sp)
        return padded[:orig_sp[0][0]][:orig_sp[0][1]][:orig_sp[0][2]]

if __name__ == '__main__':
    pass
