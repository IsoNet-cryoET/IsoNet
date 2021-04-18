import numpy as np
import mrcfile
import os
def tom_ctf1d(pixelsize, voltage, cs, defocus, amplitude, phaseshift, bfactor, length=2048):

    ny = 1 / pixelsize


    lambda1 = 12.2643247 / np.sqrt(voltage * (1.0 + voltage * 0.978466e-6)) * 1e-10
    lambda2 = lambda1 * 2


    points = np.arange(0,length)
    points = points.astype(np.float)
    points = points/(2 * length)*ny

    k2 = points**2
    term1 = lambda1**3 * cs * k2**2

    w = np.pi / 2 * (term1 + lambda2 * defocus * k2) - phaseshift

    acurve = np.cos(w) * amplitude
    pcurve = -np.sqrt(1 - amplitude**2) * np.sin(w)
    bfactor = np.exp(-bfactor * k2 * 0.25)


    return (pcurve + acurve)*bfactor

def wiener1d(angpix, defocus, snrfalloff, deconvstrength, highpassnyquist, phaseflipped, phaseshift):
    data = np.arange(0,1+1/2047.,1/2047.)
    highpass = np.minimum(np.ones(data.shape[0]), data/highpassnyquist) * np.pi
    highpass = 1-np.cos(highpass)

    snr = np.exp(-data * snrfalloff * 100 / angpix) * (10^(3 * deconvstrength)) * highpass
    #snr[0] = -1
    ctf = tom_ctf1d(angpix*1e-10, 300e3, 2.7e-3, -defocus*1e-6, 0.07, phaseshift / 180 * np.pi, 0)
    if phaseflipped:
        ctf = abs(ctf)

    wiener = ctf/(ctf*ctf+1/snr)
    return ctf, wiener

def tom_deconv_tomo(vol_file, angpix, defocus, snrfalloff, deconvstrength, highpassnyquist, phaseflipped, phaseshift):
    with mrcfile.open(vol_file) as f:
        vol = f.data
    data = np.arange(0,1+1/2047.,1/2047.)
    highpass = np.minimum(np.ones(data.shape[0]), data/highpassnyquist) * np.pi
    highpass = 1-np.cos(highpass)
    eps = 1e-10
    snr = np.exp(-data * snrfalloff * 100 / angpix) * np.power(10.0,(3.0 * deconvstrength)) * highpass + eps; 
    #snr[0] = -1
    ctf = tom_ctf1d(angpix*1e-10, 300e3, 2.7e-3, -defocus*1e-6, 0.07, phaseshift / 180 * np.pi, 0)
    if phaseflipped:
        ctf = abs(ctf)

    wiener = ctf/(ctf*ctf+1/snr)

    denom = ctf*ctf+1/snr
    #np.savetxt('den.txt',denom)
    #np.savetxt('snr.txt',snr)
    #np.savetxt('hipass.txt',highpass)
    #np.savetxt('ctf.txt',ctf)
    #np.savetxt('wiener.txt',wiener, fmt='%f')

    s1 = - int(np.shape(vol)[1] / 2)
    f1 = s1 + np.shape(vol)[1] - 1
    m1 = np.arange(s1,f1+1)     

    s2 = - int(np.shape(vol)[0] / 2)
    f2 = s2 + np.shape(vol)[0] - 1
    m2 = np.arange(s2,f2+1)     

    s3 = - int(np.shape(vol)[2] / 2)
    f3 = s3 + np.shape(vol)[2] - 1
    m3 = np.arange(s3,f3+1)     

#s3 = -floor(size(vol,3)/2);
#f3 = s3 + size(vol,3) - 1;
    x, y, z = np.meshgrid(m1,m2,m3)
    print(np.shape(x))
    x = x.astype(np.float32) / np.abs(s1)
    y = y.astype(np.float32) / np.abs(s2)
    z = z.astype(np.float32) / np.maximum(1, np.abs(s3))
    #z = z.astype(float) / np.abs(s3);
    r = np.sqrt(x**2+y**2+z**2)
    del x,y,z
    r = np.minimum(1, r)
    r = np.fft.ifftshift(r)

    #x = 0:1/2047:1;
    print(data.shape, wiener.shape, r.shape)
    ramp = np.interp(r, data,wiener).astype(np.float32)
    del r
    #ramp = np.interp(data,wiener,r);
    deconv = np.real(np.fft.ifftn(np.fft.fftn(vol) * ramp))

    with mrcfile.new(os.path.splitext(vol_file)[0]+'_deconv.mrc',overwrite=True) as n:
        n.set_data(deconv.astype(type(vol[0,0,0])))
    #return real(ifftn(fftn(single(vol)).*ramp));
    return os.path.splitext(vol_file)[0]+'_deconv.mrc'

class Chunks:
    def __init__(self,num=2,overlap=0.25):
        self.overlap = overlap
        #num can be either int or tuple
        if type(num) is int:
            self.num = (num,num,num)
        else:
            self.num = num

    def get_chunks(self,tomo_name):
        #side*(1-overlap)*(num-1)+side = sp + side*overlap -> side *(1-overlap) * num = side
        root_name = os.path.splitext(os.path.basename(tomo_name))[0]
        with mrcfile.open(tomo_name) as f:
            vol = f.data
        cube_size = np.round(np.array(vol.shape)/((1-self.overlap)*np.array(self.num))).astype(np.int16)
        overlap_len = np.round(cube_size*self.overlap).astype(np.int16)
        overlap_len = overlap_len + overlap_len %2
        eff_len = cube_size-overlap_len
        padded_vol = np.pad(vol,pad_width=[(ol//2,ol//2) for ol in overlap_len],mode='symmetric')
        sp = padded_vol.shape
        chunks_file_list = []
        slice1 = [(i*eff_len[0],i*eff_len[0]+cube_size[0]) for i in range(self.num[0]-1)]
        slice1.append(((self.num[0]-1)*eff_len[0],sp[0]))
        slice2 = [(i*eff_len[1],i*eff_len[1]+cube_size[1]) for i in range(self.num[1]-1)]
        slice2.append(((self.num[1]-1)*eff_len[1],sp[1]))
        slice3 = [(i*eff_len[2],i*eff_len[2]+cube_size[2]) for i in range(self.num[2]-1)]
        slice3.append(((self.num[2]-1)*eff_len[2],sp[2]))
        # print(slice1)
        # print(slice2)
        # print(slice3)
        for n1,i in enumerate(slice1):
            for n2,j in enumerate(slice2):
                for n3,k in enumerate(slice3):
                    one_chunk = padded_vol[i[0]:i[1],j[0]:j[1],k[0]:k[1]]
                    file_name = './deconv_temp/'+root_name+'_{}_{}_{}.mrc'.format(n1,n2,n3)
                    with mrcfile.new(file_name,overwrite=True) as n:
                        n.set_data(one_chunk)
                    chunks_file_list.append(file_name)
        self.shape = vol.shape
        self.padded_shape = sp
        self.slice1 = slice1
        self.slice2 = slice2
        self.slice3 = slice3
        self.overlap_len = overlap_len
        self.datatype = type(vol[0,0,0])
        return chunks_file_list

    def restore(self,new_file_list):
        overlap_len = self.overlap_len
        new_vol = np.zeros(self.padded_shape,self.datatype)
        for n1,i in enumerate(self.slice1):
            for n2,j in enumerate(self.slice2):
                for n3,k in enumerate(self.slice3):
                    print(n1,n2,n3)
                    one_chunk_file = new_file_list[n1*len(self.slice2)*len(self.slice3)+n2*len(self.slice3)+n3]
                    with mrcfile.open(one_chunk_file) as f:
                        one_chuck = f.data
                    print(one_chuck[overlap_len[0]//2:-(overlap_len[0]//2),overlap_len[1]//2:-(overlap_len[1]//2),overlap_len[2]//2:-(overlap_len[2]//2)].shape)
                    print(one_chuck.shape)
                    new_vol[i[0]+overlap_len[0]//2:i[1]-overlap_len[0]//2,j[0]+overlap_len[1]//2:j[1]-overlap_len[1]//2, 
                    k[0]+overlap_len[2]//2:k[1]-overlap_len[2]//2] = one_chuck[overlap_len[0]//2:-(overlap_len[0]//2),overlap_len[1]//2:-(overlap_len[1]//2),overlap_len[2]//2:-(overlap_len[2]//2)]
                    
         
        # return np.multiply(new_vol,1/factor_vol)
        return new_vol[overlap_len[0]//2:-(overlap_len[0]//2),
                    overlap_len[1]//2:-(overlap_len[1]//2),
                    overlap_len[2]//2:-(overlap_len[2]//2)]