import numpy as np

def tom_ctf1d(pixelsize, voltage, cs, defocus, amplitude, phaseshift, bfactor, length=2048):

    ny = 1 / pixelsize


    lambda1 = 12.2643247 / np.sqrt(voltage * (1.0 + voltage * 0.978466e-6)) * 1e-10
    lambda2 = lambda1 * 2;


    points = np.arange(0,length)
    points = points.astype(np.float)
    points = points/(2 * length)*ny;

    k2 = points**2;
    term1 = lambda1**3 * cs * k2**2;

    w = np.pi / 2 * (term1 + lambda2 * defocus * k2) - phaseshift;

    acurve = np.cos(w) * amplitude;
    pcurve = -np.sqrt(1 - amplitude**2) * np.sin(w);
    bfactor = np.exp(-bfactor * k2 * 0.25);


    return (pcurve + acurve)*bfactor;

def wiener1d(angpix, defocus, snrfalloff, deconvstrength, highpassnyquist, phaseflipped, phaseshift):
    data = np.arange(0,1+1/2047.,1/2047.)
    highpass = np.minimum(np.ones(data.shape[0]), data/highpassnyquist) * np.pi;
    highpass = 1-np.cos(highpass);

    snr = np.exp(-data * snrfalloff * 100 / angpix) * (10^(3 * deconvstrength)) * highpass;
    #snr[0] = -1
    ctf = tom_ctf1d(angpix*1e-10, 300e3, 2.7e-3, -defocus*1e-6, 0.07, phaseshift / 180 * np.pi, 0);
    if phaseflipped:
        ctf = abs(ctf)

    wiener = ctf/(ctf*ctf+1/snr)
    return ctf, wiener

def tom_deconv_tomo(vol, angpix, defocus, snrfalloff, deconvstrength, highpassnyquist, phaseflipped, phaseshift):

    data = np.arange(0,1+1/2047.,1/2047.)
    highpass = np.minimum(np.ones(data.shape[0]), data/highpassnyquist) * np.pi;
    highpass = 1-np.cos(highpass);

    snr = np.exp(-data * snrfalloff * 100 / angpix) * (10^(3 * deconvstrength)) * highpass;
    #snr[0] = -1
    ctf = tom_ctf1d(angpix*1e-10, 300e3, 2.7e-3, -defocus*1e-6, 0.07, phaseshift / 180 * np.pi, 0);
    if phaseflipped:
        ctf = abs(ctf)

    wiener = ctf/(ctf*ctf+1/snr);

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
    x, y, z = np.meshgrid(m1,m2,m3);
    print(np.shape(x))
    x = x.astype(np.float32) / np.abs(s1);
    y = y.astype(np.float32) / np.abs(s2);
    z = z.astype(np.float32) / np.maximum(1, np.abs(s3));
    #z = z.astype(float) / np.abs(s3);
    r = np.sqrt(x**2+y**2+z**2);
    del x,y,z
    r = np.minimum(1, r);
    r = np.fft.ifftshift(r);

    #x = 0:1/2047:1;
    print(data.shape, wiener.shape, r.shape)
    ramp = np.interp(r, data,wiener).astype(np.float32);
    del r
    #ramp = np.interp(data,wiener,r);
    deconv = np.real(np.fft.ifftn(np.fft.fftn(vol) * ramp))

    #return real(ifftn(fftn(single(vol)).*ramp));
    return deconv.astype(np.float32)