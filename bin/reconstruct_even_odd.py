import numpy as np
from subprocess import call

def tilt(tiltSeries, output, tiltfile, exclude_list):
    s="tilt -input {} \
            -output {} \
            -TILTFILE {} \
            -THICKNESS {} \
            -RADIAL 0.35,0.035 \
            -RotateBy90 \
            -FakeSIRTiterations 5 \
            -EXCLUDELIST2 {}".format(tiltSeries, output, tiltfile,200, exclude_list)
    print(s)
    call(s, shell=True)

    

def generate_tomograms(tiltSeries, tiltAngle):
    tilt_angle = np.loadtxt(tiltAngle)
    num_of_images=len(tilt_angle)
    if num_of_images%2==0:
        even_list= [2*(i+1) for i in range(num_of_images//2)]
        odd_list= [2*i+1 for i in range(num_of_images//2)]
    else:
        even_list= [2*(i+1) for i in range((num_of_images-1)//2-1)]
        odd_list= [2*i+1 for i in range((num_of_images-1)//2)]
    output="odd.mrc"
    tilt(tiltSeries, "odd.mrc", tiltAngle, ','.join(map(str,even_list)))
    tilt(tiltSeries, "even.mrc", tiltAngle, ','.join(map(str,odd_list)))

   

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('tiltSeries', type=str, help='an integer for the accumulator')
    parser.add_argument('tiltAngle', type=str, help='sum the integers (default: find the max)')

    args = parser.parse_args()
    generate_tomograms(args.tiltSeries, args.tiltAngle)
