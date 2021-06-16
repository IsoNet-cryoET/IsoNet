## Installation

### Install with pip

Not ready. This should be available once GitHub link is released
pip3 install isonet (-cudax)

### Install from source

1.  IsoNet relies on Tensorflow with version at least 2.0 
Please find your cuda version and corresponding tensorflow version here: https://www.tensorflow.org/install/source#gpu

2.  Install other dependencies
pip3 install mrcfile fire scipy numpy skimage

3.  Add environment variables: 
For example added following lines in your ~/.bashrc
export PATH=PATH_TO_ISONET_FOLDER/bin:$PATH 
export PYTHOHPATH=PATH_TO_PARENT_FOLDER_OF_ISONET_FOLDER:$PYTHONPATH 

4. run isonet.py
