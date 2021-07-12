## Installation
### Install from source
python version at least 3.5 is required.

1.  IsoNet relies on Tensorflow with version at least 2.0

Please find your cuda version and corresponding tensorflow version here: https://www.tensorflow.org/install/source#gpu. For example, if you are using cude 10.1, you should install tensorflow 2.3:

pip install tensorflow-gpu==2.3.0

2.  Install other dependencies

pip install -r requirements.txt

3.  Add environment variables: 

For example add following lines in your ~/.bashrc

export PATH=PATH_TO_ISONET_FOLDER/bin:$PATH 

export PYTHOHPATH=PATH_TO_PARENT_FOLDER_OF_ISONET_FOLDER:$PYTHONPATH 

4. Open a new terminal, enter your working directory and run "isonet.py check"

Tutorial data set and tutorial video is on google drive https://drive.google.com/drive/folders/1DXjIsz6-EiQm7mBMuMHHwdZErZ_bXAgp
