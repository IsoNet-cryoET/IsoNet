##############################
## configuration
ARG TF_ver=latest

##############################
FROM tensorflow/tensorflow:${TF_ver}-gpu

RUN pip install --no-cache-dir --upgrade pip

# add user so we don't have to run under root
RUN adduser --disabled-password --gecos "" isonet

USER isonet

# preinstall all the requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
