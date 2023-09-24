FROM pyenv
SHELL ["/bin/bash", "-c"]

# Install apt dependencies
RUN apt install -y protobuf-compiler
RUN apt install -y wget

# Install object detection API
RUN (mkdir /home/tensorflow && cd /home/tensorflow && git clone --depth 1 https://github.com/tensorflow/models.git)
WORKDIR /home/tensorflow/models/research/
RUN protoc object_detection/protos/*.proto --python_out=.
RUN cp object_detection/packages/tf2/setup.py .
RUN pip3 install .

# Setup work directory
RUN mkdir /work
WORKDIR /work
