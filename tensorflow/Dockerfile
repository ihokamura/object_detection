FROM pyenv
SHELL ["/bin/bash", "-c"]

# Install TensorFlow
RUN pip3 install tensorflow
RUN pip3 install tf-models-official

# Setup work directory
RUN mkdir /work
WORKDIR /work
