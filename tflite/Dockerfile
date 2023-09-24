FROM pyenv
SHELL ["/bin/bash", "-c"]

# Install TensorFlow Lite
RUN pip3 install tflite-runtime

# Setup work directory
RUN mkdir /work
WORKDIR /work
