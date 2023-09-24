FROM ubuntu
SHELL ["/bin/bash", "-c"]

# Install apt dependencies
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN apt update
RUN apt install -y build-essential
RUN apt install -y git
RUN apt install -y libbz2-dev
RUN apt install -y libffi-dev
RUN apt install -y liblzma-dev
RUN apt install -y libopencv-dev
RUN apt install -y libreadline-dev
RUN apt install -y libsqlite3-dev
RUN apt install -y libssl-dev
RUN apt install -y tk-dev
RUN apt install -y wget
RUN apt install -y zlib1g-dev

# Setup Python environment
RUN git clone https://github.com/pyenv/pyenv.git /usr/local/.pyenv
RUN echo 'export PYENV_ROOT=/usr/local/.pyenv' >> ~/.bash_profile
RUN echo 'export PATH=$PYENV_ROOT/bin:$PATH' >> ~/.bash_profile
RUN echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
RUN source ~/.bash_profile

ENV PYENV_ROOT /usr/local/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH
ENV PATH $PYENV_ROOT/shims:$PATH
RUN pyenv install 3.10.13
RUN pyenv global 3.10.13
RUN pip3 install -U pip
