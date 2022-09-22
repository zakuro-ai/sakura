FROM  pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV MKL_THREADING_LAYER=INTEL="GNU"
RUN apt update && \
    apt install \
    sudo \
    git \
    libopenmpi-dev \
    locales -y

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen

## Add foo user
RUN adduser --disabled-password --gecos '' foo
RUN adduser foo sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN echo 'export PYTHONPATH=/workspace' >> /home/foo/.bashrc

RUN echo "alias mpirun='mpirun --mca btl_base_warn_component_unused 0'" >> /home/foo/.bashrc
