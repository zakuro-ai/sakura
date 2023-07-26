FROM zakuroai/compute

RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

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

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN echo "alias sakura='figlet Sakura ML && mpirun --mca btl_base_warn_component_unused 0 -np 2 python -W ignore'" >> /home/foo/.bashrc
RUN echo "alias sakura='figlet Sakura ML && mpirun --mca btl_base_warn_component_unused 0 -np 2 python -W ignore'" >> /root/.bashrc

CMD ["sakura"]