FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

SHELL ["/bin/bash", "-c"]
RUN /bin/sed -i 's/archive\.ubuntu\.com/mirror\.yandex\.ru/g' /etc/apt/sources.list && apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        language-pack-en \
        language-pack-ru \
        pkg-config \
        libgl1-mesa-glx \
        libpng-dev \
        libjpeg8-dev \
        imagemagick \
        git \
        cmake \
        curl \
        wget \
        unzip \
        vim \
        openssh-server \
        supervisor \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
    
WORKDIR /tmp
RUN locale-gen en_US.UTF-8 ru_RU.UTF-8 && update-locale LANG="en_US.UTF-8" LANGUAGE="en_US"
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/conda
ENV PATH "/root/conda/bin:$PATH"
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ADD docker/dev/environment.yml /tmp/environment.yml
RUN conda update -n base -c defaults conda && \
    conda env create -f environment.yml && \
    conda init bash && \
    conda clean -a
RUN echo "conda activate da-ssd" >> ~/.bashrc
RUN git clone https://github.com/NVIDIA/apex
RUN /root/conda/envs/da-ssd/bin/pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
RUN /root/conda/envs/da-ssd/bin/pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100

COPY ./setup.py .
COPY ./csrc ./csrc
RUN /root/conda/envs/da-ssd/bin/pip install .

EXPOSE 22
EXPOSE 8080
EXPOSE 8888
EXPOSE 6006

ARG UID
ENV NOTVISIBLE "in users profile"

RUN useradd -d /home/dev -ms /bin/bash -u $UID -G sudo dev
RUN echo dev:dev | chpasswd
RUN chmod 775 /root

RUN echo "export PATH=$PATH" >> /home/dev/.bashrc
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> /home/dev/.bashrc
RUN echo "export VISIBLE=now" >> /home/dev/.bashrc
RUN echo "export NOTVISIBLE=$NOTVISIBLE" >> /home/dev/.bashrc
RUN cat /root/.bashrc >> /home/dev/.bashrc

RUN mkdir /var/run/sshd
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
RUN echo "echo \"PasswordAuthentication no\" >> /etc/ssh/sshd_config" >> /root/ssh_disable_password.sh && \
    echo "service ssh restart" >> /root/ssh_disable_password.sh && \
    chmod +x /root/ssh_disable_password.sh

USER ${USERNAME}
WORKDIR /home/dev

ADD docker/dev/supervisord /etc/supervisor/conf.d/
RUN exec /bin/bash && conda activate pt
ENTRYPOINT ["supervisord", "-n"]
