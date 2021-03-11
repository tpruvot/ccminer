FROM ubuntu:rolling
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get dist-upgrade -y && \
    apt-get install libcurl4-openssl-dev \
    libssl-dev \
    libjansson-dev \
    automake \
    autotools-dev \
    build-essential \
    git \
    freeglut3 \
    freeglut3-dev \
    libxi-dev \
    libxmu-dev \
    wget \
    nvidia-driver-455 \    
    -y 
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/11.2.1/local_installers/cuda-repo-ubuntu2004-11-2-local_11.2.1-460.32.03-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2004-11-2-local_11.2.1-460.32.03-1_amd64.deb
RUN apt-key add /var/cuda-repo-ubuntu2004-11-2-local/7fa2af80.pub
RUN apt-get update
RUN apt-get -y install cuda
RUN git clone https://github.com/tpruvot/ccminer
WORKDIR ccminer
RUN git checkout linux
RUN ./build.sh
RUN ldconfig /usr/local/cuda/lib64
#RUN ./ccminer --version
RUN apt-get remove libcurl4-openssl-dev \
    libssl-dev \
    libjansson-dev \
    freeglut3 \
    freeglut3-dev \
    libxi-dev \
    libxmu-dev \
    -y 
#ENTRYPOINT [ "./ccminer" ]
