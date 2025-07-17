FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install required dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    make \
    cmake \
    g++ \
    libevent-dev \
    libfreetype6-dev \
    libsqlite3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxxf86vm-dev \
    libboost-system-dev \
    libboost-thread-dev \
    libboost-filesystem-dev \
    libopusfile-dev \
    libboost-all-dev \
    libvorbis-dev \
    libcurl4-openssl-dev \
    zlib1g-dev \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Premake5
RUN wget https://github.com/premake/premake-core/releases/download/v5.0.0-alpha15/premake-5.0.0-alpha15-linux.tar.gz && \
    tar -xvf premake-5.0.0-alpha15-linux.tar.gz && \
    mv premake5 /usr/local/bin/ && \
    hmod +x /usr/local/bin/premake5 && \
    rm premake-5.0.0-alpha15-linux.tar.gz

# Clone the official YGOPro repo
WORKDIR /opt
RUN git clone --recursive https://github.com/Fluorohydride/ygopro.git

# Copy Description Files
WORKDIR /opt/ygopro
RUN cp -r premake/* .

#RUN chmod +x premake5 && ./premake5 gmake

# Build YGOPro
#WORKDIR /opt/ygopro/build
#RUN make config=release -j$(nproc)

# Expose the duel port
#EXPOSE 7911

# Create symbolic link
#RUN ln -s bin/release/YGOPro ./ygopro

# Entry command to launch the duel server
#ENTRYPOINT ["./ygopro"]
