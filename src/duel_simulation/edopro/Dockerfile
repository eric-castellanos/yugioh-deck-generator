FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -qq -y \
    build-essential git cmake wget unzip \
    libevent-dev libcurl4-openssl-dev libssl-dev \
    libboost-all-dev libsqlite3-dev libfreetype6-dev \
    libirrlicht-dev libfmt-dev zlib1g-dev ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install premake5
RUN wget https://github.com/premake/premake-core/releases/download/v5.0.0-alpha15/premake-5.0.0-alpha15-linux.tar.gz \
 && tar -xzf premake-5.0.0-alpha15-linux.tar.gz \
 && mv premake5 /usr/local/bin/ \
 && rm premake-5.0.0-alpha15-linux.tar.gz

WORKDIR /edopro
RUN git clone --recursive https://github.com/edo9300/edopro.git

# Retrieve patches
RUN git clone --depth 1 https://github.com/crispy-chiken/YugiohAi.git /yugioh_ai

WORKDIR /edopro/edopro
# Apply patches before build
RUN cp -r /yugioh_ai/edopro_changes/* ./

# Build headless with patched automation support
RUN premake5 gmake2 --vcpkg-root=. \
 && make -C build -j$(nproc) config=debug_x64 lua clzma ocgcore
