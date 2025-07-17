FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /opt/ygopro

# Install system packages
RUN apt-get update && apt-get install -y \
    git wget curl unzip p7zip-full iproute2 netcat build-essential autoconf libtool \
    pkg-config tar xz-utils \
    libgl1-mesa-dev libglu1-mesa-dev libxxf86vm-dev \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Clone YGOPro repo with submodules
RUN git clone --recursive https://github.com/Fluorohydride/ygopro.git . && \
    cd ocgcore && git checkout master && git pull origin master && cd ..

# Download premake
ENV PREMAKE_VERSION=5.0.0-beta4
RUN wget https://github.com/premake/premake-core/releases/download/v${PREMAKE_VERSION}/premake-${PREMAKE_VERSION}-linux.tar.gz && \
    tar xf premake-${PREMAKE_VERSION}-linux.tar.gz && \
    chmod +x premake5 && \
    mv premake5 /usr/local/bin/ && \
    rm premake-${PREMAKE_VERSION}-linux.tar.gz

# -------- STATIC DEPENDENCIES --------

# Download, extract, and configure libevent
RUN wget https://github.com/libevent/libevent/releases/download/release-2.1.12-stable/libevent-2.1.12-stable.tar.gz && \
    tar xf libevent-2.1.12-stable.tar.gz && \
    mv libevent-2.1.12-stable event && \
    cd event && \
    ./configure --disable-openssl --enable-static=yes --enable-shared=no && \
    mkdir -p include/event2 && \
    sed -f make-event-config.sed < config.h > ./include/event2/event-config.h && \
    cd .. && \
    rm libevent-2.1.12-stable.tar.gz

# freetype
RUN wget https://downloads.sourceforge.net/freetype/freetype-2.13.3.tar.gz && \
    tar xf freetype-2.13.3.tar.gz && \
    mv freetype-2.13.3 freetype && \
    rm freetype-2.13.3.tar.gz

# sqlite amalgamation
RUN wget https://www.sqlite.org/2025/sqlite-amalgamation-3500100.zip && \
    7z x sqlite-amalgamation-3500100.zip && \
    mv sqlite-amalgamation-3500100 sqlite3 && \
    rm sqlite-amalgamation-3500100.zip

# lua
RUN wget https://www.lua.org/ftp/lua-5.4.8.tar.gz && \
    tar xf lua-5.4.8.tar.gz && \
    mv lua-5.4.8 lua && \
    rm lua-5.4.8.tar.gz

# miniaudio + ogg/opus/vorbis
RUN git clone --depth=1 https://github.com/mercury233/miniaudio && \
    cd miniaudio && cp extras/miniaudio_split/miniaudio.* . && cd ..

RUN wget https://github.com/xiph/ogg/releases/download/v1.3.5/libogg-1.3.5.tar.gz && \
    tar xf libogg-1.3.5.tar.gz && \
    mv libogg-1.3.5 miniaudio/external/ogg && \
    rm libogg-1.3.5.tar.gz

# After ogg is extracted
RUN cd miniaudio/external/ogg && \
    ./configure && \
    cd ../../..

RUN wget https://github.com/xiph/opus/releases/download/v1.5.2/opus-1.5.2.tar.gz && \
    tar xf opus-1.5.2.tar.gz && \
    mv opus-1.5.2 miniaudio/external/opus && \
    rm opus-1.5.2.tar.gz

RUN wget https://github.com/xiph/opusfile/releases/download/v0.12/opusfile-0.12.tar.gz && \
    tar xf opusfile-0.12.tar.gz && \
    mv opusfile-0.12 miniaudio/external/opusfile && \
    rm opusfile-0.12.tar.gz

RUN wget https://github.com/xiph/vorbis/releases/download/v1.3.7/libvorbis-1.3.7.tar.gz && \
    tar xf libvorbis-1.3.7.tar.gz && \
    mv libvorbis-1.3.7 miniaudio/external/vorbis && \
    rm libvorbis-1.3.7.tar.gz

# irrlicht
RUN git clone --depth=1 https://github.com/mercury233/irrlicht

# Copy premake/resource files
RUN cp -r premake/* . && cp -r resource/* .

# Patch premake5.lua (stub unimplemented functions)
# RUN echo 'function linktimeoptimization() end\nfunction conformancemode() end\nfunction vectorextensions(_) end\nfunction copyruntime() end\nfunction openmp() end' | cat - premake5.lua > temp && mv temp premake5.lua

# Generate makefiles for static build
RUN premake5 gmake \
    --build-event \
    --build-freetype \
    --build-sqlite \
    --build-opus-vorbis

# Build
RUN cd build && make config=release -j$(nproc)

# Entrypoint (optional: you may want to copy binary somewhere else)
WORKDIR /opt/ygopro

RUN apt-get update && apt-get install -y xvfb && \
    mkdir -p fonts && \
    wget -O cards.cdb https://raw.githubusercontent.com/cromerc/ygopro/master/cards.cdb && \
    wget -O fonts/numFont.ttf https://raw.githubusercontent.com/cromerc/ygopro/master/fonts/arialbd.ttf && \
    wget -O fonts/textFont.ttf https://raw.githubusercontent.com/cromerc/ygopro/master/fonts/simhei.ttf

ENTRYPOINT []

