FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git cmake make g++ wget unzip \
    libevent-dev libsqlite3-dev liblua5.3-dev \
    zlib1g-dev libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -ms /bin/bash edopro
USER edopro
WORKDIR /home/edopro

# Clone the EDOPro repo
RUN git clone https://github.com/ProjectIgnis/edopro.git

# Build EDOPro and ocgcore together from the correct folder
WORKDIR /home/edopro/edopro
RUN mkdir build && cd build && cmake .. && make -j

WORKDIR /home/edopro/edopro/build

# Expose port if needed
EXPOSE 7911

# Start the app (override in `docker run` if desired)
CMD ["./ygopro"]