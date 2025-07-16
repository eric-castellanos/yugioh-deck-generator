FROM ubuntu:22.04

# Install required libraries
RUN apt-get update && apt-get install -y \
    xvfb wget libgl1-mesa-glx libxcb1 libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6 libcurl4 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/edopro

# Download and extract the assets
RUN wget -O edopro.tar.gz https://github.com/ProjectIgnis/edopro-assets/releases/download/41.0.2/ProjectIgnis-EDOPro-41.0.2-linux.tar.gz && \
    tar -xzf edopro.tar.gz && rm edopro.tar.gz

WORKDIR /opt/edopro/ProjectIgnis

# Optional but recommended
RUN chmod +x EDOPro

EXPOSE 2399

# Start headless server
CMD ["sh", "-c", "echo '🚀 Launching EDOPro...' && xvfb-run -a ./EDOPro -s -i-want-to-be-admin"]
