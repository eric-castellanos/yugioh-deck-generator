FROM ubuntu:22.04

# Install dependencies and Mono
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gnupg \
        ca-certificates \
        curl \
        git \
        netbase \
        inetutils-tools \
        software-properties-common && \
    curl -sSL https://download.mono-project.com/repo/xamarin.gpg | gpg --dearmor -o /usr/share/keyrings/mono-official.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/mono-official.gpg] https://download.mono-project.com/repo/ubuntu stable-focal main" > /etc/apt/sources.list.d/mono-official.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends mono-complete && \
    rm -rf /var/lib/apt/lists/*

# Clone WindBot-Ignite
WORKDIR /windbot-source
RUN git clone --recursive https://github.com/crispy-chiken/windbot.git .

# Build WindBot (output goes to /windbot-source/bin/Debug)
RUN msbuild WindBot.csproj /p:Configuration=Release

# Ensure target directory exists, then copy actual WindBot.exe from correct path
RUN mkdir -p /windbot && \
    mkdir -p /windbot/Decks && \
    cp /windbot-source/out/WindBot/obj/x86/Release/WindBot.exe /windbot/WindBot.exe && \
    cp /windbot-source/ExecutorBase/obj/Release/ExecutorBase.dll /windbot/ExecutorBase.dll && \
    cp -r /windbot-source/Dialogs /windbot/Dialogs && \
    mkdir -p /windbot/Executors

COPY decks/known /windbot/Decks

# Download required card database
WORKDIR /windbot
RUN curl --retry 5 --connect-timeout 30 -L -o cards.cdb https://github.com/moecube/ygopro-database/raw/master/locales/zh-CN/cards.cdb

# Expose port used by WindBot server
EXPOSE 2399

# Copy startup script
COPY run-windbot.sh /run-windbot.sh
RUN chmod +x /run-windbot.sh

# Use script as entrypoint
ENTRYPOINT ["/run-windbot.sh"]
