# Install all the runtime dependencies for Multirole.
FROM alpine:edge AS base
RUN apk add --no-cache --repository "@testing http://dl-cdn.alpinelinux.org/alpine/edge/testing" \
    boost-filesystem ca-certificates libgit2 libssl3 tcmalloc-minimal@testing sqlite-libs && \
    rm -rf /var/log/* /tmp/* /var/tmp/*

# Install all the development environment that Multirole needs.
FROM base AS base-dev
RUN apk add --no-cache --repository "@testing http://dl-cdn.alpinelinux.org/alpine/edge/testing" \
    boost-dev fmt-dev g++ gperftools-dev@testing libgit2-dev meson ninja openssl-dev sqlite-dev git && \
    rm -rf /var/log/* /tmp/* /var/tmp/*

# Build multirole, stripping debug symbols to their own files.
FROM base-dev AS built
WORKDIR /root
RUN git clone https://github.com/DyXel/Multirole multirole-src
WORKDIR /root/multirole-src
ENV BOOST_INCLUDEDIR=/usr/include/boost BOOST_LIBRARYDIR=/usr/lib
RUN echo "Description: See https://gitlab.alpinelinux.org/alpine/aports/-/issues/15364" >> "/usr/lib/pkgconfig/libtcmalloc_minimal.pc" && \
    meson setup build -Doptimization=3 -Ddebug=true -Db_lto=true -Db_pie=true -Dcpp_link_args="-static-libstdc++" -Duse_tcmalloc=enabled -Dfmt_ho=true && \
    cd build && \
    meson compile && \
    objcopy --only-keep-debug hornet hornet.debug && \
    strip --strip-debug --strip-unneeded hornet && \
    objcopy --add-gnu-debuglink=hornet.debug hornet && \
    objcopy --only-keep-debug multirole multirole.debug && \
    strip --strip-debug --strip-unneeded multirole && \
    objcopy --add-gnu-debuglink=multirole.debug multirole

# Setup the final execution environment.
FROM base
RUN echo "@testing http://dl-cdn.alpinelinux.org/alpine/edge/testing" >> /etc/apk/repositories

WORKDIR /multirole

COPY --from=built /root/multirole-src/etc/config.json .
COPY --from=built /root/multirole-src/util/area-zero.sh .
COPY --from=built /root/multirole-src/build/hornet .
COPY --from=built /root/multirole-src/build/multirole .

RUN apk add --no-cache --repository "http://dl-cdn.alpinelinux.org/alpine/edge/testing" git && \
    mkdir -p sync/scripts sync/banlists sync/databases sync/bin && \
    git clone --depth 1 https://github.com/ProjectIgnis/CardScripts.git sync/scripts && \
    git clone --depth 1 https://github.com/ProjectIgnis/LFLists.git sync/banlists && \
    git clone --depth 1 https://github.com/moecube/ygopro-database.git sync/databases && \
    git clone --depth 1 https://github.com/ProjectIgnis/bin.git sync/bin

EXPOSE 7922 7911 34343 62672 49382 43632

CMD ["./area-zero.sh"]

