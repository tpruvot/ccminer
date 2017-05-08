#!/bin/bash

# Simple script to create the Makefile and build

# export PATH="$PATH:/usr/local/cuda/bin/"

make distclean || echo clean

rm -f Makefile.in
rm -f config.status
./autogen.sh || echo done

# CFLAGS="-O2" ./configure
./configure.sh

make -j 4

if [ "$(uname)" == "Darwin" ]; then
install_name_tool -change $(otool -L ./ccminer | grep -o ".*/libomp\.dylib") @executable_path/libomp.dylib ./ccminer
install_name_tool -change $(otool -L ./ccminer | grep -o ".*/libc++\..*\.dylib") /usr/lib/libc++.dylib ./ccminer
install_name_tool -change $(otool -L ./ccminer | grep -o "@rpath/libcudart\..*\.dylib") /usr/local/cuda/lib/libcudart.dylib ./ccminer
fi
