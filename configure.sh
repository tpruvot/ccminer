# To change the cuda arch, edit Makefile.am and run ./build.sh

extracflags="-march=native -D_REENTRANT -falign-functions=16 -falign-jumps=16 -falign-labels=16"

if [ "$(uname)" == "Darwin" ]; then
cxx="/usr/local/opt/llvm/bin/clang++"
fi

CXX="$cxx" \
CUDA_CFLAGS="-O3 -lineno -Xcompiler -Wall  -D_FORCE_INLINES" \
	./configure CXXFLAGS="-O3 $extracflags" --with-cuda=/usr/local/cuda --with-nvml=libnvidia-ml.so

