# allow to use Host cuda functions in C/C++
DEF_INCLUDES = @CUDA_INCLUDES@

if WANT_JANSSON
JANSSON_INCLUDES= -I$(top_srcdir)/compat/jansson
else
JANSSON_INCLUDES=
endif

EXTRA_DIST		= autogen.sh README.md README.txt LICENSE.txt \
			  ccminer.sln ccminer.vcxproj ccminer.vcxproj.filters \
			  compat/gettimeofday.c compat/getopt/getopt_long.c cpuminer-config.h.in

SUBDIRS = compat

bin_PROGRAMS = ccminer

ccminer_SOURCES	= elist.h miner.h compat.h \
			  compat/inttypes.h compat/stdbool.h compat/unistd.h \
			  compat/sys/time.h compat/getopt/getopt.h \
			  crc32.c hefty1.c \
			  ccminer.cpp pools.cpp util.cpp bench.cpp bignum.cpp \
			  api.cpp hashlog.cpp nvml.cpp stats.cpp sysinfos.cpp cuda.cpp \
			  heavy/heavy.cu \
			  heavy/cuda_blake512.cu heavy/cuda_blake512.h \
			  heavy/cuda_combine.cu heavy/cuda_combine.h \
			  heavy/cuda_groestl512.cu heavy/cuda_groestl512.h \
			  heavy/cuda_hefty1.cu heavy/cuda_hefty1.h \
			  heavy/cuda_keccak512.cu heavy/cuda_keccak512.h \
			  heavy/cuda_sha256.cu heavy/cuda_sha256.h \
			  fuguecoin.cpp Algo256/cuda_fugue256.cu sph/fugue.c uint256.h \
			  groestlcoin.cpp cuda_groestlcoin.cu cuda_groestlcoin.h \
			  myriadgroestl.cpp cuda_myriadgroestl.cu \
			  lyra2/Lyra2.c lyra2/Sponge.c \
			  lyra2/lyra2RE.cu lyra2/cuda_lyra2.cu \
		          lyra2/lyra2REv2.cu lyra2/cuda_lyra2v2.cu \
			  Algo256/cuda_bmw256.cu Algo256/cuda_cubehash256.cu \
			  Algo256/cuda_blake256.cu Algo256/cuda_groestl256.cu Algo256/cuda_keccak256.cu Algo256/cuda_skein256.cu \
			  Algo256/blake256.cu Algo256/decred.cu Algo256/vanilla.cu Algo256/keccak256.cu \
			  Algo256/bmw.cu Algo256/cuda_bmw.cu \
			  JHA/jackpotcoin.cu JHA/cuda_jha_keccak512.cu \
			  JHA/cuda_jha_compactionTest.cu cuda_checkhash.cu \
			  quark/cuda_jh512.cu quark/cuda_quark_blake512.cu quark/cuda_quark_groestl512.cu quark/cuda_skein512.cu \
			  quark/cuda_bmw512.cu quark/cuda_quark_keccak512.cu \
			  quark/quarkcoin.cu quark/cuda_quark_compactionTest.cu \
			  neoscrypt/neoscrypt.cpp neoscrypt/neoscrypt-cpu.c neoscrypt/cuda_neoscrypt.cu \
			  cuda_nist5.cu pentablake.cu skein.cu cuda_skeincoin.cu skein2.cpp zr5.cu \
			  sph/bmw.c sph/blake.c sph/groestl.c sph/jh.c sph/keccak.c sph/skein.c \
			  sph/cubehash.c sph/echo.c sph/luffa.c sph/sha2.c sph/shavite.c sph/simd.c \
			  sph/hamsi.c sph/hamsi_helper.c sph/streebog.c \
			  sph/shabal.c sph/whirlpool.c sph/sha2big.c sph/haval.c \
			  qubit/qubit.cu qubit/qubit_luffa512.cu qubit/deep.cu qubit/luffa.cu \
			  x11/x11.cu x11/fresh.cu x11/cuda_x11_luffa512.cu x11/cuda_x11_cubehash512.cu \
			  x11/cuda_x11_shavite512.cu x11/cuda_x11_simd512.cu x11/cuda_x11_echo.cu \
			  x11/cuda_x11_luffa512_Cubehash.cu \
			  x13/x13.cu x13/cuda_x13_hamsi512.cu x13/cuda_x13_fugue512.cu \
			  x15/x14.cu x15/x15.cu x15/cuda_x14_shabal512.cu x15/cuda_x15_whirlpool.cu \
			  x15/whirlpool.cu \
			  x17/x17.cu x17/cuda_x17_haval512.cu x17/cuda_x17_sha512.cu \
			  x11/c11.cu x11/s3.cu x11/sib.cu x11/cuda_streebog.cu

# scrypt
ccminer_SOURCES += scrypt.cpp scrypt-jane.cpp \
    scrypt/blake.cu scrypt/keccak.cu scrypt/sha256.cu \
    scrypt/salsa_kernel.cu scrypt/test_kernel.cu \
    scrypt/fermi_kernel.cu scrypt/kepler_kernel.cu \
    scrypt/nv_kernel.cu scrypt/nv_kernel2.cu scrypt/titan_kernel.cu

if HAVE_NVML
nvml_defs = -DUSE_WRAPNVML
nvml_libs = -ldl
endif

if HAVE_WINDOWS
ccminer_SOURCES += compat/winansi.c
endif

ccminer_LDFLAGS  = $(PTHREAD_FLAGS) @CUDA_LDFLAGS@
ccminer_LDADD    = @LIBCURL@ @JANSSON_LIBS@ @PTHREAD_LIBS@ @WS2_LIBS@ @CUDA_LIBS@ @PCILIB@ @OPENMP_CFLAGS@ @LIBS@ $(nvml_libs)
ccminer_CPPFLAGS = @LIBCURL_CPPFLAGS@ @PCIFLAGS@ @OPENMP_CFLAGS@ $(CPPFLAGS) $(PTHREAD_FLAGS) -fno-strict-aliasing $(JANSSON_INCLUDES) $(DEF_INCLUDES) $(nvml_defs)

nvcc_ARCH  = -gencode=arch=compute_50,code=\"sm_50,compute_50\"
nvcc_ARCH += -gencode=arch=compute_52,code=\"sm_52,compute_52\"
#nvcc_ARCH += -gencode=arch=compute_35,code=\"sm_35,compute_35\"
#nvcc_ARCH += -gencode=arch=compute_30,code=\"sm_30,compute_30\"
#nvcc_ARCH += -gencode=arch=compute_20,code=\"sm_21,compute_20\"

nvcc_FLAGS = $(nvcc_ARCH) @CUDA_INCLUDES@ -I. @CUDA_CFLAGS@
nvcc_FLAGS += $(JANSSON_INCLUDES) --ptxas-options="-v"

# we're now targeting all major compute architectures within one binary.
.cu.o:
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=128 -o $@ -c $<

Algo256/blake256.o: Algo256/blake256.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=64 -o $@ -c $<

Algo256/cuda_bmw.o: Algo256/cuda_bmw.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=76 -o $@ -c $<

heavy/cuda_hefty1.o: heavy/cuda_hefty1.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

qubit/qubit_luffa512.o: qubit/qubit_luffa512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

# Luffa and Echo are faster with 80 registers than 128
x11/cuda_x11_luffa512.o: x11/cuda_x11_luffa512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

x11/cuda_x11_luffa512_Cubehash.o: x11/cuda_x11_luffa512_Cubehash.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=76 -o $@ -c $<

x11/cuda_x11_simd512.o: x11/cuda_x11_simd512.cu
	$(NVCC) $(nvcc_FLAGS) -Xcompiler -Wno-unused-variable -o $@ -c $<

x13/cuda_x13_hamsi512.o: x13/cuda_x13_hamsi512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=72 -o $@ -c $<

x17/cuda_x17_sha512.o: x17/cuda_x17_sha512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

quark/cuda_quark_blake512.o: quark/cuda_quark_blake512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

quark/cuda_quark_keccak512.o: quark/cuda_quark_keccak512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=88 -o $@ -c $<

quark/cuda_quark_compactionTest.o: quark/cuda_quark_compactionTest.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

JHA/cuda_jha_compactionTest.o: JHA/cuda_jha_compactionTest.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

# This object does not use cuda device code but call the different kernels (autotune)
scrypt/salsa_kernel.o: scrypt/salsa_kernel.cu
	$(NVCC) $(JANSSON_INCLUDES) -I. @CUDA_INCLUDES@ @CUDA_CFLAGS@ -gencode=arch=compute_20,code=\"sm_21,compute_20\" -o $@ -c $<

# These kernels are for older devices (SM)

scrypt/test_kernel.o: scrypt/test_kernel.cu
	$(NVCC) $(JANSSON_INCLUDES) -I. @CUDA_INCLUDES@ @CUDA_CFLAGS@ -gencode=arch=compute_20,code=\"sm_20,compute_20\" -o $@ -c $<

scrypt/fermi_kernel.o: scrypt/fermi_kernel.cu
	$(NVCC) $(JANSSON_INCLUDES) -I. @CUDA_INCLUDES@ @CUDA_CFLAGS@ -gencode=arch=compute_20,code=\"sm_21,compute_20\" -o $@ -c $<

scrypt/kepler_kernel.o: scrypt/kepler_kernel.cu
	$(NVCC) $(JANSSON_INCLUDES) -I. @CUDA_INCLUDES@ @CUDA_CFLAGS@ -gencode=arch=compute_30,code=\"sm_30,compute_30\" -o $@ -c $<

scrypt/nv_kernel.o: scrypt/nv_kernel.cu
	$(NVCC) $(JANSSON_INCLUDES) -I. @CUDA_INCLUDES@ @CUDA_CFLAGS@ -gencode=arch=compute_30,code=\"sm_30,compute_30\" -o $@ -c $<

scrypt/titan_kernel.o: scrypt/titan_kernel.cu
	$(NVCC) $(nvcc_FLAGS) -gencode=arch=compute_35,code=\"sm_35,compute_35\" -o $@ -c $<

skein.o: skein.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=64 -o $@ -c $<

skein2.o: skein2.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=64 -o $@ -c $<

