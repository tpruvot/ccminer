AC_INIT([ccminer], [2.2.4.1-x16r], [], [ccminer], [http://github.com/todd1251/ccminer-x16r])

AC_PREREQ([2.59c])
AC_CANONICAL_SYSTEM
AC_CONFIG_SRCDIR([ccminer.cpp])
AM_INIT_AUTOMAKE([foreign subdir-objects])
AC_CONFIG_HEADERS([ccminer-config.h])

dnl Make sure anyone changing configure.ac/Makefile.am has a clue
AM_MAINTAINER_MODE

dnl Checks for programs
AC_PROG_CC_C99
AC_PROG_GCC_TRADITIONAL
AM_PROG_CC_C_O
AM_PROG_AS
AC_PROG_RANLIB
AC_PROG_CXX
AC_OPENMP

dnl Checks for header files
AC_HEADER_STDC
AC_CHECK_HEADERS([sys/endian.h sys/param.h syslog.h])
# sys/sysctl.h requires sys/types.h on FreeBSD
# sys/sysctl.h requires sys/param.h on OpenBSD
AC_CHECK_HEADERS([sys/sysctl.h], [], [],
[#include <sys/types.h>
#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif
])

AC_CHECK_DECLS([be32dec, le32dec, be32enc, le32enc], [], [],
[AC_INCLUDES_DEFAULT
#ifdef HAVE_SYS_ENDIAN_H
#include <sys/endian.h>
#endif
])

AC_FUNC_ALLOCA
AC_CHECK_FUNCS([getopt_long])

case $target in
  i*86-*-*)
    have_x86=true
    ;;
  x86_64-*-*)
    have_x86=true
    have_x86_64=true
    ;;
  amd64-*-*)
    have_x86=true
    have_x86_64=true
    ;;
esac

PTHREAD_FLAGS="-pthread"
WS2_LIBS=""

case $target in
  *-*-mingw*)
    have_win32=true
    PTHREAD_FLAGS=""
    WS2_LIBS="-lws2_32"
    ;;
  *86*-apple-darwin*)
    have_osx=true
    PTHREAD_FLAGS=""
    ;;
esac

AC_CHECK_LIB(jansson, json_loads, request_jansson=false, request_jansson=true)
AC_CHECK_LIB([pthread], [pthread_create], PTHREAD_LIBS="-lpthread",
  AC_CHECK_LIB([pthreadGC2], [pthread_create], PTHREAD_LIBS="-lpthreadGC2",
    AC_CHECK_LIB([pthreadGC1], [pthread_create], PTHREAD_LIBS="-lpthreadGC1",
      AC_CHECK_LIB([pthreadGC], [pthread_create], PTHREAD_LIBS="-lpthreadGC"
))))

AC_CHECK_LIB([z],[gzopen], [], [])
AC_CHECK_LIB([ssl],[SSL_free], [], [AC_MSG_ERROR([OpenSSL library required])])
AC_CHECK_LIB([crypto],[EVP_DigestFinal_ex], [], [AC_MSG_ERROR([OpenSSL library required])])

AM_CONDITIONAL([WANT_JANSSON], [test x$request_jansson = xtrue])
AM_CONDITIONAL([HAVE_WINDOWS], [test x$have_win32 = xtrue])
AM_CONDITIONAL([HAVE_OSX], [test x$have_osx = xtrue])
AM_CONDITIONAL([ARCH_x86], [test x$have_x86 = xtrue])
AM_CONDITIONAL([ARCH_x86_64], [test x$have_x86_64 = xtrue])

if test x$request_jansson = xtrue
then
	JANSSON_LIBS="compat/jansson/libjansson.a"
else
	JANSSON_LIBS=-ljansson
fi

LIBCURL_CHECK_CONFIG(, 7.15.2, ,
  [AC_MSG_ERROR([Missing required libcurl >= 7.15.2])])

AC_SUBST(JANSSON_LIBS)
AC_SUBST(PTHREAD_FLAGS)
AC_SUBST(PTHREAD_LIBS)
AC_SUBST(WS2_LIBS)

AC_CONFIG_FILES([
	Makefile
	compat/Makefile
	compat/jansson/Makefile
	])

dnl find out what version we are running
ARCH=`uname -m`
if [[ $ARCH == "x86_64" ]];
then
  SUFFIX="64"
else
  SUFFIX=""
fi

if test x$have_osx = xtrue
then
  SUFFIX=""
fi

dnl Setup CUDA paths
AC_ARG_WITH([cuda],
   [  --with-cuda=PATH    prefix where cuda is installed [default=/usr/local/cuda]])

AC_ARG_WITH([nvml],
   [  --with-nvml=PATH    prefix where libnvml is installed [default=/usr/lib]])

AM_CONDITIONAL([HAVE_NVML], [test -n "$with_nvml"])

NVCC="nvcc"

if test -n "$with_cuda" ; then
  CUDA_INCLUDES="-I$with_cuda/include"
  CUDA_LIBS="-lcudart"
  CUDA_LDFLAGS="-L$with_cuda/lib$SUFFIX"
  if test -x "$with_cuda/bin/nvcc" ; then
    NVCC="$with_cuda/bin/nvcc"
  fi
else
  CUDA_INCLUDES="-I/usr/local/cuda/include"
  CUDA_LIBS="-lcudart -static-libstdc++"
  CUDA_LDFLAGS="-L/usr/local/cuda/lib$SUFFIX"
fi

if test -n "$with_nvml" ; then
  NVML_LIBPATH=$with_nvml
  CUDA_LDFLAGS="$CUDA_LDFLAGS -ldl"
fi
AC_SUBST(NVML_LIBPATH)

AC_SUBST(CUDA_CFLAGS)
AC_SUBST(CUDA_INCLUDES)
AC_SUBST(CUDA_LIBS)
AC_SUBST(CUDA_LDFLAGS)
AC_SUBST(NVCC)

AC_SUBST(OPENMP_CFLAGS) 

AC_OUTPUT
