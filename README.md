ccminer
=======

Based on Christian Buchner's &amp; Christian H.'s CUDA project, no more active on github recently.

Fork by tpruvot@github with X14,X15,X17,Blake256,BlakeCoin,Lyra2RE,Skein,ZR5 and others, check the [README.txt](README.txt)

   BTC donation address: 1AJdfCpLWPNoAMDfHF1wD5y8VgKSSTHxPo
   [![tip for next commit](https://tip4commit.com/projects/927.svg)](https://tip4commit.com/github/tpruvot/ccminer)

A part of the recent algos were originally written by [djm34](https://github.com/djm34).

This variant was tested and built on Linux (ubuntu server 14.04) and VStudio 2013 on Windows 7.

Note that the x86 releases are generally faster than x64 ones on Windows.

The recommended CUDA Toolkit version is [6.5.19](http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.19_windows_general_64.exe), but some light algos could be faster with the version 7.5 (like blake and skein).

About source code dependencies
------------------------------

This project requires some libraries to be built :

- OpenSSL (prebuilt for win)

- Curl (prebuilt for win)

- pthreads (prebuilt for win)

The tree now contains recent prebuilt openssl and curl .lib for both x86 and x64 platforms (windows).

To rebuild them, you need to clone this repository and its submodules :
    git clone https://github.com/peters/curl-for-windows.git compat/curl-for-windows

On Linux, you can use the helper ./build.sh (edit it if required)

There is also an old [Tutorial for windows](http://cudamining.co.uk/url/tutorials/id/3) on [CudaMining](http://cudamining.co.uk) website.

CUDA
-------------------------------

See [CUDA Compatibility](CUDA_Compatibility.md) to ensure that your [Makefile.am](Makefile.am) will give you optimal results.

