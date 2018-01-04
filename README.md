# ccminer

Based on Christian Buchner's &amp; Christian H.'s CUDA project, no more active on github since 2014.

Check the [README.txt](README.txt) for the additions

BTC donation address: 1AJdfCpLWPNoAMDfHF1wD5y8VgKSSTHxPo (tpruvot)

BTC donation address: 1a2gWsePbgC7DNCN6yNFWqHAPotvpyXLN  (gelotus)


A part of the recent algos were originally written by [djm34](https://github.com/djm34) and [alexis78](https://github.com/alexis78)

It is built for Windows 7 to 10 with VStudio 2015.

The recommended CUDA Toolkit version was the [9.1](https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda_9.1.85_win10).

About source code dependencies
------------------------------

This project requires some libraries to be built :

- OpenSSL
- Curl
- pthreads

Those are included in the build project.

This branch was only tested in windows, it was compiled in Visual Studio 2017 with 2015 SDK, CUDA 9.1.
The libraries are all at the last stable version.
All was compiled with multithread support and dll crt linking.

Some tests show a little improvement:

Tested for 5 mins on a stock clocked Palit GeForce 750Ti against X11 algo:

VC2015 CUDA 9.1 OpenSSL 1.1.0 -> 2862.65 kH/s
VC2013 binary release         -> 2847.10 kH/s

# Building

Open ccminer.sln in VS2015 and compile, the release archive will be in ./dist folder.