
ccminer 2.2.4 (Jan. 2018)     "lyra2v2 and keccak improvements"
---------------------------------------------------------------

***************************************************************
If you find this tool useful and like to support its continuous
          development, then consider a donation.

tpruvot@github:
  BTC  : 1AJdfCpLWPNoAMDfHF1wD5y8VgKSSTHxPo
  DCR  : DsUCcACGcyP8McNMRXQwbtpDxaVUYLDQDeU

DJM34:
  BTC donation address: 1NENYmxwZGHsKFmyjTc5WferTn5VTFb7Ze

cbuchner v1.2:
  LTC donation address: LKS1WDKGED647msBQfLBHV3Ls8sveGncnm
  BTC donation address: 16hJF5mceSojnTD3ZTUDqdRhDyPJzoRakM

***************************************************************

>>> Introduction <<<

This is a CUDA accelerated mining application which handle :

Decred (Blake256 14-rounds - 180 bytes)
HeavyCoin & MjollnirCoin
FugueCoin
GroestlCoin & Myriad-Groestl
Lbry Credits
JackpotCoin (JHA)
QuarkCoin family & AnimeCoin
TalkCoin
DarkCoin and other X11 coins
Chaincoin and Flaxscript (C11)
Saffroncoin blake (256 14-rounds)
BlakeCoin (256 8-rounds)
Qubit (Digibyte, ...)
Luffa (Joincoin)
Keccak (Maxcoin)
Pentablake (Blake 512 x5)
1Coin Triple S
Neoscrypt (FeatherCoin)
Revolver (X11evo)
Scrypt and Scrypt:N
Scrypt-Jane (Chacha)
Sibcoin (sib)
Skein (Skein + SHA)
Signatum (Skein cubehash fugue Streebog)
Tribus (JH, keccak, simd)
Woodcoin (Double Skein)
Vanilla (Blake256 8-rounds - double sha256)
Vertcoin Lyra2RE
Ziftrcoin (ZR5)
Boolberry (Wild Keccak)
Monero (Cryptonight)
Aeon (Cryptonight-lite)

where some of these coins have a VERY NOTABLE nVidia advantage
over competing AMD (OpenCL Only) implementations.

We did not take a big effort on improving usability, so please set
your parameters carefuly.

THIS PROGRAMM IS PROVIDED "AS-IS", USE IT AT YOUR OWN RISK!

If you're interessted and read the source-code, please excuse
that the most of our comments are in german.

>>> Command Line Interface <<<

This code is based on the pooler cpuminer and inherits
its command line interface and options.

  -a, --algo=ALGO       specify the algorithm to use
                          bastion     use to mine Joincoin
                          bitcore     use to mine Bitcore's Timetravel10
                          blake       use to mine Saffroncoin (Blake256)
                          blakecoin   use to mine Old Blake 256
                          blake2s     use to mine Nevacoin (Blake2-S 256)
                          bmw         use to mine Midnight
                          cryptolight use to mine AEON cryptonight (MEM/2)
                          cryptonight use to mine XMR cryptonight, Bytecoin, Dash, DigitalNote, etc
                          c11/flax    use to mine Chaincoin and Flax
                          decred      use to mine Decred 180 bytes Blake256-14
                          deep        use to mine Deepcoin
                          dmd-gr      use to mine Diamond-Groestl
                          equihash    use to mine ZEC, HUSH and KMD
                          fresh       use to mine Freshcoin
                          fugue256    use to mine Fuguecoin
                          groestl     use to mine Groestlcoin
                          hsr         use to mine Hshare
                          jackpot     use to mine Sweepcoin
                          keccak      use to mine Maxcoin
                          keccakc     use to mine CreativeCoin
                          lbry        use to mine LBRY Credits
                          luffa       use to mine Joincoin
                          lyra2       use to mine CryptoCoin
                          lyra2v2     use to mine Vertcoin
                          lyra2z      use to mine Zerocoin (XZC)
                          myr-gr      use to mine Myriad-Groest
                          neoscrypt   use to mine FeatherCoin, Trezarcoin, Orbitcoin, etc
                          nist5       use to mine TalkCoin
                          penta       use to mine Joincoin / Pentablake
                          phi         use to mine LUXCoin
                          polytimos   use to mine Polytimos
                          quark       use to mine Quarkcoin
                          qubit       use to mine Qubit
                          scrypt      use to mine Scrypt coins (Litecoin, Dogecoin, etc)
                          scrypt:N    use to mine Scrypt-N (:10 for 2048 iterations)
                          scrypt-jane use to mine Chacha coins like Cache and Ultracoin
                          s3          use to mine 1coin (ONE)
                          sha256t     use to mine OneCoin (OC)
                          sia         use to mine SIA
                          sib         use to mine Sibcoin
                          skein       use to mine Skeincoin
                          skein2      use to mine Woodcoin
                          skunk       use to mine Signatum
                          timetravel  use to mine MachineCoin
                          tribus      use to mine Denarius
                          x11evo      use to mine Revolver
                          x11         use to mine DarkCoin
                          x14         use to mine X14Coin
                          x15         use to mine Halcyon
                          x17         use to mine X17
                          vanilla     use to mine Vanilla (Blake256)
                          veltor      use to mine VeltorCoin
                          whirlpool   use to mine Joincoin
                          wildkeccak  use to mine Boolberry (Stratum only)
                          zr5         use to mine ZiftrCoin

  -d, --devices         gives a comma separated list of CUDA device IDs
                        to operate on. Device IDs start counting from 0!
                        Alternatively give string names of your card like
                        gtx780ti or gt640#2 (matching 2nd gt640 in the PC).

  -i, --intensity=N[,N] GPU threads per call 8-25 (2^N + F, default: 0=auto)
                        Decimals and multiple values are allowed for fine tuning
      --cuda-schedule   Set device threads scheduling mode (default: auto)
  -f, --diff-factor     Divide difficulty by this factor (default 1.0)
  -m, --diff-multiplier Multiply difficulty by this value (default 1.0)
  -o, --url=URL         URL of mining server
  -O, --userpass=U:P    username:password pair for mining server
  -u, --user=USERNAME   username for mining server
  -p, --pass=PASSWORD   password for mining server
      --cert=FILE       certificate for mining server using SSL
  -x, --proxy=[PROTOCOL://]HOST[:PORT]  connect through a proxy
  -t, --threads=N       number of miner threads (default: number of nVidia GPUs in your system)
  -r, --retries=N       number of times to retry if a network call fails
                          (default: retry indefinitely)
  -R, --retry-pause=N   time to pause between retries, in seconds (default: 15)
      --shares-limit    maximum shares to mine before exiting the program.
      --time-limit      maximum time [s] to mine before exiting the program.
  -T, --timeout=N       network timeout, in seconds (default: 300)
  -s, --scantime=N      upper bound on time spent scanning current work when
                        long polling is unavailable, in seconds (default: 5)
      --submit-stale    ignore stale job checks, may create more rejected shares
  -n, --ndevs           list cuda devices
  -N, --statsavg        number of samples used to display hashrate (default: 30)
      --no-gbt          disable getblocktemplate support (height check in solo)
      --no-longpoll     disable X-Long-Polling support
      --no-stratum      disable X-Stratum support
  -q, --quiet           disable per-thread hashmeter output
      --no-color        disable colored output
  -D, --debug           enable debug output
  -P, --protocol-dump   verbose dump of protocol-level activities
  -b, --api-bind=port   IP:port for the miner API (default: 127.0.0.1:4068), 0 disabled
      --api-remote      Allow remote control, like pool switching, imply --api-allow=0/0
      --api-allow=...   IP/mask of the allowed api client(s), 0/0 for all
      --max-temp=N      Only mine if gpu temp is less than specified value
      --max-rate=N[KMG] Only mine if net hashrate is less than specified value
      --max-diff=N      Only mine if net difficulty is less than specified value
      --max-log-rate    Interval to reduce per gpu hashrate logs (default: 3)
      --pstate=0        will force the Geforce 9xx to run in P0 P-State
      --plimit=150W     set the gpu power limit, allow multiple values for N cards
                          on windows this parameter use percentages (like OC tools)
      --tlimit=85       Set the gpu thermal limit (windows only)
      --keep-clocks     prevent reset clocks and/or power limit on exit
      --hide-diff       Hide submitted shares diff and net difficulty
  -B, --background      run the miner in the background
      --benchmark       run in offline benchmark mode
      --cputest         debug hashes from cpu algorithms
      --cpu-affinity    set process affinity to specific cpu core(s) mask
      --cpu-priority    set process priority (default: 0 idle, 2 normal to 5 highest)
  -c, --config=FILE     load a JSON-format configuration file
                        can be from an url with the http:// prefix
  -V, --version         display version information and exit
  -h, --help            display this help text and exit


Scrypt specific options:
  -l, --launch-config   gives the launch configuration for each kernel
                        in a comma separated list, one per device.
      --interactive     comma separated list of flags (0/1) specifying
                        which of the CUDA device you need to run at inter-
                        active frame rates (because it drives a display).
  -L, --lookup-gap      Divides the per-hash memory requirement by this factor
                        by storing only every N'th value in the scratchpad.
                        Default is 1.
      --texture-cache   comma separated list of flags (0/1/2) specifying
                        which of the CUDA devices shall use the texture
                        cache for mining. Kepler devices may profit.
      --no-autotune     disable auto-tuning of kernel launch parameters

CryptoNight specific options:
  -l, --launch-config   gives the launch configuration for each kernel
                        in a comma separated list, one per device.
      --bfactor=[0-12]  Run Cryptonight core kernel in smaller pieces,
                        From 0 (ui freeze) to 12 (smooth), win default is 11
                        This is a per-device setting like the launch config.

Wildkeccak specific:
  -l, --launch-config   gives the launch configuration for each kernel
                        in a comma separated list, one per device.
  -k, --scratchpad url  Url used to download the scratchpad cache.


>>> Examples <<<


Example for Heavycoin Mining on heavycoinpool.com with a single gpu in your system
    ccminer -t 1 -a heavy -o stratum+tcp://stratum01.heavycoinpool.com:5333 -u <<username.worker>> -p <<workerpassword>> -v 8


Example for Heavycoin Mining on hvc.1gh.com with a dual gpu in your system
    ccminer -t 2 -a heavy -o stratum+tcp://hvcpool.1gh.com:5333/ -u <<WALLET>> -p x -v 8


Example for Fuguecoin solo-mining with 4 gpu's in your system and a Fuguecoin-wallet running on localhost
    ccminer -q -s 1 -t 4 -a fugue256 -o http://localhost:9089/ -u <<myusername>> -p <<mypassword>>


Example for Fuguecoin pool mining on dwarfpool.com with all your GPUs
    ccminer -q -a fugue256 -o stratum+tcp://erebor.dwarfpool.com:3340/ -u YOURWALLETADDRESS.1 -p YOUREMAILADDRESS


Example for Groestlcoin solo mining
    ccminer -q -s 1 -a groestl -o http://127.0.0.1:1441/ -u USERNAME -p PASSWORD

Example for Boolberry
    ccminer -a wildkeccak -o stratum+tcp://bbr.suprnova.cc:7777 -u tpruvot.donate -p x -k http://bbr.suprnova.cc/scratchpad.bin -l 64x360

Example for Scrypt-N (2048) on Nicehash
    ccminer -a scrypt:10 -o stratum+tcp://stratum.nicehash.com:3335 -u 3EujYFcoBzWvpUEvbe3obEG95mBuU88QBD -p x

For solo-mining you typically use -o http://127.0.0.1:xxxx where xxxx represents
the rpcport number specified in your wallet's .conf file and you have to pass the same username
and password with -O (or -u -p) as specified in the wallet config.

The wallet must also be started with the -server option and/or with the server=1 flag in the .conf file

>>> Configuration files <<<

With the -c parameter you can use a json config file to set your prefered settings.
An example is present in source tree, and is also the default one when no command line parameters are given.
This allow you to run the miner without batch/script.


>>> API and Monitoring <<<

With the -b parameter you can open your ccminer to your network, use -b 0.0.0.0:4068 if required.
On windows, setting 0.0.0.0 will ask firewall permissions on the first launch. Its normal.

Default API feature is only enabled for localhost queries by default, on port 4068.

You can test this api on linux with "telnet <miner-ip> 4068" and type "help" to list the commands.
Default api format is delimited text. If required a php json wrapper is present in api/ folder.

I plan to add a json format later, if requests are formatted in json too..


>>> Additional Notes <<<

This code should be running on nVidia GPUs ranging from compute capability
3.0 up to compute capability 5.2. Support for Compute 2.0 has been dropped
so we can more efficiently implement new algorithms using the latest hardware
features.

>>> RELEASE HISTORY <<<
  Jan. 04th 2017  v2.2.4
                  Improve lyra2v2
                  Higher keccak default intensity
                  Drop SM 2.x support by default, for CUDA 9 and more recent

  Dec. 04th 2017  v2.2.3
                  Polytimos Algo
                  Handle keccakc variant (with refreshed sha256d merkle)
                  Optimised keccak for SM5+, based on alexis improvements

  Oct. 09th 2017  v2.2.2
                  Import and clean the hsr algo (x13 + custom hash)
                  Import and optimise phi algo from LuxCoin repository
                  Improve sib algo too for maxwell and pascal cards
                  Small fix to handle more than 9 cards on linux (-d 10+)
                  Attempt to free equihash memory "properly"
                  --submit-stale parameter for supernova pool (which change diff too fast)

  Sep. 01st 2017  v2.2.1
                  Improve tribus algo on recent cards (up to +10%)

  Aug. 13th 2017  v2.2
                  New skunk algo, using the heavy streebog algorithm
                  Enhance tribus algo (+10%)
                  equihash protocol enhancement on yiimp.ccminer.org and zpool.ca

  June 16th 2017  v2.1-tribus
                  Interface equihash algo with djeZo solver (from nheqminer 0.5c)
                  New api parameters (and multicast announces for local networks)
                  New tribus algo

  May. 14th 2017  v2.0
                  Handle cryptonight, wildkeccak and cryptonight-lite
                  Add a serie of new algos: timetravel, bastion, hmq1725, sha256t
                  Import lyra2z from djm34 work...
                  Rework the common skein512 (used in most algos except skein ;)
                  Upgrade whirlpool algo with alexis version (2x faster)
                  Store the share diff of second nonce(s) in most algos
                  Hardware monitoring thread to get more accurate power readings
                  Small changes for the quiet mode & max-log-rate to reduce logs
                  Add bitcore and a compatible jha algo

  Dec. 21th 2016  v1.8.4
                  Improve streebog based algos, veltor and sib (from alexis work)
                  Blake2s greetly improved (3x), thanks to alexis too...

  Sep. 28th 2016  v1.8.3
                  show intensity on startup for each cards
                  show-diff is now used by default, use --hide-diff if not wanted

  Sep. 22th 2016  v1.8.2
                  lbry improvements by Alexis Provos
                  Prevent Windows hibernate while mining
                  veltor algo (basic implementation)

  Aug. 10th 2016  v1.8.1
                  SIA Blake2-B Algo (getwork over stratum for Suprnova)
                  SIA Nanopool RPC (getwork over http)
                  Update also the older lyra2 with Nanashi version

  July 20th 2016  v1.8.0
                  Pascal support with cuda 8
                  lbry new multi sha / ripemd algo (LBC)
                  x11evo algo (XRE)
                  Lyra2v2, Neoscrypt and Decred improvements
                  Enhance windows NVAPI clock and power limits
                  Led support for mining/shares activity on windows

  May  18th 2016  v1.7.6
                  Decred vote support
                  X17 cleanup and improvement
                  Add mining.ping stratum method and handle unknown methods
                  Implement a pool stats/benchmark mode (-p stats on yiimp)
                  Add --shares-limit parameter, can be used for benchmarks

  Mar. 13th 2016  v1.7.5
                  Blake2S Algo (NEVA/OXEN)

  Feb. 28th 2016  v1.7.4 (1.7.3 was a preview, not official)
                  Decred simplified stratum (getwork over stratum)
                  Vanilla kernel by MrMad
                  Drop/Disable WhirlpoolX

  Feb. 11th 2016  v1.7.2
                  Decred Algo (longpoll only)
                  Blake256 improvements/cleanup

  Jan. 26th 2016  v1.7.1
                  Implement sib algo (X11 + Russian Streebog-512/GOST)
                  Whirlpool speed x2 with the midstate precompute
                  Small bug fixes about device ids mapping (and vendor names)
                  Add Vanilla algo (Blake256 8-rounds - double sha256)

  Nov. 06th 2015  v1.7
                  Improve old devices compatibility (x11, lyra2v2, quark, qubit...)
                  Add windows support for SM 2.1 and drop SM 3.5 (x86)
                  Improve lyra2 (v1/v2) cuda implementations
                  Improve most common algos on SM5+ with sp blake kernel
                  Restore whirlpool algo (and whirlcoin variant)
                  Prepare algo/pool switch ability, trivial method
                  Add --benchmark alone to run a benchmark for all algos
                  Add --cuda-schedule parameter
                  Add --show-diff parameter, which display shares diff,
                    and is able to detect real solved blocks on pools.

  Aug. 28th 2015  v1.6.6
                  Allow to load remote config with curl (-c http://...)
                  Add Lyra2REv2 algo (Vertcoin/Zoom)
                  Restore WhirlpoolX algo (VNL)
                  Drop Animecoin support
                  Add bmw (Midnight) algo

  July 06th 2015  v1.6.5-C11
                  Nvml api power limits
                  Add chaincoin c11 algo (used by Flaxscript too)
                  Remove pluck algo

  June 23th 2015  v1.6.5
                  Handle Ziftrcoin PoK solo mining
                  Basic compatibility with CUDA 7.0 (generally slower hashrate)
                  Show gpus vendor names on linux (windows test branch is pciutils)
                  Remove -v and -m short params specific to heavycoin
                  Add --diff-multiplier (-m) and rename --diff to --diff-factor (-f)
                  First steps to handle nvml application clocks and P0 on the GTX9xx
                  Various improvements on multipool and cmdline parameters
                  Optimize a bit qubit, deep, luffa, x11 and quark algos

  May 26th 2015   v1.6.4
                  Implement multi-pool support (failover and time rotate)
                    try "ccminer -c pools.conf" to test the sample config
                  Update the API to allow remote pool switching and pool stats
                  Auto bind the api port to the first available when using default
                  Try to compute network difficulty on pools too (for most algos)
                  Drop Whirlpool and whirpoolx algos, no more used...

  May 15th 2015   v1.6.3
                  Import and adapt Neoscrypt from djm34 work (SM 5+ only)
                  Conditional mining options based on gpu temp, network diff and rate
                  background option implementation for windows too
                  "Multithreaded" devices (-d 0,0) intensity and stats changes
                  SM5+ Optimisation of skein based on sp/klaus method (+20%)

  Apr. 21th 2015  v1.6.2
                  Import Scrypt, Scrypt:N and Scrypt-jane from Cudaminer
                  Add the --time-limit command line parameter

  Apr. 14th 2015  v1.6.1
                  Add the Double Skein Algo for Woodcoin
                  Skein/Skein2 SM 3.0 devices support

  Mar. 27th 2015  v1.6.0
                  Add the ZR5 Algo for Ziftcoin
                  Implement Skeincoin algo (skein + sha)
                  Import pluck (djm34) and whirlpoolx (alexis78) algos
                  Hashrate units based on hashing rate values (Hs/kHs/MHs/GHs)
                  Default config file (also help to debug without command line)
                  Various small fixes

  Feb. 11th 2015  v1.5.3
                  Fix anime algo
                  Allow a default config file in user or ccminer folder
                  SM 2.1 windows binary (lyra2 and blake/blakecoin for the moment)

  Jan. 24th 2015  v1.5.2
                  Allow per device intensity, example: -i 20,19.5
                  Add process CPU priority and affinity mask parameters
                  Intelligent duplicate shares check feature (enabled if needed)
                  api: Fan RPM (windows), Cuda threads count, linux kernel ver.
                  More X11 optimisations from sp and KlausT
                  SM 3.0 enhancements

  Dec. 16th 2014  v1.5.1
                  Add lyra2RE algo for Vertcoin based on djm34/vtc code
                  Multiple shares support (2 for the moment)
                  X11 optimisations (From klaust and sp-hash)
                  HTML5 WebSocket api compatibility (see api/websocket.htm)
                  Solo mode height checks with getblocktemplate rpc calls

  Nov. 27th 2014  v1.5.0
                  Upgrade compat jansson to 2.6 (for windows)
                  Add pool mining.set_extranonce support
                  Allow intermediate intensity with decimals
                  Update prebuilt x86 openssl lib to 1.0.1i
                  Fix heavy algo on linux (broken since 1.4)
                  Some internal changes to use the C++ compiler
                  New API 1.2 with some new commands (read only)
                  Add some of sp x11/x15 optimisations (and tsiv x13)

  Nov. 15th 2014  v1.4.9
                  Support of nvml and nvapi(windows) to monitor gpus
                  Fix (again) displayed hashrate for multi gpus systems
                    Average is now made by card (30 scans of the card)
                  Final API v1.1 (new fields + histo command)
                  Add support of telnet queries "telnet 127.0.0.1 4068"
                  add histo api command to get performance debug details
                  Add a rig sample php ui using json wrapper (php)
                  Restore quark/jackpot previous speed (differently)

  Nov. 12th 2014  v1.4.8
                  Add a basic API and a sample php json wrapper
                  Add statsavg (def 20) and api-bind parameters

  Nov. 11th 2014  v1.4.7
                  Average hashrate (based on the 20 last scans)
                  Rewrite blake algo
                  Add the -i (gpu threads/intensity parameter)
                  Add some X11 optimisations based on sp_ commits
                  Fix quark reported hashrate and benchmark mode for some algos
                  Enhance json config file param (int/float/false) (-c config.json)
                  Update windows prebuilt curl to 7.38.0

  Oct. 26th 2014  v1.4.6
                  Add S3 algo reusing existing code (onecoin)
                  Small X11 (simd512) enhancement

  Oct. 20th 2014  v1.4.5
                  Add keccak algo from djm34 repo (maxcoin)
                  Curl 7.35 and OpenSSL are now included in the binary (and win tree)
                  Enhance windows terminal support (--help was broken)

  Sep. 27th 2014  v1.4.4
                  First SM 5.2 Release (GTX 970 & 980)
                  CUDA Runtime included in binary
                  Colors enabled by default

  Sep. 10th 2014  v1.4.3
                  Add algos from djm34 repo (deep, doom, qubit)
                  Goalcoin seems to be dead, not imported.
                  Create also the pentablake algo (5x Blake 512)

  Sept  6th 2014  Almost twice the speed on blake256 algos with the "midstate" cache

  Sep.  1st 2014  add X17, optimized x15 and whirl
                  add blake (256 variant)
                  color support on Windows,
                  remove some dll dependencies (pthreads, msvcp)

  Aug. 18th 2014  add X14, X15, Whirl, and Fresh algos,
                  also add colors and nvprof cmd line support

  June 15th 2014  add X13 and Diamond Groestl support.
                  Thanks to tsiv and to Bombadil for the contributions!

  June 14th 2014  released Killer Groestl quad version which I deem
                  sufficiently hard to port over to AMD. It isn't
                  the fastest option for Compute 3.5 and 5.0 cards,
                  but it is still much faster than the table based
                  versions.

  May 10th 2014   added X11, but without the bells & whistles
                  (no killer Groestl, SIMD hash quite slow still)

  May 6th 2014    this adds the quark and animecoin algorithms.

  May 3rd 2014    add the MjollnirCoin hash algorithm for the upcomin
                  MjollnirCoin relaunch.

                  Add the -f (--diff) option to adjust the difficulty
                  e.g. for the erebor Dwarfpool myr-gr SaffronCoin pool.
                  Use -f 256 there.

  May 1st 2014    adapt the Jackpot algorithms to changes made by the
                  coin developers. We keep our unique nVidia advantage
                  because we have a way to break up the divergence.
                  NOTE: Jackpot Hash now requires Compute 3.0 or later.

  April, 27 2014  this release adds Myriad-Groestl and Jackpot Coin.
                  we apply an optimization to Jackpot that turns this
                  into a Keccak-only CUDA coin ;) Jackpot is tested with
                  solo--mining only at the moment.

  March, 27 2014  Heavycoin exchange rates soar, and as a result this coin
                  gets some love: We greatly optimized the Hefty1 kernel
                  for speed. Expect some hefty gains, especially on 750Ti's!

                  By popular demand, we added the -d option as known from
                  cudaminer.

                  different compute capability builds are now provided until
                  we figure out how to pack everything into a single executable
                  in a Windows build.

  March, 24 2014  fixed Groestl pool support

                  went back to Compute 1.x for cuda_hefty1.cu kernel by
                  default after numerous reports of ccminer v0.2/v0.3
                  not working with HeavyCoin for some people.

  March, 23 2014  added Groestlcoin support. stratum status unknown
                  (the only pool is currently down for fixing issues)

  March, 21 2014  use of shared memory in Fugue256 kernel boosts hash rates
                  on Fermi and Maxwell devices. Kepler may suffer slightly
                  (3-5%)

                  Fixed Stratum for Fuguecoin. Tested on dwarfpool.

  March, 18 2014  initial release.


>>> AUTHORS <<<

Notable contributors to this application are:

Christian Buchner, Christian H. (Germany): Initial CUDA implementation

djm34, tsiv, sp and klausT for cuda algos implementation and optimisation

Tanguy Pruvot : 750Ti tuning, blake, colors, zr5, skein, general code cleanup
                API monitoring, linux Config/Makefile and vstudio libs...

and also many thanks to anyone else who contributed to the original
cpuminer application (Jeff Garzik, pooler), it's original HVC-fork
and the HVC-fork available at hvc.1gh.com

Source code is included to satisfy GNU GPL V3 requirements.


With kind regards,

   Christian Buchner ( Christian.Buchner@gmail.com )
   Christian H. ( Chris84 )
   Tanguy Pruvot ( tpruvot@github )
