
#define N_COLS          4
#define WPOLY           0x011b
/*
static const __device__ __align__(16) uint32_t d_t_fn[1024] = {
  0xa56363c6U, 0x847c7cf8U, 0x997777eeU, 0x8d7b7bf6U, 0x0df2f2ffU, 0xbd6b6bd6U, 0xb16f6fdeU, 0x54c5c591U,
  0x50303060U, 0x03010102U, 0xa96767ceU, 0x7d2b2b56U, 0x19fefee7U, 0x62d7d7b5U, 0xe6abab4dU, 0x9a7676ecU,
  0x45caca8fU, 0x9d82821fU, 0x40c9c989U, 0x877d7dfaU, 0x15fafaefU, 0xeb5959b2U, 0xc947478eU, 0x0bf0f0fbU,
  0xecadad41U, 0x67d4d4b3U, 0xfda2a25fU, 0xeaafaf45U, 0xbf9c9c23U, 0xf7a4a453U, 0x967272e4U, 0x5bc0c09bU,
  0xc2b7b775U, 0x1cfdfde1U, 0xae93933dU, 0x6a26264cU, 0x5a36366cU, 0x413f3f7eU, 0x02f7f7f5U, 0x4fcccc83U,
  0x5c343468U, 0xf4a5a551U, 0x34e5e5d1U, 0x08f1f1f9U, 0x937171e2U, 0x73d8d8abU, 0x53313162U, 0x3f15152aU,
  0x0c040408U, 0x52c7c795U, 0x65232346U, 0x5ec3c39dU, 0x28181830U, 0xa1969637U, 0x0f05050aU, 0xb59a9a2fU,
  0x0907070eU, 0x36121224U, 0x9b80801bU, 0x3de2e2dfU, 0x26ebebcdU, 0x6927274eU, 0xcdb2b27fU, 0x9f7575eaU,
  0x1b090912U, 0x9e83831dU, 0x742c2c58U, 0x2e1a1a34U, 0x2d1b1b36U, 0xb26e6edcU, 0xee5a5ab4U, 0xfba0a05bU,
  0xf65252a4U, 0x4d3b3b76U, 0x61d6d6b7U, 0xceb3b37dU, 0x7b292952U, 0x3ee3e3ddU, 0x712f2f5eU, 0x97848413U,
  0xf55353a6U, 0x68d1d1b9U, 0x00000000U, 0x2cededc1U, 0x60202040U, 0x1ffcfce3U, 0xc8b1b179U, 0xed5b5bb6U,
  0xbe6a6ad4U, 0x46cbcb8dU, 0xd9bebe67U, 0x4b393972U, 0xde4a4a94U, 0xd44c4c98U, 0xe85858b0U, 0x4acfcf85U,
  0x6bd0d0bbU, 0x2aefefc5U, 0xe5aaaa4fU, 0x16fbfbedU, 0xc5434386U, 0xd74d4d9aU, 0x55333366U, 0x94858511U,
  0xcf45458aU, 0x10f9f9e9U, 0x06020204U, 0x817f7ffeU, 0xf05050a0U, 0x443c3c78U, 0xba9f9f25U, 0xe3a8a84bU,
  0xf35151a2U, 0xfea3a35dU, 0xc0404080U, 0x8a8f8f05U, 0xad92923fU, 0xbc9d9d21U, 0x48383870U, 0x04f5f5f1U,
  0xdfbcbc63U, 0xc1b6b677U, 0x75dadaafU, 0x63212142U, 0x30101020U, 0x1affffe5U, 0x0ef3f3fdU, 0x6dd2d2bfU,
  0x4ccdcd81U, 0x140c0c18U, 0x35131326U, 0x2fececc3U, 0xe15f5fbeU, 0xa2979735U, 0xcc444488U, 0x3917172eU,
  0x57c4c493U, 0xf2a7a755U, 0x827e7efcU, 0x473d3d7aU, 0xac6464c8U, 0xe75d5dbaU, 0x2b191932U, 0x957373e6U,
  0xa06060c0U, 0x98818119U, 0xd14f4f9eU, 0x7fdcdca3U, 0x66222244U, 0x7e2a2a54U, 0xab90903bU, 0x8388880bU,
  0xca46468cU, 0x29eeeec7U, 0xd3b8b86bU, 0x3c141428U, 0x79dedea7U, 0xe25e5ebcU, 0x1d0b0b16U, 0x76dbdbadU,
  0x3be0e0dbU, 0x56323264U, 0x4e3a3a74U, 0x1e0a0a14U, 0xdb494992U, 0x0a06060cU, 0x6c242448U, 0xe45c5cb8U,
  0x5dc2c29fU, 0x6ed3d3bdU, 0xefacac43U, 0xa66262c4U, 0xa8919139U, 0xa4959531U, 0x37e4e4d3U, 0x8b7979f2U,
  0x32e7e7d5U, 0x43c8c88bU, 0x5937376eU, 0xb76d6ddaU, 0x8c8d8d01U, 0x64d5d5b1U, 0xd24e4e9cU, 0xe0a9a949U,
  0xb46c6cd8U, 0xfa5656acU, 0x07f4f4f3U, 0x25eaeacfU, 0xaf6565caU, 0x8e7a7af4U, 0xe9aeae47U, 0x18080810U,
  0xd5baba6fU, 0x887878f0U, 0x6f25254aU, 0x722e2e5cU, 0x241c1c38U, 0xf1a6a657U, 0xc7b4b473U, 0x51c6c697U,
  0x23e8e8cbU, 0x7cdddda1U, 0x9c7474e8U, 0x211f1f3eU, 0xdd4b4b96U, 0xdcbdbd61U, 0x868b8b0dU, 0x858a8a0fU,
  0x907070e0U, 0x423e3e7cU, 0xc4b5b571U, 0xaa6666ccU, 0xd8484890U, 0x05030306U, 0x01f6f6f7U, 0x120e0e1cU,
  0xa36161c2U, 0x5f35356aU, 0xf95757aeU, 0xd0b9b969U, 0x91868617U, 0x58c1c199U, 0x271d1d3aU, 0xb99e9e27U,
  0x38e1e1d9U, 0x13f8f8ebU, 0xb398982bU, 0x33111122U, 0xbb6969d2U, 0x70d9d9a9U, 0x898e8e07U, 0xa7949433U,
  0xb69b9b2dU, 0x221e1e3cU, 0x92878715U, 0x20e9e9c9U, 0x49cece87U, 0xff5555aaU, 0x78282850U, 0x7adfdfa5U,
  0x8f8c8c03U, 0xf8a1a159U, 0x80898909U, 0x170d0d1aU, 0xdabfbf65U, 0x31e6e6d7U, 0xc6424284U, 0xb86868d0U,
  0xc3414182U, 0xb0999929U, 0x772d2d5aU, 0x110f0f1eU, 0xcbb0b07bU, 0xfc5454a8U, 0xd6bbbb6dU, 0x3a16162cU,

  0x6363c6a5U, 0x7c7cf884U, 0x7777ee99U, 0x7b7bf68dU, 0xf2f2ff0dU, 0x6b6bd6bdU, 0x6f6fdeb1U, 0xc5c59154U,
  0x30306050U, 0x01010203U, 0x6767cea9U, 0x2b2b567dU, 0xfefee719U, 0xd7d7b562U, 0xabab4de6U, 0x7676ec9aU,
  0xcaca8f45U, 0x82821f9dU, 0xc9c98940U, 0x7d7dfa87U, 0xfafaef15U, 0x5959b2ebU, 0x47478ec9U, 0xf0f0fb0bU,
  0xadad41ecU, 0xd4d4b367U, 0xa2a25ffdU, 0xafaf45eaU, 0x9c9c23bfU, 0xa4a453f7U, 0x7272e496U, 0xc0c09b5bU,
  0xb7b775c2U, 0xfdfde11cU, 0x93933daeU, 0x26264c6aU, 0x36366c5aU, 0x3f3f7e41U, 0xf7f7f502U, 0xcccc834fU,
  0x3434685cU, 0xa5a551f4U, 0xe5e5d134U, 0xf1f1f908U, 0x7171e293U, 0xd8d8ab73U, 0x31316253U, 0x15152a3fU,
  0x0404080cU, 0xc7c79552U, 0x23234665U, 0xc3c39d5eU, 0x18183028U, 0x969637a1U, 0x05050a0fU, 0x9a9a2fb5U,
  0x07070e09U, 0x12122436U, 0x80801b9bU, 0xe2e2df3dU, 0xebebcd26U, 0x27274e69U, 0xb2b27fcdU, 0x7575ea9fU,
  0x0909121bU, 0x83831d9eU, 0x2c2c5874U, 0x1a1a342eU, 0x1b1b362dU, 0x6e6edcb2U, 0x5a5ab4eeU, 0xa0a05bfbU,
  0x5252a4f6U, 0x3b3b764dU, 0xd6d6b761U, 0xb3b37dceU, 0x2929527bU, 0xe3e3dd3eU, 0x2f2f5e71U, 0x84841397U,
  0x5353a6f5U, 0xd1d1b968U, 0x00000000U, 0xededc12cU, 0x20204060U, 0xfcfce31fU, 0xb1b179c8U, 0x5b5bb6edU,
  0x6a6ad4beU, 0xcbcb8d46U, 0xbebe67d9U, 0x3939724bU, 0x4a4a94deU, 0x4c4c98d4U, 0x5858b0e8U, 0xcfcf854aU,
  0xd0d0bb6bU, 0xefefc52aU, 0xaaaa4fe5U, 0xfbfbed16U, 0x434386c5U, 0x4d4d9ad7U, 0x33336655U, 0x85851194U,
  0x45458acfU, 0xf9f9e910U, 0x02020406U, 0x7f7ffe81U, 0x5050a0f0U, 0x3c3c7844U, 0x9f9f25baU, 0xa8a84be3U,
  0x5151a2f3U, 0xa3a35dfeU, 0x404080c0U, 0x8f8f058aU, 0x92923fadU, 0x9d9d21bcU, 0x38387048U, 0xf5f5f104U,
  0xbcbc63dfU, 0xb6b677c1U, 0xdadaaf75U, 0x21214263U, 0x10102030U, 0xffffe51aU, 0xf3f3fd0eU, 0xd2d2bf6dU,
  0xcdcd814cU, 0x0c0c1814U, 0x13132635U, 0xececc32fU, 0x5f5fbee1U, 0x979735a2U, 0x444488ccU, 0x17172e39U,
  0xc4c49357U, 0xa7a755f2U, 0x7e7efc82U, 0x3d3d7a47U, 0x6464c8acU, 0x5d5dbae7U, 0x1919322bU, 0x7373e695U,
  0x6060c0a0U, 0x81811998U, 0x4f4f9ed1U, 0xdcdca37fU, 0x22224466U, 0x2a2a547eU, 0x90903babU, 0x88880b83U,
  0x46468ccaU, 0xeeeec729U, 0xb8b86bd3U, 0x1414283cU, 0xdedea779U, 0x5e5ebce2U, 0x0b0b161dU, 0xdbdbad76U,
  0xe0e0db3bU, 0x32326456U, 0x3a3a744eU, 0x0a0a141eU, 0x494992dbU, 0x06060c0aU, 0x2424486cU, 0x5c5cb8e4U,
  0xc2c29f5dU, 0xd3d3bd6eU, 0xacac43efU, 0x6262c4a6U, 0x919139a8U, 0x959531a4U, 0xe4e4d337U, 0x7979f28bU,
  0xe7e7d532U, 0xc8c88b43U, 0x37376e59U, 0x6d6ddab7U, 0x8d8d018cU, 0xd5d5b164U, 0x4e4e9cd2U, 0xa9a949e0U,
  0x6c6cd8b4U, 0x5656acfaU, 0xf4f4f307U, 0xeaeacf25U, 0x6565caafU, 0x7a7af48eU, 0xaeae47e9U, 0x08081018U,
  0xbaba6fd5U, 0x7878f088U, 0x25254a6fU, 0x2e2e5c72U, 0x1c1c3824U, 0xa6a657f1U, 0xb4b473c7U, 0xc6c69751U,
  0xe8e8cb23U, 0xdddda17cU, 0x7474e89cU, 0x1f1f3e21U, 0x4b4b96ddU, 0xbdbd61dcU, 0x8b8b0d86U, 0x8a8a0f85U,
  0x7070e090U, 0x3e3e7c42U, 0xb5b571c4U, 0x6666ccaaU, 0x484890d8U, 0x03030605U, 0xf6f6f701U, 0x0e0e1c12U,
  0x6161c2a3U, 0x35356a5fU, 0x5757aef9U, 0xb9b969d0U, 0x86861791U, 0xc1c19958U, 0x1d1d3a27U, 0x9e9e27b9U,
  0xe1e1d938U, 0xf8f8eb13U, 0x98982bb3U, 0x11112233U, 0x6969d2bbU, 0xd9d9a970U, 0x8e8e0789U, 0x949433a7U,
  0x9b9b2db6U, 0x1e1e3c22U, 0x87871592U, 0xe9e9c920U, 0xcece8749U, 0x5555aaffU, 0x28285078U, 0xdfdfa57aU,
  0x8c8c038fU, 0xa1a159f8U, 0x89890980U, 0x0d0d1a17U, 0xbfbf65daU, 0xe6e6d731U, 0x424284c6U, 0x6868d0b8U,
  0x414182c3U, 0x999929b0U, 0x2d2d5a77U, 0x0f0f1e11U, 0xb0b07bcbU, 0x5454a8fcU, 0xbbbb6dd6U, 0x16162c3aU,

  0x63c6a563U, 0x7cf8847cU, 0x77ee9977U, 0x7bf68d7bU, 0xf2ff0df2U, 0x6bd6bd6bU, 0x6fdeb16fU, 0xc59154c5U,
  0x30605030U, 0x01020301U, 0x67cea967U, 0x2b567d2bU, 0xfee719feU, 0xd7b562d7U, 0xab4de6abU, 0x76ec9a76U,
  0xca8f45caU, 0x821f9d82U, 0xc98940c9U, 0x7dfa877dU, 0xfaef15faU, 0x59b2eb59U, 0x478ec947U, 0xf0fb0bf0U,
  0xad41ecadU, 0xd4b367d4U, 0xa25ffda2U, 0xaf45eaafU, 0x9c23bf9cU, 0xa453f7a4U, 0x72e49672U, 0xc09b5bc0U,
  0xb775c2b7U, 0xfde11cfdU, 0x933dae93U, 0x264c6a26U, 0x366c5a36U, 0x3f7e413fU, 0xf7f502f7U, 0xcc834fccU,
  0x34685c34U, 0xa551f4a5U, 0xe5d134e5U, 0xf1f908f1U, 0x71e29371U, 0xd8ab73d8U, 0x31625331U, 0x152a3f15U,
  0x04080c04U, 0xc79552c7U, 0x23466523U, 0xc39d5ec3U, 0x18302818U, 0x9637a196U, 0x050a0f05U, 0x9a2fb59aU,
  0x070e0907U, 0x12243612U, 0x801b9b80U, 0xe2df3de2U, 0xebcd26ebU, 0x274e6927U, 0xb27fcdb2U, 0x75ea9f75U,
  0x09121b09U, 0x831d9e83U, 0x2c58742cU, 0x1a342e1aU, 0x1b362d1bU, 0x6edcb26eU, 0x5ab4ee5aU, 0xa05bfba0U,
  0x52a4f652U, 0x3b764d3bU, 0xd6b761d6U, 0xb37dceb3U, 0x29527b29U, 0xe3dd3ee3U, 0x2f5e712fU, 0x84139784U,
  0x53a6f553U, 0xd1b968d1U, 0x00000000U, 0xedc12cedU, 0x20406020U, 0xfce31ffcU, 0xb179c8b1U, 0x5bb6ed5bU,
  0x6ad4be6aU, 0xcb8d46cbU, 0xbe67d9beU, 0x39724b39U, 0x4a94de4aU, 0x4c98d44cU, 0x58b0e858U, 0xcf854acfU,
  0xd0bb6bd0U, 0xefc52aefU, 0xaa4fe5aaU, 0xfbed16fbU, 0x4386c543U, 0x4d9ad74dU, 0x33665533U, 0x85119485U,
  0x458acf45U, 0xf9e910f9U, 0x02040602U, 0x7ffe817fU, 0x50a0f050U, 0x3c78443cU, 0x9f25ba9fU, 0xa84be3a8U,
  0x51a2f351U, 0xa35dfea3U, 0x4080c040U, 0x8f058a8fU, 0x923fad92U, 0x9d21bc9dU, 0x38704838U, 0xf5f104f5U,
  0xbc63dfbcU, 0xb677c1b6U, 0xdaaf75daU, 0x21426321U, 0x10203010U, 0xffe51affU, 0xf3fd0ef3U, 0xd2bf6dd2U,
  0xcd814ccdU, 0x0c18140cU, 0x13263513U, 0xecc32fecU, 0x5fbee15fU, 0x9735a297U, 0x4488cc44U, 0x172e3917U,
  0xc49357c4U, 0xa755f2a7U, 0x7efc827eU, 0x3d7a473dU, 0x64c8ac64U, 0x5dbae75dU, 0x19322b19U, 0x73e69573U,
  0x60c0a060U, 0x81199881U, 0x4f9ed14fU, 0xdca37fdcU, 0x22446622U, 0x2a547e2aU, 0x903bab90U, 0x880b8388U,
  0x468cca46U, 0xeec729eeU, 0xb86bd3b8U, 0x14283c14U, 0xdea779deU, 0x5ebce25eU, 0x0b161d0bU, 0xdbad76dbU,
  0xe0db3be0U, 0x32645632U, 0x3a744e3aU, 0x0a141e0aU, 0x4992db49U, 0x060c0a06U, 0x24486c24U, 0x5cb8e45cU,
  0xc29f5dc2U, 0xd3bd6ed3U, 0xac43efacU, 0x62c4a662U, 0x9139a891U, 0x9531a495U, 0xe4d337e4U, 0x79f28b79U,
  0xe7d532e7U, 0xc88b43c8U, 0x376e5937U, 0x6ddab76dU, 0x8d018c8dU, 0xd5b164d5U, 0x4e9cd24eU, 0xa949e0a9U,
  0x6cd8b46cU, 0x56acfa56U, 0xf4f307f4U, 0xeacf25eaU, 0x65caaf65U, 0x7af48e7aU, 0xae47e9aeU, 0x08101808U,
  0xba6fd5baU, 0x78f08878U, 0x254a6f25U, 0x2e5c722eU, 0x1c38241cU, 0xa657f1a6U, 0xb473c7b4U, 0xc69751c6U,
  0xe8cb23e8U, 0xdda17cddU, 0x74e89c74U, 0x1f3e211fU, 0x4b96dd4bU, 0xbd61dcbdU, 0x8b0d868bU, 0x8a0f858aU,
  0x70e09070U, 0x3e7c423eU, 0xb571c4b5U, 0x66ccaa66U, 0x4890d848U, 0x03060503U, 0xf6f701f6U, 0x0e1c120eU,
  0x61c2a361U, 0x356a5f35U, 0x57aef957U, 0xb969d0b9U, 0x86179186U, 0xc19958c1U, 0x1d3a271dU, 0x9e27b99eU,
  0xe1d938e1U, 0xf8eb13f8U, 0x982bb398U, 0x11223311U, 0x69d2bb69U, 0xd9a970d9U, 0x8e07898eU, 0x9433a794U,
  0x9b2db69bU, 0x1e3c221eU, 0x87159287U, 0xe9c920e9U, 0xce8749ceU, 0x55aaff55U, 0x28507828U, 0xdfa57adfU,
  0x8c038f8cU, 0xa159f8a1U, 0x89098089U, 0x0d1a170dU, 0xbf65dabfU, 0xe6d731e6U, 0x4284c642U, 0x68d0b868U,
  0x4182c341U, 0x9929b099U, 0x2d5a772dU, 0x0f1e110fU, 0xb07bcbb0U, 0x54a8fc54U, 0xbb6dd6bbU, 0x162c3a16U,

  0xc6a56363U, 0xf8847c7cU, 0xee997777U, 0xf68d7b7bU, 0xff0df2f2U, 0xd6bd6b6bU, 0xdeb16f6fU, 0x9154c5c5U,
  0x60503030U, 0x02030101U, 0xcea96767U, 0x567d2b2bU, 0xe719fefeU, 0xb562d7d7U, 0x4de6ababU, 0xec9a7676U,
  0x8f45cacaU, 0x1f9d8282U, 0x8940c9c9U, 0xfa877d7dU, 0xef15fafaU, 0xb2eb5959U, 0x8ec94747U, 0xfb0bf0f0U,
  0x41ecadadU, 0xb367d4d4U, 0x5ffda2a2U, 0x45eaafafU, 0x23bf9c9cU, 0x53f7a4a4U, 0xe4967272U, 0x9b5bc0c0U,
  0x75c2b7b7U, 0xe11cfdfdU, 0x3dae9393U, 0x4c6a2626U, 0x6c5a3636U, 0x7e413f3fU, 0xf502f7f7U, 0x834fccccU,
  0x685c3434U, 0x51f4a5a5U, 0xd134e5e5U, 0xf908f1f1U, 0xe2937171U, 0xab73d8d8U, 0x62533131U, 0x2a3f1515U,
  0x080c0404U, 0x9552c7c7U, 0x46652323U, 0x9d5ec3c3U, 0x30281818U, 0x37a19696U, 0x0a0f0505U, 0x2fb59a9aU,
  0x0e090707U, 0x24361212U, 0x1b9b8080U, 0xdf3de2e2U, 0xcd26ebebU, 0x4e692727U, 0x7fcdb2b2U, 0xea9f7575U,
  0x121b0909U, 0x1d9e8383U, 0x58742c2cU, 0x342e1a1aU, 0x362d1b1bU, 0xdcb26e6eU, 0xb4ee5a5aU, 0x5bfba0a0U,
  0xa4f65252U, 0x764d3b3bU, 0xb761d6d6U, 0x7dceb3b3U, 0x527b2929U, 0xdd3ee3e3U, 0x5e712f2fU, 0x13978484U,
  0xa6f55353U, 0xb968d1d1U, 0x00000000U, 0xc12cededU, 0x40602020U, 0xe31ffcfcU, 0x79c8b1b1U, 0xb6ed5b5bU,
  0xd4be6a6aU, 0x8d46cbcbU, 0x67d9bebeU, 0x724b3939U, 0x94de4a4aU, 0x98d44c4cU, 0xb0e85858U, 0x854acfcfU,
  0xbb6bd0d0U, 0xc52aefefU, 0x4fe5aaaaU, 0xed16fbfbU, 0x86c54343U, 0x9ad74d4dU, 0x66553333U, 0x11948585U,
  0x8acf4545U, 0xe910f9f9U, 0x04060202U, 0xfe817f7fU, 0xa0f05050U, 0x78443c3cU, 0x25ba9f9fU, 0x4be3a8a8U,
  0xa2f35151U, 0x5dfea3a3U, 0x80c04040U, 0x058a8f8fU, 0x3fad9292U, 0x21bc9d9dU, 0x70483838U, 0xf104f5f5U,
  0x63dfbcbcU, 0x77c1b6b6U, 0xaf75dadaU, 0x42632121U, 0x20301010U, 0xe51affffU, 0xfd0ef3f3U, 0xbf6dd2d2U,
  0x814ccdcdU, 0x18140c0cU, 0x26351313U, 0xc32fececU, 0xbee15f5fU, 0x35a29797U, 0x88cc4444U, 0x2e391717U,
  0x9357c4c4U, 0x55f2a7a7U, 0xfc827e7eU, 0x7a473d3dU, 0xc8ac6464U, 0xbae75d5dU, 0x322b1919U, 0xe6957373U,
  0xc0a06060U, 0x19988181U, 0x9ed14f4fU, 0xa37fdcdcU, 0x44662222U, 0x547e2a2aU, 0x3bab9090U, 0x0b838888U,
  0x8cca4646U, 0xc729eeeeU, 0x6bd3b8b8U, 0x283c1414U, 0xa779dedeU, 0xbce25e5eU, 0x161d0b0bU, 0xad76dbdbU,
  0xdb3be0e0U, 0x64563232U, 0x744e3a3aU, 0x141e0a0aU, 0x92db4949U, 0x0c0a0606U, 0x486c2424U, 0xb8e45c5cU,
  0x9f5dc2c2U, 0xbd6ed3d3U, 0x43efacacU, 0xc4a66262U, 0x39a89191U, 0x31a49595U, 0xd337e4e4U, 0xf28b7979U,
  0xd532e7e7U, 0x8b43c8c8U, 0x6e593737U, 0xdab76d6dU, 0x018c8d8dU, 0xb164d5d5U, 0x9cd24e4eU, 0x49e0a9a9U,
  0xd8b46c6cU, 0xacfa5656U, 0xf307f4f4U, 0xcf25eaeaU, 0xcaaf6565U, 0xf48e7a7aU, 0x47e9aeaeU, 0x10180808U,
  0x6fd5babaU, 0xf0887878U, 0x4a6f2525U, 0x5c722e2eU, 0x38241c1cU, 0x57f1a6a6U, 0x73c7b4b4U, 0x9751c6c6U,
  0xcb23e8e8U, 0xa17cddddU, 0xe89c7474U, 0x3e211f1fU, 0x96dd4b4bU, 0x61dcbdbdU, 0x0d868b8bU, 0x0f858a8aU,
  0xe0907070U, 0x7c423e3eU, 0x71c4b5b5U, 0xccaa6666U, 0x90d84848U, 0x06050303U, 0xf701f6f6U, 0x1c120e0eU,
  0xc2a36161U, 0x6a5f3535U, 0xaef95757U, 0x69d0b9b9U, 0x17918686U, 0x9958c1c1U, 0x3a271d1dU, 0x27b99e9eU,
  0xd938e1e1U, 0xeb13f8f8U, 0x2bb39898U, 0x22331111U, 0xd2bb6969U, 0xa970d9d9U, 0x07898e8eU, 0x33a79494U,
  0x2db69b9bU, 0x3c221e1eU, 0x15928787U, 0xc920e9e9U, 0x8749ceceU, 0xaaff5555U, 0x50782828U, 0xa57adfdfU,
  0x038f8c8cU, 0x59f8a1a1U, 0x09808989U, 0x1a170d0dU, 0x65dabfbfU, 0xd731e6e6U, 0x84c64242U, 0xd0b86868U,
  0x82c34141U, 0x29b09999U, 0x5a772d2dU, 0x1e110f0fU, 0x7bcbb0b0U, 0xa8fc5454U, 0x6dd6bbbbU, 0x2c3a1616U
};
*/

#define AS_U32(addr) *((uint32_t*)(addr))
#define AS_UINT2(addr) *((uint2*)(addr))
#define AS_UINT4(addr) *((uint4*)(addr))
#define AS_UL2(addr) *((ulonglong2*)(addr))

#define t_fn0(x) (sharedMemory[x])
#define t_fn1(x) (sharedMemory[0x100U | (x)])
#define t_fn2(x) (sharedMemory[0x200U | (x)])
#define t_fn3(x) (sharedMemory[0x300U | (x)])

#define round(shared, out, x, k) \
  out[0] = (k)[0] ^ (t_fn0(x[0] & 0xff) ^ t_fn1((x[1] >> 8) & 0xff) ^ t_fn2((x[2] >> 16) & 0xff) ^ t_fn3((x[3] >> 24) & 0xff)); \
  out[1] = (k)[1] ^ (t_fn0(x[1] & 0xff) ^ t_fn1((x[2] >> 8) & 0xff) ^ t_fn2((x[3] >> 16) & 0xff) ^ t_fn3((x[0] >> 24) & 0xff)); \
  out[2] = (k)[2] ^ (t_fn0(x[2] & 0xff) ^ t_fn1((x[3] >> 8) & 0xff) ^ t_fn2((x[0] >> 16) & 0xff) ^ t_fn3((x[1] >> 24) & 0xff)); \
  out[3] = (k)[3] ^ (t_fn0(x[3] & 0xff) ^ t_fn1((x[0] >> 8) & 0xff) ^ t_fn2((x[1] >> 16) & 0xff) ^ t_fn3((x[2] >> 24) & 0xff));

#define round_u4(shared, out, in, k) \
  ((uint32_t*)out)[0] = (k)[0] ^ t_fn0(in[0].x) ^ t_fn1(in[1].y) ^ t_fn2(in[2].z) ^ t_fn3(in[3].w); \
  ((uint32_t*)out)[1] = (k)[1] ^ t_fn0(in[1].x) ^ t_fn1(in[2].y) ^ t_fn2(in[3].z) ^ t_fn3(in[0].w); \
  ((uint32_t*)out)[2] = (k)[2] ^ t_fn0(in[2].x) ^ t_fn1(in[3].y) ^ t_fn2(in[0].z) ^ t_fn3(in[1].w); \
  ((uint32_t*)out)[3] = (k)[3] ^ t_fn0(in[3].x) ^ t_fn1(in[0].y) ^ t_fn2(in[1].z) ^ t_fn3(in[2].w);

#ifdef __INTELLISENSE__
#define __byte_perm(a,b,c) a
#endif

#define OFF32_0(x) (x & 0xFFu)
#define OFF32_1(x) __byte_perm(x, 0x01, 0x5541)
#define OFF32_2(x) __byte_perm(x, 0x02, 0x5542)
#define OFF32_3(x) __byte_perm(x, 0x03, 0x5543)

#define SHARED_0(x) sharedMemory[OFF32_0(x)]
#define SHARED_1(x) sharedMemory[OFF32_1(x)]
#define SHARED_2(x) sharedMemory[OFF32_2(x)]
#define SHARED_3(x) sharedMemory[OFF32_3(x)]

__device__ __forceinline__
void cn_aes_single_round(uint32_t * const sharedMemory, uint32_t * const in, uint32_t * out, uint32_t* expandedKey)
{
  asm("// aes_single_round");
  out[0] = expandedKey[0] ^ SHARED_0(in[0]) ^ SHARED_1(in[1]) ^ SHARED_2(in[2]) ^ SHARED_3(in[3]);
  out[1] = expandedKey[1] ^ SHARED_0(in[1]) ^ SHARED_1(in[2]) ^ SHARED_2(in[3]) ^ SHARED_3(in[0]);
  out[2] = expandedKey[2] ^ SHARED_0(in[2]) ^ SHARED_1(in[3]) ^ SHARED_2(in[0]) ^ SHARED_3(in[1]);
  out[3] = expandedKey[3] ^ SHARED_0(in[3]) ^ SHARED_1(in[0]) ^ SHARED_2(in[1]) ^ SHARED_3(in[2]);
}

//
#ifdef _WIN64
/* do a mul.wide.u32 to prevent a shl + cvt 32 to 64 on ld.shared [ptr] */
#define OFF8_0(x) (x & 0xFFu) * sizeof(uint32_t)
#define OFF8_1(x) __byte_perm(x, 0x01, 0x5541) * sizeof(uint32_t)
#define OFF8_2(x) __byte_perm(x, 0x02, 0x5542) * sizeof(uint32_t)
#define OFF8_3(x) __byte_perm(x, 0x03, 0x5543) * sizeof(uint32_t)
#else
#define OFF8_0(x) (x & 0xFFu) << 2
#define OFF8_1(x) __byte_perm(x, 0x01, 0x5541) << 2
#define OFF8_2(x) __byte_perm(x, 0x02, 0x5542) << 2
#define OFF8_3(x) __byte_perm(x, 0x03, 0x5543) << 2
#endif

#define SHAR8_0(x) AS_U32(&sharedMemory[OFF8_0(x)])
#define SHAR8_1(x) AS_U32(&sharedMemory[OFF8_1(x)])
#define SHAR8_2(x) AS_U32(&sharedMemory[OFF8_2(x)])
#define SHAR8_3(x) AS_U32(&sharedMemory[OFF8_3(x)])

__device__ __forceinline__
void cn_aes_single_round_b(uint8_t * const sharedMemory, void * const long_state, const uint4 key, uint4 *res)
{
  asm("// aes_single_round_b");
  uint4 in = AS_UINT4(long_state);
  *res = key;
  res->x ^= SHAR8_0(in.x) ^ SHAR8_1(in.y) ^ SHAR8_2(in.z) ^ SHAR8_3(in.w);
  res->y ^= SHAR8_0(in.y) ^ SHAR8_1(in.z) ^ SHAR8_2(in.w) ^ SHAR8_3(in.x);
  res->z ^= SHAR8_0(in.z) ^ SHAR8_1(in.w) ^ SHAR8_2(in.x) ^ SHAR8_3(in.y);
  res->w ^= SHAR8_0(in.w) ^ SHAR8_1(in.x) ^ SHAR8_2(in.y) ^ SHAR8_3(in.z);
}

#define round_perm(shared, out, in, k) \
  out[0] = (k)[0] ^ SHARED_0(in[0]) ^ SHARED_1(in[1]) ^ SHARED_2(in[2]) ^ SHARED_3(in[3]); \
  out[1] = (k)[1] ^ SHARED_0(in[1]) ^ SHARED_1(in[2]) ^ SHARED_2(in[3]) ^ SHARED_3(in[0]); \
  out[2] = (k)[2] ^ SHARED_0(in[2]) ^ SHARED_1(in[3]) ^ SHARED_2(in[0]) ^ SHARED_3(in[1]); \
  out[3] = (k)[3] ^ SHARED_0(in[3]) ^ SHARED_1(in[0]) ^ SHARED_2(in[1]) ^ SHARED_3(in[2]);

__device__ __forceinline__
void cn_aes_pseudo_round_mut(const uint32_t * sharedMemory, uint32_t * val, uint32_t const * expandedKey)
{
  asm("// aes_pseudo_round_mut");
  uint32_t b[4];
  round_perm(sharedMemory, b, val, expandedKey);
  round_perm(sharedMemory, val, b, expandedKey + (1 * N_COLS));
  round_perm(sharedMemory, b, val, expandedKey + (2 * N_COLS));
  round_perm(sharedMemory, val, b, expandedKey + (3 * N_COLS));
  round_perm(sharedMemory, b, val, expandedKey + (4 * N_COLS));
  round_perm(sharedMemory, val, b, expandedKey + (5 * N_COLS));
  round_perm(sharedMemory, b, val, expandedKey + (6 * N_COLS));
  round_perm(sharedMemory, val, b, expandedKey + (7 * N_COLS));
  round_perm(sharedMemory, b, val, expandedKey + (8 * N_COLS));
  round_perm(sharedMemory, val, b, expandedKey + (9 * N_COLS));
}

static __forceinline__ __device__ uint4 operator ^ (const uint4 &a, const uint4 &b) {
  return make_uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w);
}

#define round_perm4(in, k) {\
  uint4 tmp; \
  tmp.x = SHARED_0(in.x) ^ SHARED_1(in.y) ^ SHARED_2(in.z) ^ SHARED_3(in.w); \
  tmp.y = SHARED_0(in.y) ^ SHARED_1(in.z) ^ SHARED_2(in.w) ^ SHARED_3(in.x); \
  tmp.z = SHARED_0(in.z) ^ SHARED_1(in.w) ^ SHARED_2(in.x) ^ SHARED_3(in.y); \
  tmp.w = SHARED_0(in.w) ^ SHARED_1(in.x) ^ SHARED_2(in.y) ^ SHARED_3(in.z); \
  val = tmp ^ key[k]; \
}

__device__ __forceinline__
void cn_aes_pseudo_round_mut_uint4(uint32_t * const sharedMemory, uint4 &val, uint4 const key[10])
{
  asm("// aes_pseudo_round_mut_uint4");
  round_perm4(val, 0);
  round_perm4(val, 1);
  round_perm4(val, 2);
  round_perm4(val, 3);
  round_perm4(val, 4);
  round_perm4(val, 5);
  round_perm4(val, 6);
  round_perm4(val, 7);
  round_perm4(val, 8);
  round_perm4(val, 9);
}

/*
__device__ __forceinline__
void cn_aes_gpu_init2(uint32_t* sharedMemory)
{
#if 0
  if(blockDim.x >= 64)
  {
    if(threadIdx.x < 64) {
      #define thrX (threadIdx.x << 2U) // ensure offsets aligned (16) to vector
      #pragma unroll 4
      for (uint32_t i = 0; i < 1024U; i += 256U) // 32x32 = 1024, 4 * 256 also
        AS_UINT4(&sharedMemory[i + thrX]) = AS_UINT4(&d_t_fn[i + thrX]);
    }

  } else
#endif
  if(blockDim.x >= 32) {

    if(threadIdx.x < 32) {
#if 0
      #pragma unroll 32
      for(uint32_t i = 0; i < 1024; i += 32)
        sharedMemory[threadIdx.x + i] = d_t_fn[threadIdx.x + i];
#else
      #define thrX (threadIdx.x << 2U) // ensure offsets aligned (16) to vector
      #pragma unroll 8
      for (uint32_t i = 0; i < 1024; i += 128U) // 32x32 = 1024, 8 * 128 also
        AS_UINT4(&sharedMemory[i + thrX]) = AS_UINT4(&d_t_fn[i + thrX]);
#endif
    }

  } else {

    if(threadIdx.x < 4) {
#if 0
      for (uint32_t i = 0; i < 1024; i += 4)
        sharedMemory[threadIdx.x + i] = d_t_fn[threadIdx.x + i];
#else
      #define thrX (threadIdx.x << 2U) // ensure offsets aligned (16) to vector
      #pragma unroll 64
      for (uint32_t i = 0; i < 1024; i += 16U)
        AS_UINT4(&sharedMemory[i + thrX]) = AS_UINT4(&d_t_fn[i + thrX]);
#endif
    }
  }
}
*/

__device__ __forceinline__
void cn_aes_gpu_init(uint32_t* sharedMemory)
{
  // AES 0
  switch (threadIdx.x) {
  case 0:
    AS_UL2(&sharedMemory[0x000]) = make_ulonglong2(0x847c7cf8a56363c6, 0x8d7b7bf6997777ee);
    AS_UL2(&sharedMemory[0x004]) = make_ulonglong2(0xbd6b6bd60df2f2ff, 0x54c5c591b16f6fde);
    AS_UL2(&sharedMemory[0x008]) = make_ulonglong2(0x0301010250303060, 0x7d2b2b56a96767ce);
    AS_UL2(&sharedMemory[0x00C]) = make_ulonglong2(0x62d7d7b519fefee7, 0x9a7676ece6abab4d);
    AS_UL2(&sharedMemory[0x010]) = make_ulonglong2(0x9d82821f45caca8f, 0x877d7dfa40c9c989);
    AS_UL2(&sharedMemory[0x014]) = make_ulonglong2(0xeb5959b215fafaef, 0x0bf0f0fbc947478e);
    AS_UL2(&sharedMemory[0x018]) = make_ulonglong2(0x67d4d4b3ecadad41, 0xeaafaf45fda2a25f);
    AS_UL2(&sharedMemory[0x01C]) = make_ulonglong2(0xf7a4a453bf9c9c23, 0x5bc0c09b967272e4);
    break;
  case 1:
    AS_UL2(&sharedMemory[0x020]) = make_ulonglong2(0x1cfdfde1c2b7b775, 0x6a26264cae93933d);
    AS_UL2(&sharedMemory[0x024]) = make_ulonglong2(0x413f3f7e5a36366c, 0x4fcccc8302f7f7f5);
    AS_UL2(&sharedMemory[0x028]) = make_ulonglong2(0xf4a5a5515c343468, 0x08f1f1f934e5e5d1);
    AS_UL2(&sharedMemory[0x02C]) = make_ulonglong2(0x73d8d8ab937171e2, 0x3f15152a53313162);
    AS_UL2(&sharedMemory[0x030]) = make_ulonglong2(0x52c7c7950c040408, 0x5ec3c39d65232346);
    AS_UL2(&sharedMemory[0x034]) = make_ulonglong2(0xa196963728181830, 0xb59a9a2f0f05050a);
    AS_UL2(&sharedMemory[0x038]) = make_ulonglong2(0x361212240907070e, 0x3de2e2df9b80801b);
    AS_UL2(&sharedMemory[0x03C]) = make_ulonglong2(0x6927274e26ebebcd, 0x9f7575eacdb2b27f);
    break;
  case 2:
    AS_UL2(&sharedMemory[0x040]) = make_ulonglong2(0x9e83831d1b090912, 0x2e1a1a34742c2c58);
    AS_UL2(&sharedMemory[0x044]) = make_ulonglong2(0xb26e6edc2d1b1b36, 0xfba0a05bee5a5ab4);
    AS_UL2(&sharedMemory[0x048]) = make_ulonglong2(0x4d3b3b76f65252a4, 0xceb3b37d61d6d6b7);
    AS_UL2(&sharedMemory[0x04C]) = make_ulonglong2(0x3ee3e3dd7b292952, 0x97848413712f2f5e);
    AS_UL2(&sharedMemory[0x050]) = make_ulonglong2(0x68d1d1b9f55353a6, 0x2cededc100000000);
    AS_UL2(&sharedMemory[0x054]) = make_ulonglong2(0x1ffcfce360202040, 0xed5b5bb6c8b1b179);
    AS_UL2(&sharedMemory[0x058]) = make_ulonglong2(0x46cbcb8dbe6a6ad4, 0x4b393972d9bebe67);
    AS_UL2(&sharedMemory[0x05C]) = make_ulonglong2(0xd44c4c98de4a4a94, 0x4acfcf85e85858b0);
    break;
  case 3:
    AS_UL2(&sharedMemory[0x060]) = make_ulonglong2(0x2aefefc56bd0d0bb, 0x16fbfbede5aaaa4f);
    AS_UL2(&sharedMemory[0x064]) = make_ulonglong2(0xd74d4d9ac5434386, 0x9485851155333366);
    AS_UL2(&sharedMemory[0x068]) = make_ulonglong2(0x10f9f9e9cf45458a, 0x817f7ffe06020204);
    AS_UL2(&sharedMemory[0x06C]) = make_ulonglong2(0x443c3c78f05050a0, 0xe3a8a84bba9f9f25);
    AS_UL2(&sharedMemory[0x070]) = make_ulonglong2(0xfea3a35df35151a2, 0x8a8f8f05c0404080);
    AS_UL2(&sharedMemory[0x074]) = make_ulonglong2(0xbc9d9d21ad92923f, 0x04f5f5f148383870);
    AS_UL2(&sharedMemory[0x078]) = make_ulonglong2(0xc1b6b677dfbcbc63, 0x6321214275dadaaf);
    AS_UL2(&sharedMemory[0x07C]) = make_ulonglong2(0x1affffe530101020, 0x6dd2d2bf0ef3f3fd);
    break;
  case 4:
    AS_UL2(&sharedMemory[0x080]) = make_ulonglong2(0x140c0c184ccdcd81, 0x2fececc335131326);
    AS_UL2(&sharedMemory[0x084]) = make_ulonglong2(0xa2979735e15f5fbe, 0x3917172ecc444488);
    AS_UL2(&sharedMemory[0x088]) = make_ulonglong2(0xf2a7a75557c4c493, 0x473d3d7a827e7efc);
    AS_UL2(&sharedMemory[0x08C]) = make_ulonglong2(0xe75d5dbaac6464c8, 0x957373e62b191932);
    AS_UL2(&sharedMemory[0x090]) = make_ulonglong2(0x98818119a06060c0, 0x7fdcdca3d14f4f9e);
    AS_UL2(&sharedMemory[0x094]) = make_ulonglong2(0x7e2a2a5466222244, 0x8388880bab90903b);
    AS_UL2(&sharedMemory[0x098]) = make_ulonglong2(0x29eeeec7ca46468c, 0x3c141428d3b8b86b);
    AS_UL2(&sharedMemory[0x09C]) = make_ulonglong2(0xe25e5ebc79dedea7, 0x76dbdbad1d0b0b16);
    break;
  case 5:
    AS_UL2(&sharedMemory[0x0A0]) = make_ulonglong2(0x563232643be0e0db, 0x1e0a0a144e3a3a74);
    AS_UL2(&sharedMemory[0x0A4]) = make_ulonglong2(0x0a06060cdb494992, 0xe45c5cb86c242448);
    AS_UL2(&sharedMemory[0x0A8]) = make_ulonglong2(0x6ed3d3bd5dc2c29f, 0xa66262c4efacac43);
    AS_UL2(&sharedMemory[0x0AC]) = make_ulonglong2(0xa4959531a8919139, 0x8b7979f237e4e4d3);
    AS_UL2(&sharedMemory[0x0B0]) = make_ulonglong2(0x43c8c88b32e7e7d5, 0xb76d6dda5937376e);
    AS_UL2(&sharedMemory[0x0B4]) = make_ulonglong2(0x64d5d5b18c8d8d01, 0xe0a9a949d24e4e9c);
    AS_UL2(&sharedMemory[0x0B8]) = make_ulonglong2(0xfa5656acb46c6cd8, 0x25eaeacf07f4f4f3);
    AS_UL2(&sharedMemory[0x0BC]) = make_ulonglong2(0x8e7a7af4af6565ca, 0x18080810e9aeae47);
    break;
  case 6:
    AS_UL2(&sharedMemory[0x0C0]) = make_ulonglong2(0x887878f0d5baba6f, 0x722e2e5c6f25254a);
    AS_UL2(&sharedMemory[0x0C4]) = make_ulonglong2(0xf1a6a657241c1c38, 0x51c6c697c7b4b473);
    AS_UL2(&sharedMemory[0x0C8]) = make_ulonglong2(0x7cdddda123e8e8cb, 0x211f1f3e9c7474e8);
    AS_UL2(&sharedMemory[0x0CC]) = make_ulonglong2(0xdcbdbd61dd4b4b96, 0x858a8a0f868b8b0d);
    AS_UL2(&sharedMemory[0x0D0]) = make_ulonglong2(0x423e3e7c907070e0, 0xaa6666ccc4b5b571);
    AS_UL2(&sharedMemory[0x0D4]) = make_ulonglong2(0x05030306d8484890, 0x120e0e1c01f6f6f7);
    AS_UL2(&sharedMemory[0x0D8]) = make_ulonglong2(0x5f35356aa36161c2, 0xd0b9b969f95757ae);
    AS_UL2(&sharedMemory[0x0DC]) = make_ulonglong2(0x58c1c19991868617, 0xb99e9e27271d1d3a);
    break;
  case 7:
    AS_UL2(&sharedMemory[0x0E0]) = make_ulonglong2(0x13f8f8eb38e1e1d9, 0x33111122b398982b);
    AS_UL2(&sharedMemory[0x0E4]) = make_ulonglong2(0x70d9d9a9bb6969d2, 0xa7949433898e8e07);
    AS_UL2(&sharedMemory[0x0E8]) = make_ulonglong2(0x221e1e3cb69b9b2d, 0x20e9e9c992878715);
    AS_UL2(&sharedMemory[0x0EC]) = make_ulonglong2(0xff5555aa49cece87, 0x7adfdfa578282850);
    AS_UL2(&sharedMemory[0x0F0]) = make_ulonglong2(0xf8a1a1598f8c8c03, 0x170d0d1a80898909);
    AS_UL2(&sharedMemory[0x0F4]) = make_ulonglong2(0x31e6e6d7dabfbf65, 0xb86868d0c6424284);
    AS_UL2(&sharedMemory[0x0F8]) = make_ulonglong2(0xb0999929c3414182, 0x110f0f1e772d2d5a);
    AS_UL2(&sharedMemory[0x0FC]) = make_ulonglong2(0xfc5454a8cbb0b07b, 0x3a16162cd6bbbb6d);
    break;
  }
  // AES 1
  switch (threadIdx.x) {
  case 0:
    AS_UL2(&sharedMemory[0x100]) = make_ulonglong2(0x7c7cf8846363c6a5, 0x7b7bf68d7777ee99);
    AS_UL2(&sharedMemory[0x104]) = make_ulonglong2(0x6b6bd6bdf2f2ff0d, 0xc5c591546f6fdeb1);
    AS_UL2(&sharedMemory[0x108]) = make_ulonglong2(0x0101020330306050, 0x2b2b567d6767cea9);
    AS_UL2(&sharedMemory[0x10C]) = make_ulonglong2(0xd7d7b562fefee719, 0x7676ec9aabab4de6);
    AS_UL2(&sharedMemory[0x110]) = make_ulonglong2(0x82821f9dcaca8f45, 0x7d7dfa87c9c98940);
    AS_UL2(&sharedMemory[0x114]) = make_ulonglong2(0x5959b2ebfafaef15, 0xf0f0fb0b47478ec9);
    AS_UL2(&sharedMemory[0x118]) = make_ulonglong2(0xd4d4b367adad41ec, 0xafaf45eaa2a25ffd);
    AS_UL2(&sharedMemory[0x11C]) = make_ulonglong2(0xa4a453f79c9c23bf, 0xc0c09b5b7272e496);
    break;
  case 1:
    AS_UL2(&sharedMemory[0x120]) = make_ulonglong2(0xfdfde11cb7b775c2, 0x26264c6a93933dae);
    AS_UL2(&sharedMemory[0x124]) = make_ulonglong2(0x3f3f7e4136366c5a, 0xcccc834ff7f7f502);
    AS_UL2(&sharedMemory[0x128]) = make_ulonglong2(0xa5a551f43434685c, 0xf1f1f908e5e5d134);
    AS_UL2(&sharedMemory[0x12C]) = make_ulonglong2(0xd8d8ab737171e293, 0x15152a3f31316253);
    AS_UL2(&sharedMemory[0x130]) = make_ulonglong2(0xc7c795520404080c, 0xc3c39d5e23234665);
    AS_UL2(&sharedMemory[0x134]) = make_ulonglong2(0x969637a118183028, 0x9a9a2fb505050a0f);
    AS_UL2(&sharedMemory[0x138]) = make_ulonglong2(0x1212243607070e09, 0xe2e2df3d80801b9b);
    AS_UL2(&sharedMemory[0x13C]) = make_ulonglong2(0x27274e69ebebcd26, 0x7575ea9fb2b27fcd);
    break;
  case 2:
    AS_UL2(&sharedMemory[0x140]) = make_ulonglong2(0x83831d9e0909121b, 0x1a1a342e2c2c5874);
    AS_UL2(&sharedMemory[0x144]) = make_ulonglong2(0x6e6edcb21b1b362d, 0xa0a05bfb5a5ab4ee);
    AS_UL2(&sharedMemory[0x148]) = make_ulonglong2(0x3b3b764d5252a4f6, 0xb3b37dced6d6b761);
    AS_UL2(&sharedMemory[0x14C]) = make_ulonglong2(0xe3e3dd3e2929527b, 0x848413972f2f5e71);
    AS_UL2(&sharedMemory[0x150]) = make_ulonglong2(0xd1d1b9685353a6f5, 0xededc12c00000000);
    AS_UL2(&sharedMemory[0x154]) = make_ulonglong2(0xfcfce31f20204060, 0x5b5bb6edb1b179c8);
    AS_UL2(&sharedMemory[0x158]) = make_ulonglong2(0xcbcb8d466a6ad4be, 0x3939724bbebe67d9);
    AS_UL2(&sharedMemory[0x15C]) = make_ulonglong2(0x4c4c98d44a4a94de, 0xcfcf854a5858b0e8);
    break;
  case 3:
    AS_UL2(&sharedMemory[0x160]) = make_ulonglong2(0xefefc52ad0d0bb6b, 0xfbfbed16aaaa4fe5);
    AS_UL2(&sharedMemory[0x164]) = make_ulonglong2(0x4d4d9ad7434386c5, 0x8585119433336655);
    AS_UL2(&sharedMemory[0x168]) = make_ulonglong2(0xf9f9e91045458acf, 0x7f7ffe8102020406);
    AS_UL2(&sharedMemory[0x16C]) = make_ulonglong2(0x3c3c78445050a0f0, 0xa8a84be39f9f25ba);
    AS_UL2(&sharedMemory[0x170]) = make_ulonglong2(0xa3a35dfe5151a2f3, 0x8f8f058a404080c0);
    AS_UL2(&sharedMemory[0x174]) = make_ulonglong2(0x9d9d21bc92923fad, 0xf5f5f10438387048);
    AS_UL2(&sharedMemory[0x178]) = make_ulonglong2(0xb6b677c1bcbc63df, 0x21214263dadaaf75);
    AS_UL2(&sharedMemory[0x17C]) = make_ulonglong2(0xffffe51a10102030, 0xd2d2bf6df3f3fd0e);
    break;
  case 4:
    AS_UL2(&sharedMemory[0x180]) = make_ulonglong2(0x0c0c1814cdcd814c, 0xececc32f13132635);
    AS_UL2(&sharedMemory[0x184]) = make_ulonglong2(0x979735a25f5fbee1, 0x17172e39444488cc);
    AS_UL2(&sharedMemory[0x188]) = make_ulonglong2(0xa7a755f2c4c49357, 0x3d3d7a477e7efc82);
    AS_UL2(&sharedMemory[0x18C]) = make_ulonglong2(0x5d5dbae76464c8ac, 0x7373e6951919322b);
    AS_UL2(&sharedMemory[0x190]) = make_ulonglong2(0x818119986060c0a0, 0xdcdca37f4f4f9ed1);
    AS_UL2(&sharedMemory[0x194]) = make_ulonglong2(0x2a2a547e22224466, 0x88880b8390903bab);
    AS_UL2(&sharedMemory[0x198]) = make_ulonglong2(0xeeeec72946468cca, 0x1414283cb8b86bd3);
    AS_UL2(&sharedMemory[0x19C]) = make_ulonglong2(0x5e5ebce2dedea779, 0xdbdbad760b0b161d);
    break;
  case 5:
    AS_UL2(&sharedMemory[0x1A0]) = make_ulonglong2(0x32326456e0e0db3b, 0x0a0a141e3a3a744e);
    AS_UL2(&sharedMemory[0x1A4]) = make_ulonglong2(0x06060c0a494992db, 0x5c5cb8e42424486c);
    AS_UL2(&sharedMemory[0x1A8]) = make_ulonglong2(0xd3d3bd6ec2c29f5d, 0x6262c4a6acac43ef);
    AS_UL2(&sharedMemory[0x1AC]) = make_ulonglong2(0x959531a4919139a8, 0x7979f28be4e4d337);
    AS_UL2(&sharedMemory[0x1B0]) = make_ulonglong2(0xc8c88b43e7e7d532, 0x6d6ddab737376e59);
    AS_UL2(&sharedMemory[0x1B4]) = make_ulonglong2(0xd5d5b1648d8d018c, 0xa9a949e04e4e9cd2);
    AS_UL2(&sharedMemory[0x1B8]) = make_ulonglong2(0x5656acfa6c6cd8b4, 0xeaeacf25f4f4f307);
    AS_UL2(&sharedMemory[0x1BC]) = make_ulonglong2(0x7a7af48e6565caaf, 0x08081018aeae47e9);
    break;
  case 6:
    AS_UL2(&sharedMemory[0x1C0]) = make_ulonglong2(0x7878f088baba6fd5, 0x2e2e5c7225254a6f);
    AS_UL2(&sharedMemory[0x1C4]) = make_ulonglong2(0xa6a657f11c1c3824, 0xc6c69751b4b473c7);
    AS_UL2(&sharedMemory[0x1C8]) = make_ulonglong2(0xdddda17ce8e8cb23, 0x1f1f3e217474e89c);
    AS_UL2(&sharedMemory[0x1CC]) = make_ulonglong2(0xbdbd61dc4b4b96dd, 0x8a8a0f858b8b0d86);
    AS_UL2(&sharedMemory[0x1D0]) = make_ulonglong2(0x3e3e7c427070e090, 0x6666ccaab5b571c4);
    AS_UL2(&sharedMemory[0x1D4]) = make_ulonglong2(0x03030605484890d8, 0x0e0e1c12f6f6f701);
    AS_UL2(&sharedMemory[0x1D8]) = make_ulonglong2(0x35356a5f6161c2a3, 0xb9b969d05757aef9);
    AS_UL2(&sharedMemory[0x1DC]) = make_ulonglong2(0xc1c1995886861791, 0x9e9e27b91d1d3a27);
    break;
  case 7:
    AS_UL2(&sharedMemory[0x1E0]) = make_ulonglong2(0xf8f8eb13e1e1d938, 0x1111223398982bb3);
    AS_UL2(&sharedMemory[0x1E4]) = make_ulonglong2(0xd9d9a9706969d2bb, 0x949433a78e8e0789);
    AS_UL2(&sharedMemory[0x1E8]) = make_ulonglong2(0x1e1e3c229b9b2db6, 0xe9e9c92087871592);
    AS_UL2(&sharedMemory[0x1EC]) = make_ulonglong2(0x5555aaffcece8749, 0xdfdfa57a28285078);
    AS_UL2(&sharedMemory[0x1F0]) = make_ulonglong2(0xa1a159f88c8c038f, 0x0d0d1a1789890980);
    AS_UL2(&sharedMemory[0x1F4]) = make_ulonglong2(0xe6e6d731bfbf65da, 0x6868d0b8424284c6);
    AS_UL2(&sharedMemory[0x1F8]) = make_ulonglong2(0x999929b0414182c3, 0x0f0f1e112d2d5a77);
    AS_UL2(&sharedMemory[0x1FC]) = make_ulonglong2(0x5454a8fcb0b07bcb, 0x16162c3abbbb6dd6);
    break;
  }
  // AES 2
  switch (threadIdx.x) {
  case 0:
    AS_UL2(&sharedMemory[0x200]) = make_ulonglong2(0x7cf8847c63c6a563, 0x7bf68d7b77ee9977);
    AS_UL2(&sharedMemory[0x204]) = make_ulonglong2(0x6bd6bd6bf2ff0df2, 0xc59154c56fdeb16f);
    AS_UL2(&sharedMemory[0x208]) = make_ulonglong2(0x0102030130605030, 0x2b567d2b67cea967);
    AS_UL2(&sharedMemory[0x20C]) = make_ulonglong2(0xd7b562d7fee719fe, 0x76ec9a76ab4de6ab);
    AS_UL2(&sharedMemory[0x210]) = make_ulonglong2(0x821f9d82ca8f45ca, 0x7dfa877dc98940c9);
    AS_UL2(&sharedMemory[0x214]) = make_ulonglong2(0x59b2eb59faef15fa, 0xf0fb0bf0478ec947);
    AS_UL2(&sharedMemory[0x218]) = make_ulonglong2(0xd4b367d4ad41ecad, 0xaf45eaafa25ffda2);
    AS_UL2(&sharedMemory[0x21C]) = make_ulonglong2(0xa453f7a49c23bf9c, 0xc09b5bc072e49672);
    break;
  case 1:
    AS_UL2(&sharedMemory[0x220]) = make_ulonglong2(0xfde11cfdb775c2b7, 0x264c6a26933dae93);
    AS_UL2(&sharedMemory[0x224]) = make_ulonglong2(0x3f7e413f366c5a36, 0xcc834fccf7f502f7);
    AS_UL2(&sharedMemory[0x228]) = make_ulonglong2(0xa551f4a534685c34, 0xf1f908f1e5d134e5);
    AS_UL2(&sharedMemory[0x22C]) = make_ulonglong2(0xd8ab73d871e29371, 0x152a3f1531625331);
    AS_UL2(&sharedMemory[0x230]) = make_ulonglong2(0xc79552c704080c04, 0xc39d5ec323466523);
    AS_UL2(&sharedMemory[0x234]) = make_ulonglong2(0x9637a19618302818, 0x9a2fb59a050a0f05);
    AS_UL2(&sharedMemory[0x238]) = make_ulonglong2(0x12243612070e0907, 0xe2df3de2801b9b80);
    AS_UL2(&sharedMemory[0x23C]) = make_ulonglong2(0x274e6927ebcd26eb, 0x75ea9f75b27fcdb2);
    break;
  case 2:
    AS_UL2(&sharedMemory[0x240]) = make_ulonglong2(0x831d9e8309121b09, 0x1a342e1a2c58742c);
    AS_UL2(&sharedMemory[0x244]) = make_ulonglong2(0x6edcb26e1b362d1b, 0xa05bfba05ab4ee5a);
    AS_UL2(&sharedMemory[0x248]) = make_ulonglong2(0x3b764d3b52a4f652, 0xb37dceb3d6b761d6);
    AS_UL2(&sharedMemory[0x24C]) = make_ulonglong2(0xe3dd3ee329527b29, 0x841397842f5e712f);
    AS_UL2(&sharedMemory[0x250]) = make_ulonglong2(0xd1b968d153a6f553, 0xedc12ced00000000);
    AS_UL2(&sharedMemory[0x254]) = make_ulonglong2(0xfce31ffc20406020, 0x5bb6ed5bb179c8b1);
    AS_UL2(&sharedMemory[0x258]) = make_ulonglong2(0xcb8d46cb6ad4be6a, 0x39724b39be67d9be);
    AS_UL2(&sharedMemory[0x25C]) = make_ulonglong2(0x4c98d44c4a94de4a, 0xcf854acf58b0e858);
    break;
  case 3:
    AS_UL2(&sharedMemory[0x260]) = make_ulonglong2(0xefc52aefd0bb6bd0, 0xfbed16fbaa4fe5aa);
    AS_UL2(&sharedMemory[0x264]) = make_ulonglong2(0x4d9ad74d4386c543, 0x8511948533665533);
    AS_UL2(&sharedMemory[0x268]) = make_ulonglong2(0xf9e910f9458acf45, 0x7ffe817f02040602);
    AS_UL2(&sharedMemory[0x26C]) = make_ulonglong2(0x3c78443c50a0f050, 0xa84be3a89f25ba9f);
    AS_UL2(&sharedMemory[0x270]) = make_ulonglong2(0xa35dfea351a2f351, 0x8f058a8f4080c040);
    AS_UL2(&sharedMemory[0x274]) = make_ulonglong2(0x9d21bc9d923fad92, 0xf5f104f538704838);
    AS_UL2(&sharedMemory[0x278]) = make_ulonglong2(0xb677c1b6bc63dfbc, 0x21426321daaf75da);
    AS_UL2(&sharedMemory[0x27C]) = make_ulonglong2(0xffe51aff10203010, 0xd2bf6dd2f3fd0ef3);
    break;
  case 4:
    AS_UL2(&sharedMemory[0x280]) = make_ulonglong2(0x0c18140ccd814ccd, 0xecc32fec13263513);
    AS_UL2(&sharedMemory[0x284]) = make_ulonglong2(0x9735a2975fbee15f, 0x172e39174488cc44);
    AS_UL2(&sharedMemory[0x288]) = make_ulonglong2(0xa755f2a7c49357c4, 0x3d7a473d7efc827e);
    AS_UL2(&sharedMemory[0x28C]) = make_ulonglong2(0x5dbae75d64c8ac64, 0x73e6957319322b19);
    AS_UL2(&sharedMemory[0x290]) = make_ulonglong2(0x8119988160c0a060, 0xdca37fdc4f9ed14f);
    AS_UL2(&sharedMemory[0x294]) = make_ulonglong2(0x2a547e2a22446622, 0x880b8388903bab90);
    AS_UL2(&sharedMemory[0x298]) = make_ulonglong2(0xeec729ee468cca46, 0x14283c14b86bd3b8);
    AS_UL2(&sharedMemory[0x29C]) = make_ulonglong2(0x5ebce25edea779de, 0xdbad76db0b161d0b);
    break;
  case 5:
    AS_UL2(&sharedMemory[0x2A0]) = make_ulonglong2(0x32645632e0db3be0, 0x0a141e0a3a744e3a);
    AS_UL2(&sharedMemory[0x2A4]) = make_ulonglong2(0x060c0a064992db49, 0x5cb8e45c24486c24);
    AS_UL2(&sharedMemory[0x2A8]) = make_ulonglong2(0xd3bd6ed3c29f5dc2, 0x62c4a662ac43efac);
    AS_UL2(&sharedMemory[0x2AC]) = make_ulonglong2(0x9531a4959139a891, 0x79f28b79e4d337e4);
    AS_UL2(&sharedMemory[0x2B0]) = make_ulonglong2(0xc88b43c8e7d532e7, 0x6ddab76d376e5937);
    AS_UL2(&sharedMemory[0x2B4]) = make_ulonglong2(0xd5b164d58d018c8d, 0xa949e0a94e9cd24e);
    AS_UL2(&sharedMemory[0x2B8]) = make_ulonglong2(0x56acfa566cd8b46c, 0xeacf25eaf4f307f4);
    AS_UL2(&sharedMemory[0x2BC]) = make_ulonglong2(0x7af48e7a65caaf65, 0x08101808ae47e9ae);
    break;
  case 6:
    AS_UL2(&sharedMemory[0x2C0]) = make_ulonglong2(0x78f08878ba6fd5ba, 0x2e5c722e254a6f25);
    AS_UL2(&sharedMemory[0x2C4]) = make_ulonglong2(0xa657f1a61c38241c, 0xc69751c6b473c7b4);
    AS_UL2(&sharedMemory[0x2C8]) = make_ulonglong2(0xdda17cdde8cb23e8, 0x1f3e211f74e89c74);
    AS_UL2(&sharedMemory[0x2CC]) = make_ulonglong2(0xbd61dcbd4b96dd4b, 0x8a0f858a8b0d868b);
    AS_UL2(&sharedMemory[0x2D0]) = make_ulonglong2(0x3e7c423e70e09070, 0x66ccaa66b571c4b5);
    AS_UL2(&sharedMemory[0x2D4]) = make_ulonglong2(0x030605034890d848, 0x0e1c120ef6f701f6);
    AS_UL2(&sharedMemory[0x2D8]) = make_ulonglong2(0x356a5f3561c2a361, 0xb969d0b957aef957);
    AS_UL2(&sharedMemory[0x2DC]) = make_ulonglong2(0xc19958c186179186, 0x9e27b99e1d3a271d);
    break;
  case 7:
    AS_UL2(&sharedMemory[0x2E0]) = make_ulonglong2(0xf8eb13f8e1d938e1, 0x11223311982bb398);
    AS_UL2(&sharedMemory[0x2E4]) = make_ulonglong2(0xd9a970d969d2bb69, 0x9433a7948e07898e);
    AS_UL2(&sharedMemory[0x2E8]) = make_ulonglong2(0x1e3c221e9b2db69b, 0xe9c920e987159287);
    AS_UL2(&sharedMemory[0x2EC]) = make_ulonglong2(0x55aaff55ce8749ce, 0xdfa57adf28507828);
    AS_UL2(&sharedMemory[0x2F0]) = make_ulonglong2(0xa159f8a18c038f8c, 0x0d1a170d89098089);
    AS_UL2(&sharedMemory[0x2F4]) = make_ulonglong2(0xe6d731e6bf65dabf, 0x68d0b8684284c642);
    AS_UL2(&sharedMemory[0x2F8]) = make_ulonglong2(0x9929b0994182c341, 0x0f1e110f2d5a772d);
    AS_UL2(&sharedMemory[0x2FC]) = make_ulonglong2(0x54a8fc54b07bcbb0, 0x162c3a16bb6dd6bb);
    break;
  }
  // AES 3
  switch (threadIdx.x) {
  case 0:
    AS_UL2(&sharedMemory[0x300]) = make_ulonglong2(0xf8847c7cc6a56363, 0xf68d7b7bee997777);
    AS_UL2(&sharedMemory[0x304]) = make_ulonglong2(0xd6bd6b6bff0df2f2, 0x9154c5c5deb16f6f);
    AS_UL2(&sharedMemory[0x308]) = make_ulonglong2(0x0203010160503030, 0x567d2b2bcea96767);
    AS_UL2(&sharedMemory[0x30C]) = make_ulonglong2(0xb562d7d7e719fefe, 0xec9a76764de6abab);
    AS_UL2(&sharedMemory[0x310]) = make_ulonglong2(0x1f9d82828f45caca, 0xfa877d7d8940c9c9);
    AS_UL2(&sharedMemory[0x314]) = make_ulonglong2(0xb2eb5959ef15fafa, 0xfb0bf0f08ec94747);
    AS_UL2(&sharedMemory[0x318]) = make_ulonglong2(0xb367d4d441ecadad, 0x45eaafaf5ffda2a2);
    AS_UL2(&sharedMemory[0x31C]) = make_ulonglong2(0x53f7a4a423bf9c9c, 0x9b5bc0c0e4967272);
    break;
  case 1:
    AS_UL2(&sharedMemory[0x320]) = make_ulonglong2(0xe11cfdfd75c2b7b7, 0x4c6a26263dae9393);
    AS_UL2(&sharedMemory[0x324]) = make_ulonglong2(0x7e413f3f6c5a3636, 0x834fccccf502f7f7);
    AS_UL2(&sharedMemory[0x328]) = make_ulonglong2(0x51f4a5a5685c3434, 0xf908f1f1d134e5e5);
    AS_UL2(&sharedMemory[0x32C]) = make_ulonglong2(0xab73d8d8e2937171, 0x2a3f151562533131);
    AS_UL2(&sharedMemory[0x330]) = make_ulonglong2(0x9552c7c7080c0404, 0x9d5ec3c346652323);
    AS_UL2(&sharedMemory[0x334]) = make_ulonglong2(0x37a1969630281818, 0x2fb59a9a0a0f0505);
    AS_UL2(&sharedMemory[0x338]) = make_ulonglong2(0x243612120e090707, 0xdf3de2e21b9b8080);
    AS_UL2(&sharedMemory[0x33C]) = make_ulonglong2(0x4e692727cd26ebeb, 0xea9f75757fcdb2b2);
    break;
  case 2:
    AS_UL2(&sharedMemory[0x340]) = make_ulonglong2(0x1d9e8383121b0909, 0x342e1a1a58742c2c);
    AS_UL2(&sharedMemory[0x344]) = make_ulonglong2(0xdcb26e6e362d1b1b, 0x5bfba0a0b4ee5a5a);
    AS_UL2(&sharedMemory[0x348]) = make_ulonglong2(0x764d3b3ba4f65252, 0x7dceb3b3b761d6d6);
    AS_UL2(&sharedMemory[0x34C]) = make_ulonglong2(0xdd3ee3e3527b2929, 0x139784845e712f2f);
    AS_UL2(&sharedMemory[0x350]) = make_ulonglong2(0xb968d1d1a6f55353, 0xc12ceded00000000);
    AS_UL2(&sharedMemory[0x354]) = make_ulonglong2(0xe31ffcfc40602020, 0xb6ed5b5b79c8b1b1);
    AS_UL2(&sharedMemory[0x358]) = make_ulonglong2(0x8d46cbcbd4be6a6a, 0x724b393967d9bebe);
    AS_UL2(&sharedMemory[0x35C]) = make_ulonglong2(0x98d44c4c94de4a4a, 0x854acfcfb0e85858);
    break;
  case 3:
    AS_UL2(&sharedMemory[0x360]) = make_ulonglong2(0xc52aefefbb6bd0d0, 0xed16fbfb4fe5aaaa);
    AS_UL2(&sharedMemory[0x364]) = make_ulonglong2(0x9ad74d4d86c54343, 0x1194858566553333);
    AS_UL2(&sharedMemory[0x368]) = make_ulonglong2(0xe910f9f98acf4545, 0xfe817f7f04060202);
    AS_UL2(&sharedMemory[0x36C]) = make_ulonglong2(0x78443c3ca0f05050, 0x4be3a8a825ba9f9f);
    AS_UL2(&sharedMemory[0x370]) = make_ulonglong2(0x5dfea3a3a2f35151, 0x058a8f8f80c04040);
    AS_UL2(&sharedMemory[0x374]) = make_ulonglong2(0x21bc9d9d3fad9292, 0xf104f5f570483838);
    AS_UL2(&sharedMemory[0x378]) = make_ulonglong2(0x77c1b6b663dfbcbc, 0x42632121af75dada);
    AS_UL2(&sharedMemory[0x37C]) = make_ulonglong2(0xe51affff20301010, 0xbf6dd2d2fd0ef3f3);
    break;
  case 4:
    AS_UL2(&sharedMemory[0x380]) = make_ulonglong2(0x18140c0c814ccdcd, 0xc32fecec26351313);
    AS_UL2(&sharedMemory[0x384]) = make_ulonglong2(0x35a29797bee15f5f, 0x2e39171788cc4444);
    AS_UL2(&sharedMemory[0x388]) = make_ulonglong2(0x55f2a7a79357c4c4, 0x7a473d3dfc827e7e);
    AS_UL2(&sharedMemory[0x38C]) = make_ulonglong2(0xbae75d5dc8ac6464, 0xe6957373322b1919);
    AS_UL2(&sharedMemory[0x390]) = make_ulonglong2(0x19988181c0a06060, 0xa37fdcdc9ed14f4f);
    AS_UL2(&sharedMemory[0x394]) = make_ulonglong2(0x547e2a2a44662222, 0x0b8388883bab9090);
    AS_UL2(&sharedMemory[0x398]) = make_ulonglong2(0xc729eeee8cca4646, 0x283c14146bd3b8b8);
    AS_UL2(&sharedMemory[0x39C]) = make_ulonglong2(0xbce25e5ea779dede, 0xad76dbdb161d0b0b);
    break;
  case 5:
    AS_UL2(&sharedMemory[0x3A0]) = make_ulonglong2(0x64563232db3be0e0, 0x141e0a0a744e3a3a);
    AS_UL2(&sharedMemory[0x3A4]) = make_ulonglong2(0x0c0a060692db4949, 0xb8e45c5c486c2424);
    AS_UL2(&sharedMemory[0x3A8]) = make_ulonglong2(0xbd6ed3d39f5dc2c2, 0xc4a6626243efacac);
    AS_UL2(&sharedMemory[0x3AC]) = make_ulonglong2(0x31a4959539a89191, 0xf28b7979d337e4e4);
    AS_UL2(&sharedMemory[0x3B0]) = make_ulonglong2(0x8b43c8c8d532e7e7, 0xdab76d6d6e593737);
    AS_UL2(&sharedMemory[0x3B4]) = make_ulonglong2(0xb164d5d5018c8d8d, 0x49e0a9a99cd24e4e);
    AS_UL2(&sharedMemory[0x3B8]) = make_ulonglong2(0xacfa5656d8b46c6c, 0xcf25eaeaf307f4f4);
    AS_UL2(&sharedMemory[0x3BC]) = make_ulonglong2(0xf48e7a7acaaf6565, 0x1018080847e9aeae);
    break;
  case 6:
    AS_UL2(&sharedMemory[0x3C0]) = make_ulonglong2(0xf08878786fd5baba, 0x5c722e2e4a6f2525);
    AS_UL2(&sharedMemory[0x3C4]) = make_ulonglong2(0x57f1a6a638241c1c, 0x9751c6c673c7b4b4);
    AS_UL2(&sharedMemory[0x3C8]) = make_ulonglong2(0xa17cddddcb23e8e8, 0x3e211f1fe89c7474);
    AS_UL2(&sharedMemory[0x3CC]) = make_ulonglong2(0x61dcbdbd96dd4b4b, 0x0f858a8a0d868b8b);
    AS_UL2(&sharedMemory[0x3D0]) = make_ulonglong2(0x7c423e3ee0907070, 0xccaa666671c4b5b5);
    AS_UL2(&sharedMemory[0x3D4]) = make_ulonglong2(0x0605030390d84848, 0x1c120e0ef701f6f6);
    AS_UL2(&sharedMemory[0x3D8]) = make_ulonglong2(0x6a5f3535c2a36161, 0x69d0b9b9aef95757);
    AS_UL2(&sharedMemory[0x3DC]) = make_ulonglong2(0x9958c1c117918686, 0x27b99e9e3a271d1d);
    break;
  case 7:
    AS_UL2(&sharedMemory[0x3E0]) = make_ulonglong2(0xeb13f8f8d938e1e1, 0x223311112bb39898);
    AS_UL2(&sharedMemory[0x3E4]) = make_ulonglong2(0xa970d9d9d2bb6969, 0x33a7949407898e8e);
    AS_UL2(&sharedMemory[0x3E8]) = make_ulonglong2(0x3c221e1e2db69b9b, 0xc920e9e915928787);
    AS_UL2(&sharedMemory[0x3EC]) = make_ulonglong2(0xaaff55558749cece, 0xa57adfdf50782828);
    AS_UL2(&sharedMemory[0x3F0]) = make_ulonglong2(0x59f8a1a1038f8c8c, 0x1a170d0d09808989);
    AS_UL2(&sharedMemory[0x3F4]) = make_ulonglong2(0xd731e6e665dabfbf, 0xd0b8686884c64242);
    AS_UL2(&sharedMemory[0x3F8]) = make_ulonglong2(0x29b0999982c34141, 0x1e110f0f5a772d2d);
    AS_UL2(&sharedMemory[0x3FC]) = make_ulonglong2(0xa8fc54547bcbb0b0, 0x2c3a16166dd6bbbb);
    break;
  }
}

__device__ __forceinline__
void cn_aes_gpu_init_u4(uint32_t* sharedMemory)
{
  // AES 0
  switch (threadIdx.x) {
  case 0:
    AS_UINT4(&sharedMemory[0x000]) = make_uint4(0xa56363c6, 0x847c7cf8, 0x997777ee, 0x8d7b7bf6);
    AS_UINT4(&sharedMemory[0x004]) = make_uint4(0x0df2f2ff, 0xbd6b6bd6, 0xb16f6fde, 0x54c5c591);
    AS_UINT4(&sharedMemory[0x008]) = make_uint4(0x50303060, 0x03010102, 0xa96767ce, 0x7d2b2b56);
    AS_UINT4(&sharedMemory[0x00C]) = make_uint4(0x19fefee7, 0x62d7d7b5, 0xe6abab4d, 0x9a7676ec);
    AS_UINT4(&sharedMemory[0x010]) = make_uint4(0x45caca8f, 0x9d82821f, 0x40c9c989, 0x877d7dfa);
    AS_UINT4(&sharedMemory[0x014]) = make_uint4(0x15fafaef, 0xeb5959b2, 0xc947478e, 0x0bf0f0fb);
    AS_UINT4(&sharedMemory[0x018]) = make_uint4(0xecadad41, 0x67d4d4b3, 0xfda2a25f, 0xeaafaf45);
    AS_UINT4(&sharedMemory[0x01C]) = make_uint4(0xbf9c9c23, 0xf7a4a453, 0x967272e4, 0x5bc0c09b);
    break;
  case 1:
    AS_UINT4(&sharedMemory[0x020]) = make_uint4(0xc2b7b775, 0x1cfdfde1, 0xae93933d, 0x6a26264c);
    AS_UINT4(&sharedMemory[0x024]) = make_uint4(0x5a36366c, 0x413f3f7e, 0x02f7f7f5, 0x4fcccc83);
    AS_UINT4(&sharedMemory[0x028]) = make_uint4(0x5c343468, 0xf4a5a551, 0x34e5e5d1, 0x08f1f1f9);
    AS_UINT4(&sharedMemory[0x02C]) = make_uint4(0x937171e2, 0x73d8d8ab, 0x53313162, 0x3f15152a);
    AS_UINT4(&sharedMemory[0x030]) = make_uint4(0x0c040408, 0x52c7c795, 0x65232346, 0x5ec3c39d);
    AS_UINT4(&sharedMemory[0x034]) = make_uint4(0x28181830, 0xa1969637, 0x0f05050a, 0xb59a9a2f);
    AS_UINT4(&sharedMemory[0x038]) = make_uint4(0x0907070e, 0x36121224, 0x9b80801b, 0x3de2e2df);
    AS_UINT4(&sharedMemory[0x03C]) = make_uint4(0x26ebebcd, 0x6927274e, 0xcdb2b27f, 0x9f7575ea);
    break;
  case 2:
    AS_UINT4(&sharedMemory[0x040]) = make_uint4(0x1b090912, 0x9e83831d, 0x742c2c58, 0x2e1a1a34);
    AS_UINT4(&sharedMemory[0x044]) = make_uint4(0x2d1b1b36, 0xb26e6edc, 0xee5a5ab4, 0xfba0a05b);
    AS_UINT4(&sharedMemory[0x048]) = make_uint4(0xf65252a4, 0x4d3b3b76, 0x61d6d6b7, 0xceb3b37d);
    AS_UINT4(&sharedMemory[0x04C]) = make_uint4(0x7b292952, 0x3ee3e3dd, 0x712f2f5e, 0x97848413);
    AS_UINT4(&sharedMemory[0x050]) = make_uint4(0xf55353a6, 0x68d1d1b9, 0x00000000, 0x2cededc1);
    AS_UINT4(&sharedMemory[0x054]) = make_uint4(0x60202040, 0x1ffcfce3, 0xc8b1b179, 0xed5b5bb6);
    AS_UINT4(&sharedMemory[0x058]) = make_uint4(0xbe6a6ad4, 0x46cbcb8d, 0xd9bebe67, 0x4b393972);
    AS_UINT4(&sharedMemory[0x05C]) = make_uint4(0xde4a4a94, 0xd44c4c98, 0xe85858b0, 0x4acfcf85);
    break;
  case 3:
    AS_UINT4(&sharedMemory[0x060]) = make_uint4(0x6bd0d0bb, 0x2aefefc5, 0xe5aaaa4f, 0x16fbfbed);
    AS_UINT4(&sharedMemory[0x064]) = make_uint4(0xc5434386, 0xd74d4d9a, 0x55333366, 0x94858511);
    AS_UINT4(&sharedMemory[0x068]) = make_uint4(0xcf45458a, 0x10f9f9e9, 0x06020204, 0x817f7ffe);
    AS_UINT4(&sharedMemory[0x06C]) = make_uint4(0xf05050a0, 0x443c3c78, 0xba9f9f25, 0xe3a8a84b);
    AS_UINT4(&sharedMemory[0x070]) = make_uint4(0xf35151a2, 0xfea3a35d, 0xc0404080, 0x8a8f8f05);
    AS_UINT4(&sharedMemory[0x074]) = make_uint4(0xad92923f, 0xbc9d9d21, 0x48383870, 0x04f5f5f1);
    AS_UINT4(&sharedMemory[0x078]) = make_uint4(0xdfbcbc63, 0xc1b6b677, 0x75dadaaf, 0x63212142);
    AS_UINT4(&sharedMemory[0x07C]) = make_uint4(0x30101020, 0x1affffe5, 0x0ef3f3fd, 0x6dd2d2bf);
    break;
  case 4:
    AS_UINT4(&sharedMemory[0x080]) = make_uint4(0x4ccdcd81, 0x140c0c18, 0x35131326, 0x2fececc3);
    AS_UINT4(&sharedMemory[0x084]) = make_uint4(0xe15f5fbe, 0xa2979735, 0xcc444488, 0x3917172e);
    AS_UINT4(&sharedMemory[0x088]) = make_uint4(0x57c4c493, 0xf2a7a755, 0x827e7efc, 0x473d3d7a);
    AS_UINT4(&sharedMemory[0x08C]) = make_uint4(0xac6464c8, 0xe75d5dba, 0x2b191932, 0x957373e6);
    AS_UINT4(&sharedMemory[0x090]) = make_uint4(0xa06060c0, 0x98818119, 0xd14f4f9e, 0x7fdcdca3);
    AS_UINT4(&sharedMemory[0x094]) = make_uint4(0x66222244, 0x7e2a2a54, 0xab90903b, 0x8388880b);
    AS_UINT4(&sharedMemory[0x098]) = make_uint4(0xca46468c, 0x29eeeec7, 0xd3b8b86b, 0x3c141428);
    AS_UINT4(&sharedMemory[0x09C]) = make_uint4(0x79dedea7, 0xe25e5ebc, 0x1d0b0b16, 0x76dbdbad);
    break;
  case 5:
    AS_UINT4(&sharedMemory[0x0A0]) = make_uint4(0x3be0e0db, 0x56323264, 0x4e3a3a74, 0x1e0a0a14);
    AS_UINT4(&sharedMemory[0x0A4]) = make_uint4(0xdb494992, 0x0a06060c, 0x6c242448, 0xe45c5cb8);
    AS_UINT4(&sharedMemory[0x0A8]) = make_uint4(0x5dc2c29f, 0x6ed3d3bd, 0xefacac43, 0xa66262c4);
    AS_UINT4(&sharedMemory[0x0AC]) = make_uint4(0xa8919139, 0xa4959531, 0x37e4e4d3, 0x8b7979f2);
    AS_UINT4(&sharedMemory[0x0B0]) = make_uint4(0x32e7e7d5, 0x43c8c88b, 0x5937376e, 0xb76d6dda);
    AS_UINT4(&sharedMemory[0x0B4]) = make_uint4(0x8c8d8d01, 0x64d5d5b1, 0xd24e4e9c, 0xe0a9a949);
    AS_UINT4(&sharedMemory[0x0B8]) = make_uint4(0xb46c6cd8, 0xfa5656ac, 0x07f4f4f3, 0x25eaeacf);
    AS_UINT4(&sharedMemory[0x0BC]) = make_uint4(0xaf6565ca, 0x8e7a7af4, 0xe9aeae47, 0x18080810);
    break;
  case 6:
    AS_UINT4(&sharedMemory[0x0C0]) = make_uint4(0xd5baba6f, 0x887878f0, 0x6f25254a, 0x722e2e5c);
    AS_UINT4(&sharedMemory[0x0C4]) = make_uint4(0x241c1c38, 0xf1a6a657, 0xc7b4b473, 0x51c6c697);
    AS_UINT4(&sharedMemory[0x0C8]) = make_uint4(0x23e8e8cb, 0x7cdddda1, 0x9c7474e8, 0x211f1f3e);
    AS_UINT4(&sharedMemory[0x0CC]) = make_uint4(0xdd4b4b96, 0xdcbdbd61, 0x868b8b0d, 0x858a8a0f);
    AS_UINT4(&sharedMemory[0x0D0]) = make_uint4(0x907070e0, 0x423e3e7c, 0xc4b5b571, 0xaa6666cc);
    AS_UINT4(&sharedMemory[0x0D4]) = make_uint4(0xd8484890, 0x05030306, 0x01f6f6f7, 0x120e0e1c);
    AS_UINT4(&sharedMemory[0x0D8]) = make_uint4(0xa36161c2, 0x5f35356a, 0xf95757ae, 0xd0b9b969);
    AS_UINT4(&sharedMemory[0x0DC]) = make_uint4(0x91868617, 0x58c1c199, 0x271d1d3a, 0xb99e9e27);
    break;
  case 7:
    AS_UINT4(&sharedMemory[0x0E0]) = make_uint4(0x38e1e1d9, 0x13f8f8eb, 0xb398982b, 0x33111122);
    AS_UINT4(&sharedMemory[0x0E4]) = make_uint4(0xbb6969d2, 0x70d9d9a9, 0x898e8e07, 0xa7949433);
    AS_UINT4(&sharedMemory[0x0E8]) = make_uint4(0xb69b9b2d, 0x221e1e3c, 0x92878715, 0x20e9e9c9);
    AS_UINT4(&sharedMemory[0x0EC]) = make_uint4(0x49cece87, 0xff5555aa, 0x78282850, 0x7adfdfa5);
    AS_UINT4(&sharedMemory[0x0F0]) = make_uint4(0x8f8c8c03, 0xf8a1a159, 0x80898909, 0x170d0d1a);
    AS_UINT4(&sharedMemory[0x0F4]) = make_uint4(0xdabfbf65, 0x31e6e6d7, 0xc6424284, 0xb86868d0);
    AS_UINT4(&sharedMemory[0x0F8]) = make_uint4(0xc3414182, 0xb0999929, 0x772d2d5a, 0x110f0f1e);
    AS_UINT4(&sharedMemory[0x0FC]) = make_uint4(0xcbb0b07b, 0xfc5454a8, 0xd6bbbb6d, 0x3a16162c);
    break;
  }
  // AES 1
  switch (threadIdx.x) {
  case 0:
    AS_UINT4(&sharedMemory[0x100]) = make_uint4(0x6363c6a5, 0x7c7cf884, 0x7777ee99, 0x7b7bf68d);
    AS_UINT4(&sharedMemory[0x104]) = make_uint4(0xf2f2ff0d, 0x6b6bd6bd, 0x6f6fdeb1, 0xc5c59154);
    AS_UINT4(&sharedMemory[0x108]) = make_uint4(0x30306050, 0x01010203, 0x6767cea9, 0x2b2b567d);
    AS_UINT4(&sharedMemory[0x10C]) = make_uint4(0xfefee719, 0xd7d7b562, 0xabab4de6, 0x7676ec9a);
    AS_UINT4(&sharedMemory[0x110]) = make_uint4(0xcaca8f45, 0x82821f9d, 0xc9c98940, 0x7d7dfa87);
    AS_UINT4(&sharedMemory[0x114]) = make_uint4(0xfafaef15, 0x5959b2eb, 0x47478ec9, 0xf0f0fb0b);
    AS_UINT4(&sharedMemory[0x118]) = make_uint4(0xadad41ec, 0xd4d4b367, 0xa2a25ffd, 0xafaf45ea);
    AS_UINT4(&sharedMemory[0x11C]) = make_uint4(0x9c9c23bf, 0xa4a453f7, 0x7272e496, 0xc0c09b5b);
    break;
  case 1:
    AS_UINT4(&sharedMemory[0x120]) = make_uint4(0xb7b775c2, 0xfdfde11c, 0x93933dae, 0x26264c6a);
    AS_UINT4(&sharedMemory[0x124]) = make_uint4(0x36366c5a, 0x3f3f7e41, 0xf7f7f502, 0xcccc834f);
    AS_UINT4(&sharedMemory[0x128]) = make_uint4(0x3434685c, 0xa5a551f4, 0xe5e5d134, 0xf1f1f908);
    AS_UINT4(&sharedMemory[0x12C]) = make_uint4(0x7171e293, 0xd8d8ab73, 0x31316253, 0x15152a3f);
    AS_UINT4(&sharedMemory[0x130]) = make_uint4(0x0404080c, 0xc7c79552, 0x23234665, 0xc3c39d5e);
    AS_UINT4(&sharedMemory[0x134]) = make_uint4(0x18183028, 0x969637a1, 0x05050a0f, 0x9a9a2fb5);
    AS_UINT4(&sharedMemory[0x138]) = make_uint4(0x07070e09, 0x12122436, 0x80801b9b, 0xe2e2df3d);
    AS_UINT4(&sharedMemory[0x13C]) = make_uint4(0xebebcd26, 0x27274e69, 0xb2b27fcd, 0x7575ea9f);
    break;
  case 2:
    AS_UINT4(&sharedMemory[0x140]) = make_uint4(0x0909121b, 0x83831d9e, 0x2c2c5874, 0x1a1a342e);
    AS_UINT4(&sharedMemory[0x144]) = make_uint4(0x1b1b362d, 0x6e6edcb2, 0x5a5ab4ee, 0xa0a05bfb);
    AS_UINT4(&sharedMemory[0x148]) = make_uint4(0x5252a4f6, 0x3b3b764d, 0xd6d6b761, 0xb3b37dce);
    AS_UINT4(&sharedMemory[0x14C]) = make_uint4(0x2929527b, 0xe3e3dd3e, 0x2f2f5e71, 0x84841397);
    AS_UINT4(&sharedMemory[0x150]) = make_uint4(0x5353a6f5, 0xd1d1b968, 0x00000000, 0xededc12c);
    AS_UINT4(&sharedMemory[0x154]) = make_uint4(0x20204060, 0xfcfce31f, 0xb1b179c8, 0x5b5bb6ed);
    AS_UINT4(&sharedMemory[0x158]) = make_uint4(0x6a6ad4be, 0xcbcb8d46, 0xbebe67d9, 0x3939724b);
    AS_UINT4(&sharedMemory[0x15C]) = make_uint4(0x4a4a94de, 0x4c4c98d4, 0x5858b0e8, 0xcfcf854a);
    break;
  case 3:
    AS_UINT4(&sharedMemory[0x160]) = make_uint4(0xd0d0bb6b, 0xefefc52a, 0xaaaa4fe5, 0xfbfbed16);
    AS_UINT4(&sharedMemory[0x164]) = make_uint4(0x434386c5, 0x4d4d9ad7, 0x33336655, 0x85851194);
    AS_UINT4(&sharedMemory[0x168]) = make_uint4(0x45458acf, 0xf9f9e910, 0x02020406, 0x7f7ffe81);
    AS_UINT4(&sharedMemory[0x16C]) = make_uint4(0x5050a0f0, 0x3c3c7844, 0x9f9f25ba, 0xa8a84be3);
    AS_UINT4(&sharedMemory[0x170]) = make_uint4(0x5151a2f3, 0xa3a35dfe, 0x404080c0, 0x8f8f058a);
    AS_UINT4(&sharedMemory[0x174]) = make_uint4(0x92923fad, 0x9d9d21bc, 0x38387048, 0xf5f5f104);
    AS_UINT4(&sharedMemory[0x178]) = make_uint4(0xbcbc63df, 0xb6b677c1, 0xdadaaf75, 0x21214263);
    AS_UINT4(&sharedMemory[0x17C]) = make_uint4(0x10102030, 0xffffe51a, 0xf3f3fd0e, 0xd2d2bf6d);
    break;
  case 4:
    AS_UINT4(&sharedMemory[0x180]) = make_uint4(0xcdcd814c, 0x0c0c1814, 0x13132635, 0xececc32f);
    AS_UINT4(&sharedMemory[0x184]) = make_uint4(0x5f5fbee1, 0x979735a2, 0x444488cc, 0x17172e39);
    AS_UINT4(&sharedMemory[0x188]) = make_uint4(0xc4c49357, 0xa7a755f2, 0x7e7efc82, 0x3d3d7a47);
    AS_UINT4(&sharedMemory[0x18C]) = make_uint4(0x6464c8ac, 0x5d5dbae7, 0x1919322b, 0x7373e695);
    AS_UINT4(&sharedMemory[0x190]) = make_uint4(0x6060c0a0, 0x81811998, 0x4f4f9ed1, 0xdcdca37f);
    AS_UINT4(&sharedMemory[0x194]) = make_uint4(0x22224466, 0x2a2a547e, 0x90903bab, 0x88880b83);
    AS_UINT4(&sharedMemory[0x198]) = make_uint4(0x46468cca, 0xeeeec729, 0xb8b86bd3, 0x1414283c);
    AS_UINT4(&sharedMemory[0x19C]) = make_uint4(0xdedea779, 0x5e5ebce2, 0x0b0b161d, 0xdbdbad76);
    break;
  case 5:
    AS_UINT4(&sharedMemory[0x1A0]) = make_uint4(0xe0e0db3b, 0x32326456, 0x3a3a744e, 0x0a0a141e);
    AS_UINT4(&sharedMemory[0x1A4]) = make_uint4(0x494992db, 0x06060c0a, 0x2424486c, 0x5c5cb8e4);
    AS_UINT4(&sharedMemory[0x1A8]) = make_uint4(0xc2c29f5d, 0xd3d3bd6e, 0xacac43ef, 0x6262c4a6);
    AS_UINT4(&sharedMemory[0x1AC]) = make_uint4(0x919139a8, 0x959531a4, 0xe4e4d337, 0x7979f28b);
    AS_UINT4(&sharedMemory[0x1B0]) = make_uint4(0xe7e7d532, 0xc8c88b43, 0x37376e59, 0x6d6ddab7);
    AS_UINT4(&sharedMemory[0x1B4]) = make_uint4(0x8d8d018c, 0xd5d5b164, 0x4e4e9cd2, 0xa9a949e0);
    AS_UINT4(&sharedMemory[0x1B8]) = make_uint4(0x6c6cd8b4, 0x5656acfa, 0xf4f4f307, 0xeaeacf25);
    AS_UINT4(&sharedMemory[0x1BC]) = make_uint4(0x6565caaf, 0x7a7af48e, 0xaeae47e9, 0x08081018);
    break;
  case 6:
    AS_UINT4(&sharedMemory[0x1C0]) = make_uint4(0xbaba6fd5, 0x7878f088, 0x25254a6f, 0x2e2e5c72);
    AS_UINT4(&sharedMemory[0x1C4]) = make_uint4(0x1c1c3824, 0xa6a657f1, 0xb4b473c7, 0xc6c69751);
    AS_UINT4(&sharedMemory[0x1C8]) = make_uint4(0xe8e8cb23, 0xdddda17c, 0x7474e89c, 0x1f1f3e21);
    AS_UINT4(&sharedMemory[0x1CC]) = make_uint4(0x4b4b96dd, 0xbdbd61dc, 0x8b8b0d86, 0x8a8a0f85);
    AS_UINT4(&sharedMemory[0x1D0]) = make_uint4(0x7070e090, 0x3e3e7c42, 0xb5b571c4, 0x6666ccaa);
    AS_UINT4(&sharedMemory[0x1D4]) = make_uint4(0x484890d8, 0x03030605, 0xf6f6f701, 0x0e0e1c12);
    AS_UINT4(&sharedMemory[0x1D8]) = make_uint4(0x6161c2a3, 0x35356a5f, 0x5757aef9, 0xb9b969d0);
    AS_UINT4(&sharedMemory[0x1DC]) = make_uint4(0x86861791, 0xc1c19958, 0x1d1d3a27, 0x9e9e27b9);
    break;
  case 7:
    AS_UINT4(&sharedMemory[0x1E0]) = make_uint4(0xe1e1d938, 0xf8f8eb13, 0x98982bb3, 0x11112233);
    AS_UINT4(&sharedMemory[0x1E4]) = make_uint4(0x6969d2bb, 0xd9d9a970, 0x8e8e0789, 0x949433a7);
    AS_UINT4(&sharedMemory[0x1E8]) = make_uint4(0x9b9b2db6, 0x1e1e3c22, 0x87871592, 0xe9e9c920);
    AS_UINT4(&sharedMemory[0x1EC]) = make_uint4(0xcece8749, 0x5555aaff, 0x28285078, 0xdfdfa57a);
    AS_UINT4(&sharedMemory[0x1F0]) = make_uint4(0x8c8c038f, 0xa1a159f8, 0x89890980, 0x0d0d1a17);
    AS_UINT4(&sharedMemory[0x1F4]) = make_uint4(0xbfbf65da, 0xe6e6d731, 0x424284c6, 0x6868d0b8);
    AS_UINT4(&sharedMemory[0x1F8]) = make_uint4(0x414182c3, 0x999929b0, 0x2d2d5a77, 0x0f0f1e11);
    AS_UINT4(&sharedMemory[0x1FC]) = make_uint4(0xb0b07bcb, 0x5454a8fc, 0xbbbb6dd6, 0x16162c3a);
    break;
  }
  // AES 2
  switch (threadIdx.x) {
  case 0:
    AS_UINT4(&sharedMemory[0x200]) = make_uint4(0x63c6a563, 0x7cf8847c, 0x77ee9977, 0x7bf68d7b);
    AS_UINT4(&sharedMemory[0x204]) = make_uint4(0xf2ff0df2, 0x6bd6bd6b, 0x6fdeb16f, 0xc59154c5);
    AS_UINT4(&sharedMemory[0x208]) = make_uint4(0x30605030, 0x01020301, 0x67cea967, 0x2b567d2b);
    AS_UINT4(&sharedMemory[0x20C]) = make_uint4(0xfee719fe, 0xd7b562d7, 0xab4de6ab, 0x76ec9a76);
    AS_UINT4(&sharedMemory[0x210]) = make_uint4(0xca8f45ca, 0x821f9d82, 0xc98940c9, 0x7dfa877d);
    AS_UINT4(&sharedMemory[0x214]) = make_uint4(0xfaef15fa, 0x59b2eb59, 0x478ec947, 0xf0fb0bf0);
    AS_UINT4(&sharedMemory[0x218]) = make_uint4(0xad41ecad, 0xd4b367d4, 0xa25ffda2, 0xaf45eaaf);
    AS_UINT4(&sharedMemory[0x21C]) = make_uint4(0x9c23bf9c, 0xa453f7a4, 0x72e49672, 0xc09b5bc0);
    break;
  case 1:
    AS_UINT4(&sharedMemory[0x220]) = make_uint4(0xb775c2b7, 0xfde11cfd, 0x933dae93, 0x264c6a26);
    AS_UINT4(&sharedMemory[0x224]) = make_uint4(0x366c5a36, 0x3f7e413f, 0xf7f502f7, 0xcc834fcc);
    AS_UINT4(&sharedMemory[0x228]) = make_uint4(0x34685c34, 0xa551f4a5, 0xe5d134e5, 0xf1f908f1);
    AS_UINT4(&sharedMemory[0x22C]) = make_uint4(0x71e29371, 0xd8ab73d8, 0x31625331, 0x152a3f15);
    AS_UINT4(&sharedMemory[0x230]) = make_uint4(0x04080c04, 0xc79552c7, 0x23466523, 0xc39d5ec3);
    AS_UINT4(&sharedMemory[0x234]) = make_uint4(0x18302818, 0x9637a196, 0x050a0f05, 0x9a2fb59a);
    AS_UINT4(&sharedMemory[0x238]) = make_uint4(0x070e0907, 0x12243612, 0x801b9b80, 0xe2df3de2);
    AS_UINT4(&sharedMemory[0x23C]) = make_uint4(0xebcd26eb, 0x274e6927, 0xb27fcdb2, 0x75ea9f75);
    break;
  case 2:
    AS_UINT4(&sharedMemory[0x240]) = make_uint4(0x09121b09, 0x831d9e83, 0x2c58742c, 0x1a342e1a);
    AS_UINT4(&sharedMemory[0x244]) = make_uint4(0x1b362d1b, 0x6edcb26e, 0x5ab4ee5a, 0xa05bfba0);
    AS_UINT4(&sharedMemory[0x248]) = make_uint4(0x52a4f652, 0x3b764d3b, 0xd6b761d6, 0xb37dceb3);
    AS_UINT4(&sharedMemory[0x24C]) = make_uint4(0x29527b29, 0xe3dd3ee3, 0x2f5e712f, 0x84139784);
    AS_UINT4(&sharedMemory[0x250]) = make_uint4(0x53a6f553, 0xd1b968d1, 0x00000000, 0xedc12ced);
    AS_UINT4(&sharedMemory[0x254]) = make_uint4(0x20406020, 0xfce31ffc, 0xb179c8b1, 0x5bb6ed5b);
    AS_UINT4(&sharedMemory[0x258]) = make_uint4(0x6ad4be6a, 0xcb8d46cb, 0xbe67d9be, 0x39724b39);
    AS_UINT4(&sharedMemory[0x25C]) = make_uint4(0x4a94de4a, 0x4c98d44c, 0x58b0e858, 0xcf854acf);
    break;
  case 3:
    AS_UINT4(&sharedMemory[0x260]) = make_uint4(0xd0bb6bd0, 0xefc52aef, 0xaa4fe5aa, 0xfbed16fb);
    AS_UINT4(&sharedMemory[0x264]) = make_uint4(0x4386c543, 0x4d9ad74d, 0x33665533, 0x85119485);
    AS_UINT4(&sharedMemory[0x268]) = make_uint4(0x458acf45, 0xf9e910f9, 0x02040602, 0x7ffe817f);
    AS_UINT4(&sharedMemory[0x26C]) = make_uint4(0x50a0f050, 0x3c78443c, 0x9f25ba9f, 0xa84be3a8);
    AS_UINT4(&sharedMemory[0x270]) = make_uint4(0x51a2f351, 0xa35dfea3, 0x4080c040, 0x8f058a8f);
    AS_UINT4(&sharedMemory[0x274]) = make_uint4(0x923fad92, 0x9d21bc9d, 0x38704838, 0xf5f104f5);
    AS_UINT4(&sharedMemory[0x278]) = make_uint4(0xbc63dfbc, 0xb677c1b6, 0xdaaf75da, 0x21426321);
    AS_UINT4(&sharedMemory[0x27C]) = make_uint4(0x10203010, 0xffe51aff, 0xf3fd0ef3, 0xd2bf6dd2);
    break;
  case 4:
    AS_UINT4(&sharedMemory[0x280]) = make_uint4(0xcd814ccd, 0x0c18140c, 0x13263513, 0xecc32fec);
    AS_UINT4(&sharedMemory[0x284]) = make_uint4(0x5fbee15f, 0x9735a297, 0x4488cc44, 0x172e3917);
    AS_UINT4(&sharedMemory[0x288]) = make_uint4(0xc49357c4, 0xa755f2a7, 0x7efc827e, 0x3d7a473d);
    AS_UINT4(&sharedMemory[0x28C]) = make_uint4(0x64c8ac64, 0x5dbae75d, 0x19322b19, 0x73e69573);
    AS_UINT4(&sharedMemory[0x290]) = make_uint4(0x60c0a060, 0x81199881, 0x4f9ed14f, 0xdca37fdc);
    AS_UINT4(&sharedMemory[0x294]) = make_uint4(0x22446622, 0x2a547e2a, 0x903bab90, 0x880b8388);
    AS_UINT4(&sharedMemory[0x298]) = make_uint4(0x468cca46, 0xeec729ee, 0xb86bd3b8, 0x14283c14);
    AS_UINT4(&sharedMemory[0x29C]) = make_uint4(0xdea779de, 0x5ebce25e, 0x0b161d0b, 0xdbad76db);
    break;
  case 5:
    AS_UINT4(&sharedMemory[0x2A0]) = make_uint4(0xe0db3be0, 0x32645632, 0x3a744e3a, 0x0a141e0a);
    AS_UINT4(&sharedMemory[0x2A4]) = make_uint4(0x4992db49, 0x060c0a06, 0x24486c24, 0x5cb8e45c);
    AS_UINT4(&sharedMemory[0x2A8]) = make_uint4(0xc29f5dc2, 0xd3bd6ed3, 0xac43efac, 0x62c4a662);
    AS_UINT4(&sharedMemory[0x2AC]) = make_uint4(0x9139a891, 0x9531a495, 0xe4d337e4, 0x79f28b79);
    AS_UINT4(&sharedMemory[0x2B0]) = make_uint4(0xe7d532e7, 0xc88b43c8, 0x376e5937, 0x6ddab76d);
    AS_UINT4(&sharedMemory[0x2B4]) = make_uint4(0x8d018c8d, 0xd5b164d5, 0x4e9cd24e, 0xa949e0a9);
    AS_UINT4(&sharedMemory[0x2B8]) = make_uint4(0x6cd8b46c, 0x56acfa56, 0xf4f307f4, 0xeacf25ea);
    AS_UINT4(&sharedMemory[0x2BC]) = make_uint4(0x65caaf65, 0x7af48e7a, 0xae47e9ae, 0x08101808);
    break;
  case 6:
    AS_UINT4(&sharedMemory[0x2C0]) = make_uint4(0xba6fd5ba, 0x78f08878, 0x254a6f25, 0x2e5c722e);
    AS_UINT4(&sharedMemory[0x2C4]) = make_uint4(0x1c38241c, 0xa657f1a6, 0xb473c7b4, 0xc69751c6);
    AS_UINT4(&sharedMemory[0x2C8]) = make_uint4(0xe8cb23e8, 0xdda17cdd, 0x74e89c74, 0x1f3e211f);
    AS_UINT4(&sharedMemory[0x2CC]) = make_uint4(0x4b96dd4b, 0xbd61dcbd, 0x8b0d868b, 0x8a0f858a);
    AS_UINT4(&sharedMemory[0x2D0]) = make_uint4(0x70e09070, 0x3e7c423e, 0xb571c4b5, 0x66ccaa66);
    AS_UINT4(&sharedMemory[0x2D4]) = make_uint4(0x4890d848, 0x03060503, 0xf6f701f6, 0x0e1c120e);
    AS_UINT4(&sharedMemory[0x2D8]) = make_uint4(0x61c2a361, 0x356a5f35, 0x57aef957, 0xb969d0b9);
    AS_UINT4(&sharedMemory[0x2DC]) = make_uint4(0x86179186, 0xc19958c1, 0x1d3a271d, 0x9e27b99e);
    break;
  case 7:
    AS_UINT4(&sharedMemory[0x2E0]) = make_uint4(0xe1d938e1, 0xf8eb13f8, 0x982bb398, 0x11223311);
    AS_UINT4(&sharedMemory[0x2E4]) = make_uint4(0x69d2bb69, 0xd9a970d9, 0x8e07898e, 0x9433a794);
    AS_UINT4(&sharedMemory[0x2E8]) = make_uint4(0x9b2db69b, 0x1e3c221e, 0x87159287, 0xe9c920e9);
    AS_UINT4(&sharedMemory[0x2EC]) = make_uint4(0xce8749ce, 0x55aaff55, 0x28507828, 0xdfa57adf);
    AS_UINT4(&sharedMemory[0x2F0]) = make_uint4(0x8c038f8c, 0xa159f8a1, 0x89098089, 0x0d1a170d);
    AS_UINT4(&sharedMemory[0x2F4]) = make_uint4(0xbf65dabf, 0xe6d731e6, 0x4284c642, 0x68d0b868);
    AS_UINT4(&sharedMemory[0x2F8]) = make_uint4(0x4182c341, 0x9929b099, 0x2d5a772d, 0x0f1e110f);
    AS_UINT4(&sharedMemory[0x2FC]) = make_uint4(0xb07bcbb0, 0x54a8fc54, 0xbb6dd6bb, 0x162c3a16);
    break;
  }
  // AES 3
  switch (threadIdx.x) {
  case 0:
    AS_UINT4(&sharedMemory[0x300]) = make_uint4(0xc6a56363, 0xf8847c7c, 0xee997777, 0xf68d7b7b);
    AS_UINT4(&sharedMemory[0x304]) = make_uint4(0xff0df2f2, 0xd6bd6b6b, 0xdeb16f6f, 0x9154c5c5);
    AS_UINT4(&sharedMemory[0x308]) = make_uint4(0x60503030, 0x02030101, 0xcea96767, 0x567d2b2b);
    AS_UINT4(&sharedMemory[0x30C]) = make_uint4(0xe719fefe, 0xb562d7d7, 0x4de6abab, 0xec9a7676);
    AS_UINT4(&sharedMemory[0x310]) = make_uint4(0x8f45caca, 0x1f9d8282, 0x8940c9c9, 0xfa877d7d);
    AS_UINT4(&sharedMemory[0x314]) = make_uint4(0xef15fafa, 0xb2eb5959, 0x8ec94747, 0xfb0bf0f0);
    AS_UINT4(&sharedMemory[0x318]) = make_uint4(0x41ecadad, 0xb367d4d4, 0x5ffda2a2, 0x45eaafaf);
    AS_UINT4(&sharedMemory[0x31C]) = make_uint4(0x23bf9c9c, 0x53f7a4a4, 0xe4967272, 0x9b5bc0c0);
    break;
  case 1:
    AS_UINT4(&sharedMemory[0x320]) = make_uint4(0x75c2b7b7, 0xe11cfdfd, 0x3dae9393, 0x4c6a2626);
    AS_UINT4(&sharedMemory[0x324]) = make_uint4(0x6c5a3636, 0x7e413f3f, 0xf502f7f7, 0x834fcccc);
    AS_UINT4(&sharedMemory[0x328]) = make_uint4(0x685c3434, 0x51f4a5a5, 0xd134e5e5, 0xf908f1f1);
    AS_UINT4(&sharedMemory[0x32C]) = make_uint4(0xe2937171, 0xab73d8d8, 0x62533131, 0x2a3f1515);
    AS_UINT4(&sharedMemory[0x330]) = make_uint4(0x080c0404, 0x9552c7c7, 0x46652323, 0x9d5ec3c3);
    AS_UINT4(&sharedMemory[0x334]) = make_uint4(0x30281818, 0x37a19696, 0x0a0f0505, 0x2fb59a9a);
    AS_UINT4(&sharedMemory[0x338]) = make_uint4(0x0e090707, 0x24361212, 0x1b9b8080, 0xdf3de2e2);
    AS_UINT4(&sharedMemory[0x33C]) = make_uint4(0xcd26ebeb, 0x4e692727, 0x7fcdb2b2, 0xea9f7575);
    break;
  case 2:
    AS_UINT4(&sharedMemory[0x340]) = make_uint4(0x121b0909, 0x1d9e8383, 0x58742c2c, 0x342e1a1a);
    AS_UINT4(&sharedMemory[0x344]) = make_uint4(0x362d1b1b, 0xdcb26e6e, 0xb4ee5a5a, 0x5bfba0a0);
    AS_UINT4(&sharedMemory[0x348]) = make_uint4(0xa4f65252, 0x764d3b3b, 0xb761d6d6, 0x7dceb3b3);
    AS_UINT4(&sharedMemory[0x34C]) = make_uint4(0x527b2929, 0xdd3ee3e3, 0x5e712f2f, 0x13978484);
    AS_UINT4(&sharedMemory[0x350]) = make_uint4(0xa6f55353, 0xb968d1d1, 0x00000000, 0xc12ceded);
    AS_UINT4(&sharedMemory[0x354]) = make_uint4(0x40602020, 0xe31ffcfc, 0x79c8b1b1, 0xb6ed5b5b);
    AS_UINT4(&sharedMemory[0x358]) = make_uint4(0xd4be6a6a, 0x8d46cbcb, 0x67d9bebe, 0x724b3939);
    AS_UINT4(&sharedMemory[0x35C]) = make_uint4(0x94de4a4a, 0x98d44c4c, 0xb0e85858, 0x854acfcf);
    break;
  case 3:
    AS_UINT4(&sharedMemory[0x360]) = make_uint4(0xbb6bd0d0, 0xc52aefef, 0x4fe5aaaa, 0xed16fbfb);
    AS_UINT4(&sharedMemory[0x364]) = make_uint4(0x86c54343, 0x9ad74d4d, 0x66553333, 0x11948585);
    AS_UINT4(&sharedMemory[0x368]) = make_uint4(0x8acf4545, 0xe910f9f9, 0x04060202, 0xfe817f7f);
    AS_UINT4(&sharedMemory[0x36C]) = make_uint4(0xa0f05050, 0x78443c3c, 0x25ba9f9f, 0x4be3a8a8);
    AS_UINT4(&sharedMemory[0x370]) = make_uint4(0xa2f35151, 0x5dfea3a3, 0x80c04040, 0x058a8f8f);
    AS_UINT4(&sharedMemory[0x374]) = make_uint4(0x3fad9292, 0x21bc9d9d, 0x70483838, 0xf104f5f5);
    AS_UINT4(&sharedMemory[0x378]) = make_uint4(0x63dfbcbc, 0x77c1b6b6, 0xaf75dada, 0x42632121);
    AS_UINT4(&sharedMemory[0x37C]) = make_uint4(0x20301010, 0xe51affff, 0xfd0ef3f3, 0xbf6dd2d2);
    break;
  case 4:
    AS_UINT4(&sharedMemory[0x380]) = make_uint4(0x814ccdcd, 0x18140c0c, 0x26351313, 0xc32fecec);
    AS_UINT4(&sharedMemory[0x384]) = make_uint4(0xbee15f5f, 0x35a29797, 0x88cc4444, 0x2e391717);
    AS_UINT4(&sharedMemory[0x388]) = make_uint4(0x9357c4c4, 0x55f2a7a7, 0xfc827e7e, 0x7a473d3d);
    AS_UINT4(&sharedMemory[0x38C]) = make_uint4(0xc8ac6464, 0xbae75d5d, 0x322b1919, 0xe6957373);
    AS_UINT4(&sharedMemory[0x390]) = make_uint4(0xc0a06060, 0x19988181, 0x9ed14f4f, 0xa37fdcdc);
    AS_UINT4(&sharedMemory[0x394]) = make_uint4(0x44662222, 0x547e2a2a, 0x3bab9090, 0x0b838888);
    AS_UINT4(&sharedMemory[0x398]) = make_uint4(0x8cca4646, 0xc729eeee, 0x6bd3b8b8, 0x283c1414);
    AS_UINT4(&sharedMemory[0x39C]) = make_uint4(0xa779dede, 0xbce25e5e, 0x161d0b0b, 0xad76dbdb);
    break;
  case 5:
    AS_UINT4(&sharedMemory[0x3A0]) = make_uint4(0xdb3be0e0, 0x64563232, 0x744e3a3a, 0x141e0a0a);
    AS_UINT4(&sharedMemory[0x3A4]) = make_uint4(0x92db4949, 0x0c0a0606, 0x486c2424, 0xb8e45c5c);
    AS_UINT4(&sharedMemory[0x3A8]) = make_uint4(0x9f5dc2c2, 0xbd6ed3d3, 0x43efacac, 0xc4a66262);
    AS_UINT4(&sharedMemory[0x3AC]) = make_uint4(0x39a89191, 0x31a49595, 0xd337e4e4, 0xf28b7979);
    AS_UINT4(&sharedMemory[0x3B0]) = make_uint4(0xd532e7e7, 0x8b43c8c8, 0x6e593737, 0xdab76d6d);
    AS_UINT4(&sharedMemory[0x3B4]) = make_uint4(0x018c8d8d, 0xb164d5d5, 0x9cd24e4e, 0x49e0a9a9);
    AS_UINT4(&sharedMemory[0x3B8]) = make_uint4(0xd8b46c6c, 0xacfa5656, 0xf307f4f4, 0xcf25eaea);
    AS_UINT4(&sharedMemory[0x3BC]) = make_uint4(0xcaaf6565, 0xf48e7a7a, 0x47e9aeae, 0x10180808);
    break;
  case 6:
    AS_UINT4(&sharedMemory[0x3C0]) = make_uint4(0x6fd5baba, 0xf0887878, 0x4a6f2525, 0x5c722e2e);
    AS_UINT4(&sharedMemory[0x3C4]) = make_uint4(0x38241c1c, 0x57f1a6a6, 0x73c7b4b4, 0x9751c6c6);
    AS_UINT4(&sharedMemory[0x3C8]) = make_uint4(0xcb23e8e8, 0xa17cdddd, 0xe89c7474, 0x3e211f1f);
    AS_UINT4(&sharedMemory[0x3CC]) = make_uint4(0x96dd4b4b, 0x61dcbdbd, 0x0d868b8b, 0x0f858a8a);
    AS_UINT4(&sharedMemory[0x3D0]) = make_uint4(0xe0907070, 0x7c423e3e, 0x71c4b5b5, 0xccaa6666);
    AS_UINT4(&sharedMemory[0x3D4]) = make_uint4(0x90d84848, 0x06050303, 0xf701f6f6, 0x1c120e0e);
    AS_UINT4(&sharedMemory[0x3D8]) = make_uint4(0xc2a36161, 0x6a5f3535, 0xaef95757, 0x69d0b9b9);
    AS_UINT4(&sharedMemory[0x3DC]) = make_uint4(0x17918686, 0x9958c1c1, 0x3a271d1d, 0x27b99e9e);
    break;
  case 7:
    AS_UINT4(&sharedMemory[0x3E0]) = make_uint4(0xd938e1e1, 0xeb13f8f8, 0x2bb39898, 0x22331111);
    AS_UINT4(&sharedMemory[0x3E4]) = make_uint4(0xd2bb6969, 0xa970d9d9, 0x07898e8e, 0x33a79494);
    AS_UINT4(&sharedMemory[0x3E8]) = make_uint4(0x2db69b9b, 0x3c221e1e, 0x15928787, 0xc920e9e9);
    AS_UINT4(&sharedMemory[0x3EC]) = make_uint4(0x8749cece, 0xaaff5555, 0x50782828, 0xa57adfdf);
    AS_UINT4(&sharedMemory[0x3F0]) = make_uint4(0x038f8c8c, 0x59f8a1a1, 0x09808989, 0x1a170d0d);
    AS_UINT4(&sharedMemory[0x3F4]) = make_uint4(0x65dabfbf, 0xd731e6e6, 0x84c64242, 0xd0b86868);
    AS_UINT4(&sharedMemory[0x3F8]) = make_uint4(0x82c34141, 0x29b09999, 0x5a772d2d, 0x1e110f0f);
    AS_UINT4(&sharedMemory[0x3FC]) = make_uint4(0x7bcbb0b0, 0xa8fc5454, 0x6dd6bbbb, 0x2c3a1616);
    break;
  }
}