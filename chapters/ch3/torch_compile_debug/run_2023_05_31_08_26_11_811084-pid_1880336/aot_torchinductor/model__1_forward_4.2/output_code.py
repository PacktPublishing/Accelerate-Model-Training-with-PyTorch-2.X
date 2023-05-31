
from ctypes import c_void_p, c_long
import torch
import math
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()


kernel_cpp_0 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5,
                       float* __restrict__ out_ptr6,
                       long* __restrict__ out_ptr7)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    for(long i1=0; i1<8; i1+=1)
                    {
                        for(long i2=0; i2<16; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (256*i0) + (16384*i1));
                            tmp1_vec += tmp0;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp1)
                        for(long i2=256; i2<256; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (256*i0) + (16384*i1)];
                            tmp1 += tmp0;
                        }
                    }
                    tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                    out_ptr0[i0] = tmp1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<4; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(2048));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr1 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=64; i0<64; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(2048);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr1[i0] = tmp8;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp4 = 0;
                    auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                    for(long i1=0; i1<8; i1+=1)
                    {
                        for(long i2=0; i2<16; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (256*i0) + (16384*i1));
                            auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2.pow(2);
                            tmp4_vec += tmp3;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp4)
                        for(long i2=256; i2<256; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (256*i0) + (16384*i1)];
                            auto tmp1 = in_out_ptr0[i0];
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2 * tmp2;
                            tmp4 += tmp3;
                        }
                    }
                    tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                    out_ptr2[i0] = tmp4;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<4; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(2048));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0004885197850513));
                    auto tmp7 = tmp2 * tmp6;
                    auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp9 + tmp12;
                    tmp5.store(out_ptr3 + 16*i0);
                    tmp13.store(out_ptr4 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=64; i0<64; i0+=1)
                {
                    auto tmp0 = out_ptr2[i0];
                    auto tmp10 = in_ptr2[i0];
                    auto tmp1 = static_cast<float>(2048);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    auto tmp6 = static_cast<float>(1.0004885197850513);
                    auto tmp7 = tmp2 * tmp6;
                    auto tmp8 = static_cast<float>(0.1);
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp9 + tmp12;
                    out_ptr3[i0] = tmp5;
                    out_ptr4[i0] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<8; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<64; i1+=1)
                {
                    for(long i2=0; i2<16; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (256*i1) + (16384*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(2048));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                        tmp14.store(out_ptr5 + (16*i2) + (256*i1) + (16384*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=256; i2<256; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (256*i1) + (16384*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr2[i1];
                        auto tmp10 = in_ptr3[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(2048);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = tmp13 * (tmp13>0);
                        out_ptr5[i2 + (256*i1) + (16384*i0)] = tmp14;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long i0=0; i0<512; i0+=1)
                {
                    #pragma GCC ivdep
                    for(long i1=0; i1<8; i1+=1)
                    {
                        #pragma GCC ivdep
                        for(long i2=0; i2<8; i2+=1)
                        {
                            auto tmp0 = static_cast<long>((-1) + (2*i1));
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(16);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = static_cast<long>((-1) + (2*i2));
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = out_ptr5[(-17) + (2*i2) + (32*i1) + (256*i0)];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : -std::numeric_limits<decltype(tmp11())>::infinity();
                            auto tmp14 = static_cast<long>(2*i2);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = out_ptr5[(-16) + (2*i2) + (32*i1) + (256*i0)];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp18 ? tmp19() : -std::numeric_limits<decltype(tmp19())>::infinity();
                            auto tmp22 = (tmp13 != tmp13) ? tmp13 : std::max(tmp21, tmp13);
                            auto tmp23 = static_cast<long>(1 + (2*i2));
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = out_ptr5[(-15) + (2*i2) + (32*i1) + (256*i0)];
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp27 ? tmp28() : -std::numeric_limits<decltype(tmp28())>::infinity();
                            auto tmp31 = (tmp22 != tmp22) ? tmp22 : std::max(tmp30, tmp22);
                            auto tmp32 = static_cast<long>(2*i1);
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = out_ptr5[(-1) + (2*i2) + (32*i1) + (256*i0)];
                                return tmp38;
                            }
                            ;
                            auto tmp39 = tmp36 ? tmp37() : -std::numeric_limits<decltype(tmp37())>::infinity();
                            auto tmp40 = (tmp31 != tmp31) ? tmp31 : std::max(tmp39, tmp31);
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = out_ptr5[(2*i2) + (32*i1) + (256*i0)];
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp41 ? tmp42() : -std::numeric_limits<decltype(tmp42())>::infinity();
                            auto tmp45 = (tmp40 != tmp40) ? tmp40 : std::max(tmp44, tmp40);
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = out_ptr5[1 + (2*i2) + (32*i1) + (256*i0)];
                                return tmp48;
                            }
                            ;
                            auto tmp49 = tmp46 ? tmp47() : -std::numeric_limits<decltype(tmp47())>::infinity();
                            auto tmp50 = (tmp45 != tmp45) ? tmp45 : std::max(tmp49, tmp45);
                            auto tmp51 = static_cast<long>(1 + (2*i1));
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = out_ptr5[15 + (2*i2) + (32*i1) + (256*i0)];
                                return tmp57;
                            }
                            ;
                            auto tmp58 = tmp55 ? tmp56() : -std::numeric_limits<decltype(tmp56())>::infinity();
                            auto tmp59 = (tmp50 != tmp50) ? tmp50 : std::max(tmp58, tmp50);
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = out_ptr5[16 + (2*i2) + (32*i1) + (256*i0)];
                                return tmp62;
                            }
                            ;
                            auto tmp63 = tmp60 ? tmp61() : -std::numeric_limits<decltype(tmp61())>::infinity();
                            auto tmp64 = (tmp59 != tmp59) ? tmp59 : std::max(tmp63, tmp59);
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = out_ptr5[17 + (2*i2) + (32*i1) + (256*i0)];
                                return tmp67;
                            }
                            ;
                            auto tmp68 = tmp65 ? tmp66() : -std::numeric_limits<decltype(tmp66())>::infinity();
                            auto tmp69 = (tmp64 != tmp64) ? tmp64 : std::max(tmp68, tmp64);
                            auto tmp70 = [&]
                            {
                                auto tmp71 = out_ptr5[(-17) + (2*i2) + (32*i1) + (256*i0)];
                                return tmp71;
                            }
                            ;
                            auto tmp72 = tmp10 ? tmp70() : -std::numeric_limits<decltype(tmp70())>::infinity();
                            auto tmp73 = static_cast<long>((-17) + (2*i2) + (32*i1));
                            auto tmp74 = [&]
                            {
                                auto tmp75 = out_ptr5[(-16) + (2*i2) + (32*i1) + (256*i0)];
                                return tmp75;
                            }
                            ;
                            auto tmp76 = tmp18 ? tmp74() : -std::numeric_limits<decltype(tmp74())>::infinity();
                            auto tmp77 = static_cast<long>((-16) + (2*i2) + (32*i1));
                            auto tmp78 = tmp76 > tmp72;
                            auto tmp79 = tmp78 ? tmp77 : tmp73;
                            auto tmp80 = (tmp72 != tmp72) ? tmp72 : std::max(tmp76, tmp72);
                            auto tmp81 = [&]
                            {
                                auto tmp82 = out_ptr5[(-15) + (2*i2) + (32*i1) + (256*i0)];
                                return tmp82;
                            }
                            ;
                            auto tmp83 = tmp27 ? tmp81() : -std::numeric_limits<decltype(tmp81())>::infinity();
                            auto tmp84 = static_cast<long>((-15) + (2*i2) + (32*i1));
                            auto tmp85 = tmp83 > tmp80;
                            auto tmp86 = tmp85 ? tmp84 : tmp79;
                            auto tmp87 = (tmp80 != tmp80) ? tmp80 : std::max(tmp83, tmp80);
                            auto tmp88 = [&]
                            {
                                auto tmp89 = out_ptr5[(-1) + (2*i2) + (32*i1) + (256*i0)];
                                return tmp89;
                            }
                            ;
                            auto tmp90 = tmp36 ? tmp88() : -std::numeric_limits<decltype(tmp88())>::infinity();
                            auto tmp91 = static_cast<long>((-1) + (2*i2) + (32*i1));
                            auto tmp92 = tmp90 > tmp87;
                            auto tmp93 = tmp92 ? tmp91 : tmp86;
                            auto tmp94 = (tmp87 != tmp87) ? tmp87 : std::max(tmp90, tmp87);
                            auto tmp95 = [&]
                            {
                                auto tmp96 = out_ptr5[(2*i2) + (32*i1) + (256*i0)];
                                return tmp96;
                            }
                            ;
                            auto tmp97 = tmp41 ? tmp95() : -std::numeric_limits<decltype(tmp95())>::infinity();
                            auto tmp98 = static_cast<long>((2*i2) + (32*i1));
                            auto tmp99 = tmp97 > tmp94;
                            auto tmp100 = tmp99 ? tmp98 : tmp93;
                            auto tmp101 = (tmp94 != tmp94) ? tmp94 : std::max(tmp97, tmp94);
                            auto tmp102 = [&]
                            {
                                auto tmp103 = out_ptr5[1 + (2*i2) + (32*i1) + (256*i0)];
                                return tmp103;
                            }
                            ;
                            auto tmp104 = tmp46 ? tmp102() : -std::numeric_limits<decltype(tmp102())>::infinity();
                            auto tmp105 = static_cast<long>(1 + (2*i2) + (32*i1));
                            auto tmp106 = tmp104 > tmp101;
                            auto tmp107 = tmp106 ? tmp105 : tmp100;
                            auto tmp108 = (tmp101 != tmp101) ? tmp101 : std::max(tmp104, tmp101);
                            auto tmp109 = [&]
                            {
                                auto tmp110 = out_ptr5[15 + (2*i2) + (32*i1) + (256*i0)];
                                return tmp110;
                            }
                            ;
                            auto tmp111 = tmp55 ? tmp109() : -std::numeric_limits<decltype(tmp109())>::infinity();
                            auto tmp112 = static_cast<long>(15 + (2*i2) + (32*i1));
                            auto tmp113 = tmp111 > tmp108;
                            auto tmp114 = tmp113 ? tmp112 : tmp107;
                            auto tmp115 = (tmp108 != tmp108) ? tmp108 : std::max(tmp111, tmp108);
                            auto tmp116 = [&]
                            {
                                auto tmp117 = out_ptr5[16 + (2*i2) + (32*i1) + (256*i0)];
                                return tmp117;
                            }
                            ;
                            auto tmp118 = tmp60 ? tmp116() : -std::numeric_limits<decltype(tmp116())>::infinity();
                            auto tmp119 = static_cast<long>(16 + (2*i2) + (32*i1));
                            auto tmp120 = tmp118 > tmp115;
                            auto tmp121 = tmp120 ? tmp119 : tmp114;
                            auto tmp122 = (tmp115 != tmp115) ? tmp115 : std::max(tmp118, tmp115);
                            auto tmp123 = [&]
                            {
                                auto tmp124 = out_ptr5[17 + (2*i2) + (32*i1) + (256*i0)];
                                return tmp124;
                            }
                            ;
                            auto tmp125 = tmp65 ? tmp123() : -std::numeric_limits<decltype(tmp123())>::infinity();
                            auto tmp126 = static_cast<long>(17 + (2*i2) + (32*i1));
                            auto tmp127 = tmp125 > tmp122;
                            auto tmp128 = tmp127 ? tmp126 : tmp121;
                            auto tmp129 = (tmp122 != tmp122) ? tmp122 : std::max(tmp125, tmp122);
                            out_ptr6[i2 + (8*i1) + (64*i0)] = tmp69;
                            out_ptr7[i2 + (8*i1) + (64*i0)] = tmp128;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_1 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<64; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (4096*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i0) + (4096*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<4; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(512));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=64; i0<64; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(512);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<64; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (4096*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i0) + (4096*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<4; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(512));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0019569471624266));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=64; i0<64; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(512);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0019569471624266);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<64; i1+=1)
            {
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (16*i2) + (64*i1) + (4096*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=64; i2<64; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (64*i1) + (4096*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (64*i1) + (4096*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_2 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<64; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (4096*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i0) + (4096*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<4; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(512));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=64; i0<64; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(512);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<64; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (4096*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i0) + (4096*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<4; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(512));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0019569471624266));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=64; i0<64; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(512);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0019569471624266);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<64; i1+=1)
            {
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (16*i2) + (64*i1) + (4096*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=64; i2<64; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (64*i1) + (4096*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (64*i1) + (4096*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_3 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<256; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    for(long i1=0; i1<8; i1+=1)
                    {
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (16384*i1));
                            tmp1_vec += tmp0;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp1)
                        for(long i2=64; i2<64; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (64*i0) + (16384*i1)];
                            tmp1 += tmp0;
                        }
                    }
                    tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                    out_ptr0[i0] = tmp1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr1 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(512);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr1[i0] = tmp8;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<256; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp4 = 0;
                    auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                    for(long i1=0; i1<8; i1+=1)
                    {
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (16384*i1));
                            auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2.pow(2);
                            tmp4_vec += tmp3;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp4)
                        for(long i2=64; i2<64; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (64*i0) + (16384*i1)];
                            auto tmp1 = in_out_ptr0[i0];
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2 * tmp2;
                            tmp4 += tmp3;
                        }
                    }
                    tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                    out_ptr2[i0] = tmp4;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0019569471624266));
                    auto tmp7 = tmp2 * tmp6;
                    auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp9 + tmp12;
                    tmp5.store(out_ptr3 + 16*i0);
                    tmp13.store(out_ptr4 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr2[i0];
                    auto tmp10 = in_ptr2[i0];
                    auto tmp1 = static_cast<float>(512);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    auto tmp6 = static_cast<float>(1.0019569471624266);
                    auto tmp7 = tmp2 * tmp6;
                    auto tmp8 = static_cast<float>(0.1);
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp9 + tmp12;
                    out_ptr3[i0] = tmp5;
                    out_ptr4[i0] = tmp13;
                }
            }
        }
    }
}
''')


kernel_cpp_4 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       const float* __restrict__ in_ptr7,
                       const float* __restrict__ in_ptr8,
                       const float* __restrict__ in_ptr9,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr5 = in_out_ptr1;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<256; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    for(long i1=0; i1<8; i1+=1)
                    {
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (16384*i1));
                            tmp1_vec += tmp0;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp1)
                        for(long i2=64; i2<64; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (64*i0) + (16384*i1)];
                            tmp1 += tmp0;
                        }
                    }
                    tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                    out_ptr0[i0] = tmp1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr1 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(512);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr1[i0] = tmp8;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<256; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp4 = 0;
                    auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                    for(long i1=0; i1<8; i1+=1)
                    {
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (16384*i1));
                            auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2.pow(2);
                            tmp4_vec += tmp3;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp4)
                        for(long i2=64; i2<64; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (64*i0) + (16384*i1)];
                            auto tmp1 = in_out_ptr0[i0];
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2 * tmp2;
                            tmp4 += tmp3;
                        }
                    }
                    tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                    out_ptr2[i0] = tmp4;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0019569471624266));
                    auto tmp7 = tmp2 * tmp6;
                    auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp9 + tmp12;
                    tmp5.store(out_ptr3 + 16*i0);
                    tmp13.store(out_ptr4 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr2[i0];
                    auto tmp10 = in_ptr2[i0];
                    auto tmp1 = static_cast<float>(512);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    auto tmp6 = static_cast<float>(1.0019569471624266);
                    auto tmp7 = tmp2 * tmp6;
                    auto tmp8 = static_cast<float>(0.1);
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp9 + tmp12;
                    out_ptr3[i0] = tmp5;
                    out_ptr4[i0] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<8; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<256; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + (16*i2) + (64*i1) + (16384*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr6[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr7[i1]);
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i1) + (16384*i0));
                        auto tmp15 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr2[i1]);
                        auto tmp22 = at::vec::Vectorized<float>(in_ptr8[i1]);
                        auto tmp24 = at::vec::Vectorized<float>(in_ptr9[i1]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(512));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp16 = tmp14 - tmp15;
                        auto tmp18 = tmp17 / tmp4;
                        auto tmp19 = tmp18 + tmp6;
                        auto tmp20 = tmp19.rsqrt();
                        auto tmp21 = tmp16 * tmp20;
                        auto tmp23 = tmp21 * tmp22;
                        auto tmp25 = tmp23 + tmp24;
                        auto tmp26 = tmp13 + tmp25;
                        tmp26.store(out_ptr5 + (16*i2) + (64*i1) + (16384*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr3[i2 + (64*i1) + (16384*i0)];
                        auto tmp1 = in_ptr4[i1];
                        auto tmp3 = in_ptr5[i1];
                        auto tmp10 = in_ptr6[i1];
                        auto tmp12 = in_ptr7[i1];
                        auto tmp14 = in_ptr0[i2 + (64*i1) + (16384*i0)];
                        auto tmp15 = in_out_ptr0[i1];
                        auto tmp17 = out_ptr2[i1];
                        auto tmp22 = in_ptr8[i1];
                        auto tmp24 = in_ptr9[i1];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(512);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp16 = tmp14 - tmp15;
                        auto tmp18 = tmp17 / tmp4;
                        auto tmp19 = tmp18 + tmp6;
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = tmp16 * tmp20;
                        auto tmp23 = tmp21 * tmp22;
                        auto tmp25 = tmp23 + tmp24;
                        auto tmp26 = tmp13 + tmp25;
                        out_ptr5[i2 + (64*i1) + (16384*i0)] = tmp26;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<8192; i0+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + 16*i0);
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr1 + 16*i0);
            }
            #pragma omp for simd simdlen(8) 
            for(long i0=131072; i0<131072; i0+=1)
            {
                auto tmp0 = out_ptr5[i0];
                auto tmp1 = tmp0 * (tmp0>0);
                in_out_ptr1[i0] = tmp1;
            }
        }
    }
}
''')


kernel_cpp_5 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<64; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (4096*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i0) + (4096*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<4; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(512));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=64; i0<64; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(512);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<64; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (4096*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i0) + (4096*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<4; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(512));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0019569471624266));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=64; i0<64; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(512);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0019569471624266);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<64; i1+=1)
            {
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (16*i2) + (64*i1) + (4096*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=64; i2<64; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (64*i1) + (4096*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (64*i1) + (4096*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_6 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<64; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (4096*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i0) + (4096*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<4; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(512));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=64; i0<64; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(512);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<64; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (4096*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i0) + (4096*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<4; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(512));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0019569471624266));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=64; i0<64; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(512);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0019569471624266);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<64; i1+=1)
            {
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (16*i2) + (64*i1) + (4096*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=64; i2<64; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (64*i1) + (4096*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (64*i1) + (4096*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_7 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<256; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    for(long i1=0; i1<8; i1+=1)
                    {
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (16384*i1));
                            tmp1_vec += tmp0;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp1)
                        for(long i2=64; i2<64; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (64*i0) + (16384*i1)];
                            tmp1 += tmp0;
                        }
                    }
                    tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                    out_ptr0[i0] = tmp1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr1 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(512);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr1[i0] = tmp8;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<256; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp4 = 0;
                    auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                    for(long i1=0; i1<8; i1+=1)
                    {
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (16384*i1));
                            auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2.pow(2);
                            tmp4_vec += tmp3;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp4)
                        for(long i2=64; i2<64; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (64*i0) + (16384*i1)];
                            auto tmp1 = in_out_ptr0[i0];
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2 * tmp2;
                            tmp4 += tmp3;
                        }
                    }
                    tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                    out_ptr2[i0] = tmp4;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0019569471624266));
                    auto tmp7 = tmp2 * tmp6;
                    auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp9 + tmp12;
                    tmp5.store(out_ptr3 + 16*i0);
                    tmp13.store(out_ptr4 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr2[i0];
                    auto tmp10 = in_ptr2[i0];
                    auto tmp1 = static_cast<float>(512);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    auto tmp6 = static_cast<float>(1.0019569471624266);
                    auto tmp7 = tmp2 * tmp6;
                    auto tmp8 = static_cast<float>(0.1);
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp9 + tmp12;
                    out_ptr3[i0] = tmp5;
                    out_ptr4[i0] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<8; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<256; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i1) + (16384*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + (16*i2) + (64*i1) + (16384*i0));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(512));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                        tmp16.store(out_ptr5 + (16*i2) + (64*i1) + (16384*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i1) + (16384*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr2[i1];
                        auto tmp10 = in_ptr3[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp14 = in_ptr5[i2 + (64*i1) + (16384*i0)];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(512);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp15 * (tmp15>0);
                        out_ptr5[i2 + (64*i1) + (16384*i0)] = tmp16;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_8 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<64; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (4096*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i0) + (4096*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<4; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(512));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=64; i0<64; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(512);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<64; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (4096*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i0) + (4096*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<4; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(512));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0019569471624266));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=64; i0<64; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(512);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0019569471624266);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<64; i1+=1)
            {
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (16*i2) + (64*i1) + (4096*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=64; i2<64; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (64*i1) + (4096*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (64*i1) + (4096*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_9 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<64; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (4096*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i0) + (4096*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<4; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(512));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=64; i0<64; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(512);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<64; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (4096*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i0) + (4096*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<4; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(512));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0019569471624266));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=64; i0<64; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(512);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0019569471624266);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<64; i1+=1)
            {
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (16*i2) + (64*i1) + (4096*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=64; i2<64; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (64*i1) + (4096*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (64*i1) + (4096*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_10 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<256; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    for(long i1=0; i1<8; i1+=1)
                    {
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (16384*i1));
                            tmp1_vec += tmp0;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp1)
                        for(long i2=64; i2<64; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (64*i0) + (16384*i1)];
                            tmp1 += tmp0;
                        }
                    }
                    tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                    out_ptr0[i0] = tmp1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr1 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(512);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr1[i0] = tmp8;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<256; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp4 = 0;
                    auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                    for(long i1=0; i1<8; i1+=1)
                    {
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (16384*i1));
                            auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2.pow(2);
                            tmp4_vec += tmp3;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp4)
                        for(long i2=64; i2<64; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (64*i0) + (16384*i1)];
                            auto tmp1 = in_out_ptr0[i0];
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2 * tmp2;
                            tmp4 += tmp3;
                        }
                    }
                    tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                    out_ptr2[i0] = tmp4;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0019569471624266));
                    auto tmp7 = tmp2 * tmp6;
                    auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp9 + tmp12;
                    tmp5.store(out_ptr3 + 16*i0);
                    tmp13.store(out_ptr4 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr2[i0];
                    auto tmp10 = in_ptr2[i0];
                    auto tmp1 = static_cast<float>(512);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    auto tmp6 = static_cast<float>(1.0019569471624266);
                    auto tmp7 = tmp2 * tmp6;
                    auto tmp8 = static_cast<float>(0.1);
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp9 + tmp12;
                    out_ptr3[i0] = tmp5;
                    out_ptr4[i0] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<8; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<256; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i1) + (16384*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + (16*i2) + (64*i1) + (16384*i0));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(512));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                        tmp16.store(out_ptr5 + (16*i2) + (64*i1) + (16384*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i1) + (16384*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr2[i1];
                        auto tmp10 = in_ptr3[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp14 = in_ptr5[i2 + (64*i1) + (16384*i0)];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(512);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp15 * (tmp15>0);
                        out_ptr5[i2 + (64*i1) + (16384*i0)] = tmp16;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_11 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<128; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    for(long i1=0; i1<8; i1+=1)
                    {
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (8192*i1));
                            tmp1_vec += tmp0;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp1)
                        for(long i2=64; i2<64; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (64*i0) + (8192*i1)];
                            tmp1 += tmp0;
                        }
                    }
                    tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                    out_ptr0[i0] = tmp1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<8; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr1 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=128; i0<128; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(512);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr1[i0] = tmp8;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<128; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp4 = 0;
                    auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                    for(long i1=0; i1<8; i1+=1)
                    {
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (8192*i1));
                            auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2.pow(2);
                            tmp4_vec += tmp3;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp4)
                        for(long i2=64; i2<64; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (64*i0) + (8192*i1)];
                            auto tmp1 = in_out_ptr0[i0];
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2 * tmp2;
                            tmp4 += tmp3;
                        }
                    }
                    tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                    out_ptr2[i0] = tmp4;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<8; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0019569471624266));
                    auto tmp7 = tmp2 * tmp6;
                    auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp9 + tmp12;
                    tmp5.store(out_ptr3 + 16*i0);
                    tmp13.store(out_ptr4 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=128; i0<128; i0+=1)
                {
                    auto tmp0 = out_ptr2[i0];
                    auto tmp10 = in_ptr2[i0];
                    auto tmp1 = static_cast<float>(512);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    auto tmp6 = static_cast<float>(1.0019569471624266);
                    auto tmp7 = tmp2 * tmp6;
                    auto tmp8 = static_cast<float>(0.1);
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp9 + tmp12;
                    out_ptr3[i0] = tmp5;
                    out_ptr4[i0] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<8; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<128; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i1) + (8192*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(512));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                        tmp14.store(out_ptr5 + (16*i2) + (64*i1) + (8192*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i1) + (8192*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr2[i1];
                        auto tmp10 = in_ptr3[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(512);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = tmp13 * (tmp13>0);
                        out_ptr5[i2 + (64*i1) + (8192*i0)] = tmp14;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_12 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (2048*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i0) + (2048*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<8; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(128);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (2048*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i0) + (2048*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<8; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0078740157480315));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(128);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0078740157480315);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<128; i1+=1)
            {
                for(long i2=0; i2<1; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(128));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (16*i1) + (16*i2) + (2048*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=16; i2<16; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (16*i1) + (2048*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(128);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (16*i1) + (2048*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_13 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<512; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    for(long i1=0; i1<8; i1+=1)
                    {
                        for(long i2=0; i2<1; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (8192*i1));
                            tmp1_vec += tmp0;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp1)
                        for(long i2=16; i2<16; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (16*i0) + (8192*i1)];
                            tmp1 += tmp0;
                        }
                    }
                    tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                    out_ptr0[i0] = tmp1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<32; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr1 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=512; i0<512; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(128);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr1[i0] = tmp8;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<512; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp4 = 0;
                    auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                    for(long i1=0; i1<8; i1+=1)
                    {
                        for(long i2=0; i2<1; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (8192*i1));
                            auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2.pow(2);
                            tmp4_vec += tmp3;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp4)
                        for(long i2=16; i2<16; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (16*i0) + (8192*i1)];
                            auto tmp1 = in_out_ptr0[i0];
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2 * tmp2;
                            tmp4 += tmp3;
                        }
                    }
                    tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                    out_ptr2[i0] = tmp4;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<32; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0078740157480315));
                    auto tmp7 = tmp2 * tmp6;
                    auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp9 + tmp12;
                    tmp5.store(out_ptr3 + 16*i0);
                    tmp13.store(out_ptr4 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=512; i0<512; i0+=1)
                {
                    auto tmp0 = out_ptr2[i0];
                    auto tmp10 = in_ptr2[i0];
                    auto tmp1 = static_cast<float>(128);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    auto tmp6 = static_cast<float>(1.0078740157480315);
                    auto tmp7 = tmp2 * tmp6;
                    auto tmp8 = static_cast<float>(0.1);
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp9 + tmp12;
                    out_ptr3[i0] = tmp5;
                    out_ptr4[i0] = tmp13;
                }
            }
        }
    }
}
''')


kernel_cpp_14 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       const float* __restrict__ in_ptr7,
                       const float* __restrict__ in_ptr8,
                       const float* __restrict__ in_ptr9,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr5 = in_out_ptr1;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<512; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    for(long i1=0; i1<8; i1+=1)
                    {
                        for(long i2=0; i2<1; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (8192*i1));
                            tmp1_vec += tmp0;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp1)
                        for(long i2=16; i2<16; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (16*i0) + (8192*i1)];
                            tmp1 += tmp0;
                        }
                    }
                    tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                    out_ptr0[i0] = tmp1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<32; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr1 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=512; i0<512; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(128);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr1[i0] = tmp8;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<512; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp4 = 0;
                    auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                    for(long i1=0; i1<8; i1+=1)
                    {
                        for(long i2=0; i2<1; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (8192*i1));
                            auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2.pow(2);
                            tmp4_vec += tmp3;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp4)
                        for(long i2=16; i2<16; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (16*i0) + (8192*i1)];
                            auto tmp1 = in_out_ptr0[i0];
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2 * tmp2;
                            tmp4 += tmp3;
                        }
                    }
                    tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                    out_ptr2[i0] = tmp4;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<32; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0078740157480315));
                    auto tmp7 = tmp2 * tmp6;
                    auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp9 + tmp12;
                    tmp5.store(out_ptr3 + 16*i0);
                    tmp13.store(out_ptr4 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=512; i0<512; i0+=1)
                {
                    auto tmp0 = out_ptr2[i0];
                    auto tmp10 = in_ptr2[i0];
                    auto tmp1 = static_cast<float>(128);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    auto tmp6 = static_cast<float>(1.0078740157480315);
                    auto tmp7 = tmp2 * tmp6;
                    auto tmp8 = static_cast<float>(0.1);
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp9 + tmp12;
                    out_ptr3[i0] = tmp5;
                    out_ptr4[i0] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<8; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<512; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr6[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr7[i1]);
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp15 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr2[i1]);
                        auto tmp22 = at::vec::Vectorized<float>(in_ptr8[i1]);
                        auto tmp24 = at::vec::Vectorized<float>(in_ptr9[i1]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(128));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp16 = tmp14 - tmp15;
                        auto tmp18 = tmp17 / tmp4;
                        auto tmp19 = tmp18 + tmp6;
                        auto tmp20 = tmp19.rsqrt();
                        auto tmp21 = tmp16 * tmp20;
                        auto tmp23 = tmp21 * tmp22;
                        auto tmp25 = tmp23 + tmp24;
                        auto tmp26 = tmp13 + tmp25;
                        tmp26.store(out_ptr5 + (16*i1) + (16*i2) + (8192*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr3[i2 + (16*i1) + (8192*i0)];
                        auto tmp1 = in_ptr4[i1];
                        auto tmp3 = in_ptr5[i1];
                        auto tmp10 = in_ptr6[i1];
                        auto tmp12 = in_ptr7[i1];
                        auto tmp14 = in_ptr0[i2 + (16*i1) + (8192*i0)];
                        auto tmp15 = in_out_ptr0[i1];
                        auto tmp17 = out_ptr2[i1];
                        auto tmp22 = in_ptr8[i1];
                        auto tmp24 = in_ptr9[i1];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(128);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp16 = tmp14 - tmp15;
                        auto tmp18 = tmp17 / tmp4;
                        auto tmp19 = tmp18 + tmp6;
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = tmp16 * tmp20;
                        auto tmp23 = tmp21 * tmp22;
                        auto tmp25 = tmp23 + tmp24;
                        auto tmp26 = tmp13 + tmp25;
                        out_ptr5[i2 + (16*i1) + (8192*i0)] = tmp26;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<4096; i0+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + 16*i0);
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr1 + 16*i0);
            }
            #pragma omp for simd simdlen(8) 
            for(long i0=65536; i0<65536; i0+=1)
            {
                auto tmp0 = out_ptr5[i0];
                auto tmp1 = tmp0 * (tmp0>0);
                in_out_ptr1[i0] = tmp1;
            }
        }
    }
}
''')


kernel_cpp_15 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (2048*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i0) + (2048*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<8; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(128);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (2048*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i0) + (2048*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<8; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0078740157480315));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(128);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0078740157480315);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<128; i1+=1)
            {
                for(long i2=0; i2<1; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(128));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (16*i1) + (16*i2) + (2048*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=16; i2<16; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (16*i1) + (2048*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(128);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (16*i1) + (2048*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_16 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (2048*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i0) + (2048*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<8; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(128);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (2048*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i0) + (2048*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<8; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0078740157480315));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(128);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0078740157480315);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<128; i1+=1)
            {
                for(long i2=0; i2<1; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(128));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (16*i1) + (16*i2) + (2048*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=16; i2<16; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (16*i1) + (2048*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(128);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (16*i1) + (2048*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_17 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<512; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    for(long i1=0; i1<8; i1+=1)
                    {
                        for(long i2=0; i2<1; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (8192*i1));
                            tmp1_vec += tmp0;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp1)
                        for(long i2=16; i2<16; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (16*i0) + (8192*i1)];
                            tmp1 += tmp0;
                        }
                    }
                    tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                    out_ptr0[i0] = tmp1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<32; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr1 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=512; i0<512; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(128);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr1[i0] = tmp8;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<512; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp4 = 0;
                    auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                    for(long i1=0; i1<8; i1+=1)
                    {
                        for(long i2=0; i2<1; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (8192*i1));
                            auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2.pow(2);
                            tmp4_vec += tmp3;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp4)
                        for(long i2=16; i2<16; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (16*i0) + (8192*i1)];
                            auto tmp1 = in_out_ptr0[i0];
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2 * tmp2;
                            tmp4 += tmp3;
                        }
                    }
                    tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                    out_ptr2[i0] = tmp4;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<32; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0078740157480315));
                    auto tmp7 = tmp2 * tmp6;
                    auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp9 + tmp12;
                    tmp5.store(out_ptr3 + 16*i0);
                    tmp13.store(out_ptr4 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=512; i0<512; i0+=1)
                {
                    auto tmp0 = out_ptr2[i0];
                    auto tmp10 = in_ptr2[i0];
                    auto tmp1 = static_cast<float>(128);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    auto tmp6 = static_cast<float>(1.0078740157480315);
                    auto tmp7 = tmp2 * tmp6;
                    auto tmp8 = static_cast<float>(0.1);
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp9 + tmp12;
                    out_ptr3[i0] = tmp5;
                    out_ptr4[i0] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<8; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<512; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(128));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                        tmp16.store(out_ptr5 + (16*i1) + (16*i2) + (8192*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i1) + (8192*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr2[i1];
                        auto tmp10 = in_ptr3[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp14 = in_ptr5[i2 + (16*i1) + (8192*i0)];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(128);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp15 * (tmp15>0);
                        out_ptr5[i2 + (16*i1) + (8192*i0)] = tmp16;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_18 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (2048*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i0) + (2048*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<8; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(128);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (2048*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i0) + (2048*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<8; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0078740157480315));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(128);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0078740157480315);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<128; i1+=1)
            {
                for(long i2=0; i2<1; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(128));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (16*i1) + (16*i2) + (2048*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=16; i2<16; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (16*i1) + (2048*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(128);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (16*i1) + (2048*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_19 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (2048*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i0) + (2048*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<8; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(128);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (2048*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i0) + (2048*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<8; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0078740157480315));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(128);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0078740157480315);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<128; i1+=1)
            {
                for(long i2=0; i2<1; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(128));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (16*i1) + (16*i2) + (2048*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=16; i2<16; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (16*i1) + (2048*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(128);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (16*i1) + (2048*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_20 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<512; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    for(long i1=0; i1<8; i1+=1)
                    {
                        for(long i2=0; i2<1; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (8192*i1));
                            tmp1_vec += tmp0;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp1)
                        for(long i2=16; i2<16; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (16*i0) + (8192*i1)];
                            tmp1 += tmp0;
                        }
                    }
                    tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                    out_ptr0[i0] = tmp1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<32; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr1 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=512; i0<512; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(128);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr1[i0] = tmp8;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<512; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp4 = 0;
                    auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                    for(long i1=0; i1<8; i1+=1)
                    {
                        for(long i2=0; i2<1; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (8192*i1));
                            auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2.pow(2);
                            tmp4_vec += tmp3;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp4)
                        for(long i2=16; i2<16; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (16*i0) + (8192*i1)];
                            auto tmp1 = in_out_ptr0[i0];
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2 * tmp2;
                            tmp4 += tmp3;
                        }
                    }
                    tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                    out_ptr2[i0] = tmp4;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<32; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0078740157480315));
                    auto tmp7 = tmp2 * tmp6;
                    auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp9 + tmp12;
                    tmp5.store(out_ptr3 + 16*i0);
                    tmp13.store(out_ptr4 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=512; i0<512; i0+=1)
                {
                    auto tmp0 = out_ptr2[i0];
                    auto tmp10 = in_ptr2[i0];
                    auto tmp1 = static_cast<float>(128);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    auto tmp6 = static_cast<float>(1.0078740157480315);
                    auto tmp7 = tmp2 * tmp6;
                    auto tmp8 = static_cast<float>(0.1);
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp9 + tmp12;
                    out_ptr3[i0] = tmp5;
                    out_ptr4[i0] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<8; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<512; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(128));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                        tmp16.store(out_ptr5 + (16*i1) + (16*i2) + (8192*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i1) + (8192*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr2[i1];
                        auto tmp10 = in_ptr3[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp14 = in_ptr5[i2 + (16*i1) + (8192*i0)];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(128);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp15 * (tmp15>0);
                        out_ptr5[i2 + (16*i1) + (8192*i0)] = tmp16;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_21 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (2048*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i0) + (2048*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<8; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(128);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (2048*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i0) + (2048*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<8; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0078740157480315));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(128);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0078740157480315);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<128; i1+=1)
            {
                for(long i2=0; i2<1; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(128));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (16*i1) + (16*i2) + (2048*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=16; i2<16; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (16*i1) + (2048*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(128);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (16*i1) + (2048*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_22 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (2048*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i0) + (2048*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<8; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(128);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (2048*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i0) + (2048*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<8; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0078740157480315));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(128);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0078740157480315);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<128; i1+=1)
            {
                for(long i2=0; i2<1; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(128));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (16*i1) + (16*i2) + (2048*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=16; i2<16; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (16*i1) + (2048*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(128);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (16*i1) + (2048*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_23 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<512; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    for(long i1=0; i1<8; i1+=1)
                    {
                        for(long i2=0; i2<1; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (8192*i1));
                            tmp1_vec += tmp0;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp1)
                        for(long i2=16; i2<16; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (16*i0) + (8192*i1)];
                            tmp1 += tmp0;
                        }
                    }
                    tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                    out_ptr0[i0] = tmp1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<32; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr1 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=512; i0<512; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(128);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr1[i0] = tmp8;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<512; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp4 = 0;
                    auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                    for(long i1=0; i1<8; i1+=1)
                    {
                        for(long i2=0; i2<1; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (8192*i1));
                            auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2.pow(2);
                            tmp4_vec += tmp3;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp4)
                        for(long i2=16; i2<16; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (16*i0) + (8192*i1)];
                            auto tmp1 = in_out_ptr0[i0];
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2 * tmp2;
                            tmp4 += tmp3;
                        }
                    }
                    tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                    out_ptr2[i0] = tmp4;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<32; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0078740157480315));
                    auto tmp7 = tmp2 * tmp6;
                    auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp9 + tmp12;
                    tmp5.store(out_ptr3 + 16*i0);
                    tmp13.store(out_ptr4 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=512; i0<512; i0+=1)
                {
                    auto tmp0 = out_ptr2[i0];
                    auto tmp10 = in_ptr2[i0];
                    auto tmp1 = static_cast<float>(128);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    auto tmp6 = static_cast<float>(1.0078740157480315);
                    auto tmp7 = tmp2 * tmp6;
                    auto tmp8 = static_cast<float>(0.1);
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp9 + tmp12;
                    out_ptr3[i0] = tmp5;
                    out_ptr4[i0] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<8; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<512; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(128));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                        tmp16.store(out_ptr5 + (16*i1) + (16*i2) + (8192*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i1) + (8192*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr2[i1];
                        auto tmp10 = in_ptr3[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp14 = in_ptr5[i2 + (16*i1) + (8192*i0)];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(128);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp15 * (tmp15>0);
                        out_ptr5[i2 + (16*i1) + (8192*i0)] = tmp16;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_24 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (4096*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i0) + (4096*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(128);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (4096*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i0) + (4096*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(128));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.0078740157480315));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(128);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.0078740157480315);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; i1+=1)
            {
                for(long i2=0; i2<1; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (4096*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(128));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (16*i1) + (16*i2) + (4096*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=16; i2<16; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (16*i1) + (4096*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(128);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (16*i1) + (4096*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_25 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.032258064516129));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.032258064516129);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(32));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (4*i1) + (16*i2) + (1024*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (4*i1) + (1024*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(32);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (4*i1) + (1024*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_26 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<1024; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (4096*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (4096*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<64; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=1024; i0<1024; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<1024; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (4096*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (4096*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<64; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.032258064516129));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=1024; i0<1024; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.032258064516129);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
}
''')


kernel_cpp_27 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       const float* __restrict__ in_ptr7,
                       const float* __restrict__ in_ptr8,
                       const float* __restrict__ in_ptr9,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr5 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<1024; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (4096*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (4096*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<64; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=1024; i0<1024; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<1024; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (4096*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (4096*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<64; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.032258064516129));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=1024; i0<1024; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.032258064516129);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<1024; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + (4*i1) + (16*i2) + (4096*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(in_ptr5[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr6[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr7[i1]);
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (4096*i0));
                    auto tmp15 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp17 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp22 = at::vec::Vectorized<float>(in_ptr8[i1]);
                    auto tmp24 = at::vec::Vectorized<float>(in_ptr9[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(32));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp16 = tmp14 - tmp15;
                    auto tmp18 = tmp17 / tmp4;
                    auto tmp19 = tmp18 + tmp6;
                    auto tmp20 = tmp19.rsqrt();
                    auto tmp21 = tmp16 * tmp20;
                    auto tmp23 = tmp21 * tmp22;
                    auto tmp25 = tmp23 + tmp24;
                    auto tmp26 = tmp13 + tmp25;
                    tmp26.store(out_ptr5 + (4*i1) + (16*i2) + (4096*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_ptr3[i2 + (4*i1) + (4096*i0)];
                    auto tmp1 = in_ptr4[i1];
                    auto tmp3 = in_ptr5[i1];
                    auto tmp10 = in_ptr6[i1];
                    auto tmp12 = in_ptr7[i1];
                    auto tmp14 = in_ptr0[i2 + (4*i1) + (4096*i0)];
                    auto tmp15 = in_out_ptr0[i1];
                    auto tmp17 = out_ptr2[i1];
                    auto tmp22 = in_ptr8[i1];
                    auto tmp24 = in_ptr9[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(32);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp16 = tmp14 - tmp15;
                    auto tmp18 = tmp17 / tmp4;
                    auto tmp19 = tmp18 + tmp6;
                    auto tmp20 = 1 / std::sqrt(tmp19);
                    auto tmp21 = tmp16 * tmp20;
                    auto tmp23 = tmp21 * tmp22;
                    auto tmp25 = tmp23 + tmp24;
                    auto tmp26 = tmp13 + tmp25;
                    out_ptr5[i2 + (4*i1) + (4096*i0)] = tmp26;
                }
            }
        }
    }
    {
        for(long i0=0; i0<2048; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + 16*i0);
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=32768; i0<32768; i0+=1)
        {
            auto tmp0 = out_ptr5[i0];
            auto tmp1 = tmp0 * (tmp0>0);
            in_out_ptr1[i0] = tmp1;
        }
    }
}
''')


kernel_cpp_28 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.032258064516129));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.032258064516129);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(32));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (4*i1) + (16*i2) + (1024*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (4*i1) + (1024*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(32);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (4*i1) + (1024*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_29 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.032258064516129));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.032258064516129);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(32));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (4*i1) + (16*i2) + (1024*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (4*i1) + (1024*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(32);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (4*i1) + (1024*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_30 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<1024; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (4096*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (4096*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<64; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=1024; i0<1024; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<1024; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (4096*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (4096*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<64; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.032258064516129));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=1024; i0<1024; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.032258064516129);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<1024; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (4096*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + (4*i1) + (16*i2) + (4096*i0));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(32));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr5 + (4*i1) + (16*i2) + (4096*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (4*i1) + (4096*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp14 = in_ptr5[i2 + (4*i1) + (4096*i0)];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(32);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = tmp15 * (tmp15>0);
                    out_ptr5[i2 + (4*i1) + (4096*i0)] = tmp16;
                }
            }
        }
    }
}
''')


kernel_cpp_31 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.032258064516129));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.032258064516129);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(32));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (4*i1) + (16*i2) + (1024*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (4*i1) + (1024*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(32);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (4*i1) + (1024*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_32 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.032258064516129));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.032258064516129);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(32));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (4*i1) + (16*i2) + (1024*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (4*i1) + (1024*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(32);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (4*i1) + (1024*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_33 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<1024; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (4096*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (4096*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<64; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=1024; i0<1024; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<1024; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (4096*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (4096*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<64; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.032258064516129));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=1024; i0<1024; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.032258064516129);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<1024; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (4096*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + (4*i1) + (16*i2) + (4096*i0));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(32));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr5 + (4*i1) + (16*i2) + (4096*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (4*i1) + (4096*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp14 = in_ptr5[i2 + (4*i1) + (4096*i0)];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(32);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = tmp15 * (tmp15>0);
                    out_ptr5[i2 + (4*i1) + (4096*i0)] = tmp16;
                }
            }
        }
    }
}
''')


kernel_cpp_34 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.032258064516129));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.032258064516129);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(32));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (4*i1) + (16*i2) + (1024*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (4*i1) + (1024*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(32);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (4*i1) + (1024*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_35 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.032258064516129));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.032258064516129);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(32));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (4*i1) + (16*i2) + (1024*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (4*i1) + (1024*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(32);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (4*i1) + (1024*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_36 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<1024; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (4096*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (4096*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<64; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=1024; i0<1024; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<1024; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (4096*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (4096*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<64; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.032258064516129));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=1024; i0<1024; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.032258064516129);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<1024; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (4096*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + (4*i1) + (16*i2) + (4096*i0));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(32));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr5 + (4*i1) + (16*i2) + (4096*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (4*i1) + (4096*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp14 = in_ptr5[i2 + (4*i1) + (4096*i0)];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(32);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = tmp15 * (tmp15>0);
                    out_ptr5[i2 + (4*i1) + (4096*i0)] = tmp16;
                }
            }
        }
    }
}
''')


kernel_cpp_37 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.032258064516129));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.032258064516129);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(32));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (4*i1) + (16*i2) + (1024*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (4*i1) + (1024*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(32);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (4*i1) + (1024*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_38 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.032258064516129));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.032258064516129);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(32));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (4*i1) + (16*i2) + (1024*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (4*i1) + (1024*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(32);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (4*i1) + (1024*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_39 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<1024; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (4096*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (4096*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<64; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=1024; i0<1024; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<1024; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (4096*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (4096*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<64; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.032258064516129));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=1024; i0<1024; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.032258064516129);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<1024; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (4096*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + (4*i1) + (16*i2) + (4096*i0));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(32));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr5 + (4*i1) + (16*i2) + (4096*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (4*i1) + (4096*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp14 = in_ptr5[i2 + (4*i1) + (4096*i0)];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(32);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = tmp15 * (tmp15>0);
                    out_ptr5[i2 + (4*i1) + (4096*i0)] = tmp16;
                }
            }
        }
    }
}
''')


kernel_cpp_40 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.032258064516129));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.032258064516129);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(32));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (4*i1) + (16*i2) + (1024*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (4*i1) + (1024*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(32);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (4*i1) + (1024*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_41 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.032258064516129));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.032258064516129);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(32));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (4*i1) + (16*i2) + (1024*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (4*i1) + (1024*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(32);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (4*i1) + (1024*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_42 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<1024; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (4096*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (4096*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<64; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=1024; i0<1024; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<1024; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (4096*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (4096*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<64; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.032258064516129));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=1024; i0<1024; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.032258064516129);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<1024; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (4096*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + (4*i1) + (16*i2) + (4096*i0));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(32));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr5 + (4*i1) + (16*i2) + (4096*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (4*i1) + (4096*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp14 = in_ptr5[i2 + (4*i1) + (4096*i0)];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(32);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = tmp15 * (tmp15>0);
                    out_ptr5[i2 + (4*i1) + (4096*i0)] = tmp16;
                }
            }
        }
    }
}
''')


kernel_cpp_43 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (2048*i1));
                        tmp1_vec += tmp0;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp1)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (2048*i1)];
                        tmp1 += tmp0;
                    }
                }
                tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp4 = 0;
                auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                for(long i1=0; i1<8; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (2048*i1));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i0]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.pow(2);
                        tmp4_vec += tmp3;
                    }
                    #pragma omp simd simdlen(8)  reduction(+:tmp4)
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i0) + (2048*i1)];
                        auto tmp1 = in_out_ptr0[i0];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp4 += tmp3;
                    }
                }
                tmp4 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp4_vec);
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(32));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1.032258064516129));
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            tmp5.store(out_ptr3 + 16*i0);
            tmp13.store(out_ptr4 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp10 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(32);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = tmp2 + tmp3;
            auto tmp5 = 1 / std::sqrt(tmp4);
            auto tmp6 = static_cast<float>(1.032258064516129);
            auto tmp7 = tmp2 * tmp6;
            auto tmp8 = static_cast<float>(0.1);
            auto tmp9 = tmp7 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp9 + tmp12;
            out_ptr3[i0] = tmp5;
            out_ptr4[i0] = tmp13;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<512; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (2048*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr2[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(32));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(out_ptr5 + (4*i1) + (16*i2) + (2048*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_ptr0[i2 + (4*i1) + (2048*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr2[i1];
                    auto tmp10 = in_ptr3[i1];
                    auto tmp12 = in_ptr4[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(32);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr5[i2 + (4*i1) + (2048*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_44 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    tmp1 += tmp0;
                }
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp4 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    auto tmp1 = in_out_ptr0[i0];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    tmp4 += tmp3;
                }
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.1428571428571428));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr3 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp7 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.1428571428571428);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr3[i0] = tmp10;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i1);
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i1);
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i1);
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(8));
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp7.rsqrt();
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                tmp14.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = in_ptr0[i1 + (512*i0)];
                auto tmp1 = in_out_ptr0[i1];
                auto tmp3 = out_ptr2[i1];
                auto tmp10 = in_ptr3[i1];
                auto tmp12 = in_ptr4[i1];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(8);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr4[i1 + (512*i0)] = tmp14;
            }
        }
    }
}
''')


kernel_cpp_45 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<2048; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (2048*i1)];
                    tmp1 += tmp0;
                }
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<128; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<2048; i0+=1)
        {
            {
                float tmp4 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (2048*i1)];
                    auto tmp1 = in_out_ptr0[i0];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    tmp4 += tmp3;
                }
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<128; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.1428571428571428));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr3 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp7 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.1428571428571428);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr3[i0] = tmp10;
        }
    }
}
''')


kernel_cpp_46 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       const float* __restrict__ in_ptr7,
                       const float* __restrict__ in_ptr8,
                       const float* __restrict__ in_ptr9,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<2048; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (2048*i1)];
                    tmp1 += tmp0;
                }
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<128; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<2048; i0+=1)
        {
            {
                float tmp4 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (2048*i1)];
                    auto tmp1 = in_out_ptr0[i0];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    tmp4 += tmp3;
                }
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<128; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.1428571428571428));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr3 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp7 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.1428571428571428);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr3[i0] = tmp10;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            for(long i1=0; i1<128; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + (16*i1) + (2048*i0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i1);
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + 16*i1);
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr7 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (2048*i0));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i1);
                auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i1);
                auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr8 + 16*i1);
                auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr9 + 16*i1);
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(8));
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp7.rsqrt();
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp16 = tmp14 - tmp15;
                auto tmp18 = tmp17 / tmp4;
                auto tmp19 = tmp18 + tmp6;
                auto tmp20 = tmp19.rsqrt();
                auto tmp21 = tmp16 * tmp20;
                auto tmp23 = tmp21 * tmp22;
                auto tmp25 = tmp23 + tmp24;
                auto tmp26 = tmp13 + tmp25;
                tmp26.store(out_ptr4 + (16*i1) + (2048*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=2048; i1<2048; i1+=1)
            {
                auto tmp0 = in_ptr3[i1 + (2048*i0)];
                auto tmp1 = in_ptr4[i1];
                auto tmp3 = in_ptr5[i1];
                auto tmp10 = in_ptr6[i1];
                auto tmp12 = in_ptr7[i1];
                auto tmp14 = in_ptr0[i1 + (2048*i0)];
                auto tmp15 = in_out_ptr0[i1];
                auto tmp17 = out_ptr2[i1];
                auto tmp22 = in_ptr8[i1];
                auto tmp24 = in_ptr9[i1];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(8);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp16 = tmp14 - tmp15;
                auto tmp18 = tmp17 / tmp4;
                auto tmp19 = tmp18 + tmp6;
                auto tmp20 = 1 / std::sqrt(tmp19);
                auto tmp21 = tmp16 * tmp20;
                auto tmp23 = tmp21 * tmp22;
                auto tmp25 = tmp23 + tmp24;
                auto tmp26 = tmp13 + tmp25;
                out_ptr4[i1 + (2048*i0)] = tmp26;
            }
        }
    }
    {
        for(long i0=0; i0<1024; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + 16*i0);
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=16384; i0<16384; i0+=1)
        {
            auto tmp0 = out_ptr4[i0];
            auto tmp1 = tmp0 * (tmp0>0);
            in_out_ptr1[i0] = tmp1;
        }
    }
}
''')


kernel_cpp_47 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    tmp1 += tmp0;
                }
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp4 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    auto tmp1 = in_out_ptr0[i0];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    tmp4 += tmp3;
                }
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.1428571428571428));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr3 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp7 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.1428571428571428);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr3[i0] = tmp10;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i1);
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i1);
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i1);
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(8));
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp7.rsqrt();
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                tmp14.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = in_ptr0[i1 + (512*i0)];
                auto tmp1 = in_out_ptr0[i1];
                auto tmp3 = out_ptr2[i1];
                auto tmp10 = in_ptr3[i1];
                auto tmp12 = in_ptr4[i1];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(8);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr4[i1 + (512*i0)] = tmp14;
            }
        }
    }
}
''')


kernel_cpp_48 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    tmp1 += tmp0;
                }
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp4 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    auto tmp1 = in_out_ptr0[i0];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    tmp4 += tmp3;
                }
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.1428571428571428));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr3 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp7 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.1428571428571428);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr3[i0] = tmp10;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i1);
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i1);
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i1);
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(8));
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp7.rsqrt();
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                tmp14.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = in_ptr0[i1 + (512*i0)];
                auto tmp1 = in_out_ptr0[i1];
                auto tmp3 = out_ptr2[i1];
                auto tmp10 = in_ptr3[i1];
                auto tmp12 = in_ptr4[i1];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(8);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr4[i1 + (512*i0)] = tmp14;
            }
        }
    }
}
''')


kernel_cpp_49 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<2048; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (2048*i1)];
                    tmp1 += tmp0;
                }
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<128; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<2048; i0+=1)
        {
            {
                float tmp4 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (2048*i1)];
                    auto tmp1 = in_out_ptr0[i0];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    tmp4 += tmp3;
                }
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<128; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.1428571428571428));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr3 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp7 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.1428571428571428);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr3[i0] = tmp10;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            for(long i1=0; i1<128; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (2048*i0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i1);
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i1);
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + (16*i1) + (2048*i0));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(8));
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp7.rsqrt();
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                tmp16.store(out_ptr4 + (16*i1) + (2048*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=2048; i1<2048; i1+=1)
            {
                auto tmp0 = in_ptr0[i1 + (2048*i0)];
                auto tmp1 = in_out_ptr0[i1];
                auto tmp3 = out_ptr2[i1];
                auto tmp10 = in_ptr3[i1];
                auto tmp12 = in_ptr4[i1];
                auto tmp14 = in_ptr5[i1 + (2048*i0)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(8);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = tmp15 * (tmp15>0);
                out_ptr4[i1 + (2048*i0)] = tmp16;
            }
        }
    }
}
''')


kernel_cpp_50 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    tmp1 += tmp0;
                }
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp4 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    auto tmp1 = in_out_ptr0[i0];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    tmp4 += tmp3;
                }
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.1428571428571428));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr3 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp7 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.1428571428571428);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr3[i0] = tmp10;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i1);
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i1);
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i1);
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(8));
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp7.rsqrt();
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                tmp14.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = in_ptr0[i1 + (512*i0)];
                auto tmp1 = in_out_ptr0[i1];
                auto tmp3 = out_ptr2[i1];
                auto tmp10 = in_ptr3[i1];
                auto tmp12 = in_ptr4[i1];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(8);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr4[i1 + (512*i0)] = tmp14;
            }
        }
    }
}
''')


kernel_cpp_51 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    tmp1 += tmp0;
                }
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp4 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    auto tmp1 = in_out_ptr0[i0];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    tmp4 += tmp3;
                }
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.1428571428571428));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr3 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp7 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.1428571428571428);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr3[i0] = tmp10;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i1);
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i1);
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i1);
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(8));
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp7.rsqrt();
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                tmp14.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = in_ptr0[i1 + (512*i0)];
                auto tmp1 = in_out_ptr0[i1];
                auto tmp3 = out_ptr2[i1];
                auto tmp10 = in_ptr3[i1];
                auto tmp12 = in_ptr4[i1];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(8);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr4[i1 + (512*i0)] = tmp14;
            }
        }
    }
}
''')


kernel_cpp_52 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5,
                       bool* __restrict__ out_ptr6)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<2048; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (2048*i1)];
                    tmp1 += tmp0;
                }
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<128; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr1[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<2048; i0+=1)
        {
            {
                float tmp4 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (2048*i1)];
                    auto tmp1 = in_out_ptr0[i0];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    tmp4 += tmp3;
                }
                out_ptr2[i0] = tmp4;
            }
        }
    }
    {
        for(long i0=0; i0<128; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.1428571428571428));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr3 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp7 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.1428571428571428);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr3[i0] = tmp10;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            for(long i1=0; i1<128; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (2048*i0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i1);
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i1);
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + (16*i1) + (2048*i0));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(8));
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp7.rsqrt();
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                tmp16.store(out_ptr4 + (16*i1) + (2048*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=2048; i1<2048; i1+=1)
            {
                auto tmp0 = in_ptr0[i1 + (2048*i0)];
                auto tmp1 = in_out_ptr0[i1];
                auto tmp3 = out_ptr2[i1];
                auto tmp10 = in_ptr3[i1];
                auto tmp12 = in_ptr4[i1];
                auto tmp14 = in_ptr5[i1 + (2048*i0)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(8);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = tmp15 * (tmp15>0);
                out_ptr4[i1 + (2048*i0)] = tmp16;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16384; i0+=1)
        {
            auto tmp0 = out_ptr4[i0];
            auto tmp1 = static_cast<float>(1);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0);
            auto tmp4 = tmp0 <= tmp3;
            out_ptr5[i0] = tmp2;
            out_ptr6[i0] = tmp4;
        }
    }
}
''')


kernel_cpp_53 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(const long* __restrict__ in_ptr0,
                       const long* __restrict__ in_ptr1,
                       const long* __restrict__ in_ptr2,
                       const long* __restrict__ in_ptr3,
                       const long* __restrict__ in_ptr4,
                       const long* __restrict__ in_ptr5,
                       const long* __restrict__ in_ptr6,
                       const long* __restrict__ in_ptr7,
                       const long* __restrict__ in_ptr8,
                       const long* __restrict__ in_ptr9,
                       const long* __restrict__ in_ptr10,
                       const long* __restrict__ in_ptr11,
                       const long* __restrict__ in_ptr12,
                       const long* __restrict__ in_ptr13,
                       const long* __restrict__ in_ptr14,
                       const long* __restrict__ in_ptr15,
                       const long* __restrict__ in_ptr16,
                       const long* __restrict__ in_ptr17,
                       const long* __restrict__ in_ptr18,
                       const long* __restrict__ in_ptr19,
                       const long* __restrict__ in_ptr20,
                       const long* __restrict__ in_ptr21,
                       const long* __restrict__ in_ptr22,
                       const long* __restrict__ in_ptr23,
                       const long* __restrict__ in_ptr24,
                       const long* __restrict__ in_ptr25,
                       const long* __restrict__ in_ptr26,
                       const long* __restrict__ in_ptr27,
                       const long* __restrict__ in_ptr28,
                       const long* __restrict__ in_ptr29,
                       const long* __restrict__ in_ptr30,
                       const long* __restrict__ in_ptr31,
                       const long* __restrict__ in_ptr32,
                       const long* __restrict__ in_ptr33,
                       const long* __restrict__ in_ptr34,
                       const long* __restrict__ in_ptr35,
                       const long* __restrict__ in_ptr36,
                       const long* __restrict__ in_ptr37,
                       const long* __restrict__ in_ptr38,
                       const long* __restrict__ in_ptr39,
                       const long* __restrict__ in_ptr40,
                       const long* __restrict__ in_ptr41,
                       const long* __restrict__ in_ptr42,
                       const long* __restrict__ in_ptr43,
                       const long* __restrict__ in_ptr44,
                       const long* __restrict__ in_ptr45,
                       const long* __restrict__ in_ptr46,
                       const long* __restrict__ in_ptr47,
                       const long* __restrict__ in_ptr48,
                       const long* __restrict__ in_ptr49,
                       const long* __restrict__ in_ptr50,
                       const long* __restrict__ in_ptr51,
                       const long* __restrict__ in_ptr52,
                       long* __restrict__ out_ptr0,
                       long* __restrict__ out_ptr1,
                       long* __restrict__ out_ptr2,
                       long* __restrict__ out_ptr3,
                       long* __restrict__ out_ptr4,
                       long* __restrict__ out_ptr5,
                       long* __restrict__ out_ptr6,
                       long* __restrict__ out_ptr7,
                       long* __restrict__ out_ptr8,
                       long* __restrict__ out_ptr9,
                       long* __restrict__ out_ptr10,
                       long* __restrict__ out_ptr11,
                       long* __restrict__ out_ptr12,
                       long* __restrict__ out_ptr13,
                       long* __restrict__ out_ptr14,
                       long* __restrict__ out_ptr15,
                       long* __restrict__ out_ptr16,
                       long* __restrict__ out_ptr17,
                       long* __restrict__ out_ptr18,
                       long* __restrict__ out_ptr19,
                       long* __restrict__ out_ptr20,
                       long* __restrict__ out_ptr21,
                       long* __restrict__ out_ptr22,
                       long* __restrict__ out_ptr23,
                       long* __restrict__ out_ptr24,
                       long* __restrict__ out_ptr25,
                       long* __restrict__ out_ptr26,
                       long* __restrict__ out_ptr27,
                       long* __restrict__ out_ptr28,
                       long* __restrict__ out_ptr29,
                       long* __restrict__ out_ptr30,
                       long* __restrict__ out_ptr31,
                       long* __restrict__ out_ptr32,
                       long* __restrict__ out_ptr33,
                       long* __restrict__ out_ptr34,
                       long* __restrict__ out_ptr35,
                       long* __restrict__ out_ptr36,
                       long* __restrict__ out_ptr37,
                       long* __restrict__ out_ptr38,
                       long* __restrict__ out_ptr39,
                       long* __restrict__ out_ptr40,
                       long* __restrict__ out_ptr41,
                       long* __restrict__ out_ptr42,
                       long* __restrict__ out_ptr43,
                       long* __restrict__ out_ptr44,
                       long* __restrict__ out_ptr45,
                       long* __restrict__ out_ptr46,
                       long* __restrict__ out_ptr47,
                       long* __restrict__ out_ptr48,
                       long* __restrict__ out_ptr49,
                       long* __restrict__ out_ptr50,
                       long* __restrict__ out_ptr51,
                       long* __restrict__ out_ptr52)
{
    {
        auto tmp0 = in_ptr0[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr0[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr1[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr1[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr2[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr2[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr3[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr3[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr4[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr4[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr5[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr5[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr6[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr6[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr7[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr7[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr8[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr8[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr9[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr9[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr10[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr10[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr11[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr11[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr12[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr12[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr13[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr13[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr14[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr14[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr15[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr15[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr16[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr16[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr17[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr17[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr18[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr18[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr19[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr19[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr20[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr20[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr21[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr21[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr22[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr22[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr23[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr23[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr24[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr24[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr25[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr25[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr26[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr26[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr27[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr27[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr28[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr28[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr29[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr29[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr30[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr30[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr31[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr31[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr32[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr32[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr33[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr33[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr34[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr34[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr35[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr35[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr36[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr36[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr37[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr37[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr38[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr38[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr39[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr39[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr40[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr40[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr41[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr41[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr42[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr42[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr43[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr43[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr44[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr44[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr45[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr45[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr46[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr46[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr47[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr47[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr48[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr48[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr49[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr49[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr50[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr50[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr51[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr51[0] = tmp2;
    }
    {
        auto tmp0 = in_ptr52[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr52[0] = tmp2;
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321 = args
    args.clear()
    buf0 = aten.convolution(primals_321, primals_1, None, (2, 2), (3, 3), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf0, (8, 64, 16, 16), (16384, 256, 16, 1))
    buf1 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf2 = buf1; del buf1  # reuse
    buf5 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((8, 64, 16, 16), (16384, 256, 16, 1), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((8, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    buf9 = empty_strided((8, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.int64)
    kernel_cpp_0(c_void_p(buf2.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(primals_162.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()))
    del primals_162
    del primals_163
    del primals_3
    buf10 = aten.convolution(buf8, primals_4, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf10, (8, 64, 8, 8), (4096, 64, 8, 1))
    buf11 = buf3; del buf3  # reuse
    buf12 = buf11; del buf11  # reuse
    buf15 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf14 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf16 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf17 = empty_strided((8, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_1(c_void_p(buf12.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()))
    del primals_165
    del primals_166
    del primals_6
    buf18 = aten.convolution(buf17, primals_7, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf18, (8, 64, 8, 8), (4096, 64, 8, 1))
    buf19 = buf13; del buf13  # reuse
    buf20 = buf19; del buf19  # reuse
    buf23 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf21 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf22 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf24 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf25 = empty_strided((8, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_2(c_void_p(buf20.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(primals_168.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()))
    del primals_168
    del primals_169
    del primals_9
    buf26 = aten.convolution(buf25, primals_10, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf26, (8, 256, 8, 8), (16384, 64, 8, 1))
    buf27 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf28 = buf27; del buf27  # reuse
    buf31 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf29 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf30 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf32 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    kernel_cpp_3(c_void_p(buf28.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf32.data_ptr()))
    del primals_171
    del primals_172
    buf33 = aten.convolution(buf8, primals_13, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf33, (8, 256, 8, 8), (16384, 64, 8, 1))
    buf34 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf35 = buf34; del buf34  # reuse
    buf38 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf36 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf37 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf39 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf40 = empty_strided((8, 256, 8, 8), (16384, 64, 8, 1), device='cpu', dtype=torch.float32)
    buf41 = buf40; del buf40  # reuse
    kernel_cpp_4(c_void_p(buf35.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf39.data_ptr()))
    del primals_12
    del primals_15
    del primals_174
    del primals_175
    buf42 = aten.convolution(buf41, primals_16, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf42, (8, 64, 8, 8), (4096, 64, 8, 1))
    buf43 = buf21; del buf21  # reuse
    buf44 = buf43; del buf43  # reuse
    buf47 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf45 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf46 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf48 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf49 = empty_strided((8, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_5(c_void_p(buf44.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()))
    del primals_177
    del primals_178
    del primals_18
    buf50 = aten.convolution(buf49, primals_19, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf50, (8, 64, 8, 8), (4096, 64, 8, 1))
    buf51 = buf45; del buf45  # reuse
    buf52 = buf51; del buf51  # reuse
    buf55 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf53 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf54 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf56 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf57 = empty_strided((8, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_6(c_void_p(buf52.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()))
    del primals_180
    del primals_181
    del primals_21
    buf58 = aten.convolution(buf57, primals_22, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf58, (8, 256, 8, 8), (16384, 64, 8, 1))
    buf59 = buf36; del buf36  # reuse
    buf60 = buf59; del buf59  # reuse
    buf63 = as_strided(buf29, (256, ), (1, )); del buf29  # reuse
    buf61 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf62 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf64 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf65 = empty_strided((8, 256, 8, 8), (16384, 64, 8, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_7(c_void_p(buf60.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()))
    del primals_183
    del primals_184
    del primals_24
    buf66 = aten.convolution(buf65, primals_25, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf66, (8, 64, 8, 8), (4096, 64, 8, 1))
    buf67 = buf53; del buf53  # reuse
    buf68 = buf67; del buf67  # reuse
    buf71 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf69 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf70 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf72 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf73 = empty_strided((8, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_8(c_void_p(buf68.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()))
    del primals_186
    del primals_187
    del primals_27
    buf74 = aten.convolution(buf73, primals_28, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf74, (8, 64, 8, 8), (4096, 64, 8, 1))
    buf75 = buf69; del buf69  # reuse
    buf76 = buf75; del buf75  # reuse
    buf79 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf77 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf78 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf80 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf81 = empty_strided((8, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_9(c_void_p(buf76.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()))
    del buf77
    del primals_189
    del primals_190
    del primals_30
    buf82 = aten.convolution(buf81, primals_31, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf82, (8, 256, 8, 8), (16384, 64, 8, 1))
    buf83 = buf61; del buf61  # reuse
    buf84 = buf83; del buf83  # reuse
    buf87 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf85 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf86 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf88 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf89 = empty_strided((8, 256, 8, 8), (16384, 64, 8, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_10(c_void_p(buf84.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()))
    del primals_192
    del primals_193
    del primals_33
    buf90 = aten.convolution(buf89, primals_34, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf90, (8, 128, 8, 8), (8192, 64, 8, 1))
    buf91 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf92 = buf91; del buf91  # reuse
    buf95 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf93 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf94 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf96 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf97 = empty_strided((8, 128, 8, 8), (8192, 64, 8, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_11(c_void_p(buf92.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()))
    del primals_195
    del primals_196
    del primals_36
    buf98 = aten.convolution(buf97, primals_37, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf98, (8, 128, 4, 4), (2048, 16, 4, 1))
    buf99 = buf93; del buf93  # reuse
    buf100 = buf99; del buf99  # reuse
    buf103 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf101 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf102 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf104 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf105 = empty_strided((8, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_12(c_void_p(buf100.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(primals_199.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()))
    del primals_198
    del primals_199
    del primals_39
    buf106 = aten.convolution(buf105, primals_40, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf106, (8, 512, 4, 4), (8192, 16, 4, 1))
    buf107 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf108 = buf107; del buf107  # reuse
    buf111 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf109 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf110 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf112 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    kernel_cpp_13(c_void_p(buf108.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(primals_201.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf112.data_ptr()))
    del primals_201
    del primals_202
    buf113 = aten.convolution(buf89, primals_43, None, (2, 2), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf113, (8, 512, 4, 4), (8192, 16, 4, 1))
    buf114 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf115 = buf114; del buf114  # reuse
    buf118 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf116 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf117 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf119 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf120 = empty_strided((8, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    buf121 = buf120; del buf120  # reuse
    kernel_cpp_14(c_void_p(buf115.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(primals_204.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf119.data_ptr()))
    del primals_204
    del primals_205
    del primals_42
    del primals_45
    buf122 = aten.convolution(buf121, primals_46, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf122, (8, 128, 4, 4), (2048, 16, 4, 1))
    buf123 = buf101; del buf101  # reuse
    buf124 = buf123; del buf123  # reuse
    buf127 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf125 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf126 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf128 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf129 = empty_strided((8, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_15(c_void_p(buf124.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(primals_208.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()))
    del primals_207
    del primals_208
    del primals_48
    buf130 = aten.convolution(buf129, primals_49, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf130, (8, 128, 4, 4), (2048, 16, 4, 1))
    buf131 = buf125; del buf125  # reuse
    buf132 = buf131; del buf131  # reuse
    buf135 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf133 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf134 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf136 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf137 = empty_strided((8, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_16(c_void_p(buf132.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(primals_210.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()))
    del primals_210
    del primals_211
    del primals_51
    buf138 = aten.convolution(buf137, primals_52, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf138, (8, 512, 4, 4), (8192, 16, 4, 1))
    buf139 = buf116; del buf116  # reuse
    buf140 = buf139; del buf139  # reuse
    buf143 = as_strided(buf109, (512, ), (1, )); del buf109  # reuse
    buf141 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf142 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf144 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf145 = empty_strided((8, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_17(c_void_p(buf140.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(primals_213.data_ptr()), c_void_p(primals_214.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()))
    del primals_213
    del primals_214
    del primals_54
    buf146 = aten.convolution(buf145, primals_55, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf146, (8, 128, 4, 4), (2048, 16, 4, 1))
    buf147 = buf133; del buf133  # reuse
    buf148 = buf147; del buf147  # reuse
    buf151 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf149 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf150 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf152 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf153 = empty_strided((8, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_18(c_void_p(buf148.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(primals_216.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()))
    del primals_216
    del primals_217
    del primals_57
    buf154 = aten.convolution(buf153, primals_58, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf154, (8, 128, 4, 4), (2048, 16, 4, 1))
    buf155 = buf149; del buf149  # reuse
    buf156 = buf155; del buf155  # reuse
    buf159 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf157 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf158 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf160 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf161 = empty_strided((8, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_19(c_void_p(buf156.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()))
    del primals_219
    del primals_220
    del primals_60
    buf162 = aten.convolution(buf161, primals_61, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf162, (8, 512, 4, 4), (8192, 16, 4, 1))
    buf163 = buf141; del buf141  # reuse
    buf164 = buf163; del buf163  # reuse
    buf167 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf165 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf166 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf168 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf169 = empty_strided((8, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_20(c_void_p(buf164.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()))
    del primals_222
    del primals_223
    del primals_63
    buf170 = aten.convolution(buf169, primals_64, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf170, (8, 128, 4, 4), (2048, 16, 4, 1))
    buf171 = buf157; del buf157  # reuse
    buf172 = buf171; del buf171  # reuse
    buf175 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf173 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf174 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf176 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf177 = empty_strided((8, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_21(c_void_p(buf172.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(primals_226.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()))
    del primals_225
    del primals_226
    del primals_66
    buf178 = aten.convolution(buf177, primals_67, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf178, (8, 128, 4, 4), (2048, 16, 4, 1))
    buf179 = buf173; del buf173  # reuse
    buf180 = buf179; del buf179  # reuse
    buf183 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf181 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf182 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf184 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf185 = empty_strided((8, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_22(c_void_p(buf180.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()))
    del buf181
    del primals_228
    del primals_229
    del primals_69
    buf186 = aten.convolution(buf185, primals_70, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf186, (8, 512, 4, 4), (8192, 16, 4, 1))
    buf187 = buf165; del buf165  # reuse
    buf188 = buf187; del buf187  # reuse
    buf191 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf189 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf190 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf192 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf193 = empty_strided((8, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_23(c_void_p(buf188.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()))
    del primals_231
    del primals_232
    del primals_72
    buf194 = aten.convolution(buf193, primals_73, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf194, (8, 256, 4, 4), (4096, 16, 4, 1))
    buf195 = buf85; del buf85  # reuse
    buf196 = buf195; del buf195  # reuse
    buf199 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf197 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf198 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf200 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf201 = empty_strided((8, 256, 4, 4), (4096, 16, 4, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_24(c_void_p(buf196.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()))
    del primals_234
    del primals_235
    del primals_75
    buf202 = aten.convolution(buf201, primals_76, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf202, (8, 256, 2, 2), (1024, 4, 2, 1))
    buf203 = buf197; del buf197  # reuse
    buf204 = buf203; del buf203  # reuse
    buf207 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf205 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf206 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf208 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf209 = empty_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_25(c_void_p(buf204.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()))
    del primals_237
    del primals_238
    del primals_78
    buf210 = aten.convolution(buf209, primals_79, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf210, (8, 1024, 2, 2), (4096, 4, 2, 1))
    buf211 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf212 = buf211; del buf211  # reuse
    buf215 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf213 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf214 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf216 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    kernel_cpp_26(c_void_p(buf212.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf216.data_ptr()))
    del primals_240
    del primals_241
    buf217 = aten.convolution(buf193, primals_82, None, (2, 2), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf217, (8, 1024, 2, 2), (4096, 4, 2, 1))
    buf218 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf219 = buf218; del buf218  # reuse
    buf222 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf220 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf221 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf223 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf224 = empty_strided((8, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    buf225 = buf224; del buf224  # reuse
    kernel_cpp_27(c_void_p(buf219.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf223.data_ptr()))
    del primals_243
    del primals_244
    del primals_81
    del primals_84
    buf226 = aten.convolution(buf225, primals_85, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf226, (8, 256, 2, 2), (1024, 4, 2, 1))
    buf227 = buf205; del buf205  # reuse
    buf228 = buf227; del buf227  # reuse
    buf231 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf229 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf230 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf232 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf233 = empty_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_28(c_void_p(buf228.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(primals_246.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()))
    del primals_246
    del primals_247
    del primals_87
    buf234 = aten.convolution(buf233, primals_88, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf234, (8, 256, 2, 2), (1024, 4, 2, 1))
    buf235 = buf229; del buf229  # reuse
    buf236 = buf235; del buf235  # reuse
    buf239 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf237 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf238 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf240 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf241 = empty_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_29(c_void_p(buf236.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()))
    del primals_249
    del primals_250
    del primals_90
    buf242 = aten.convolution(buf241, primals_91, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf242, (8, 1024, 2, 2), (4096, 4, 2, 1))
    buf243 = buf220; del buf220  # reuse
    buf244 = buf243; del buf243  # reuse
    buf247 = as_strided(buf213, (1024, ), (1, )); del buf213  # reuse
    buf245 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf246 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf248 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf249 = empty_strided((8, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_30(c_void_p(buf244.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()))
    del primals_252
    del primals_253
    del primals_93
    buf250 = aten.convolution(buf249, primals_94, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf250, (8, 256, 2, 2), (1024, 4, 2, 1))
    buf251 = buf237; del buf237  # reuse
    buf252 = buf251; del buf251  # reuse
    buf255 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf253 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf254 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf256 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf257 = empty_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_31(c_void_p(buf252.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()))
    del primals_255
    del primals_256
    del primals_96
    buf258 = aten.convolution(buf257, primals_97, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf258, (8, 256, 2, 2), (1024, 4, 2, 1))
    buf259 = buf253; del buf253  # reuse
    buf260 = buf259; del buf259  # reuse
    buf263 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf261 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf262 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf264 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf265 = empty_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_32(c_void_p(buf260.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()))
    del primals_258
    del primals_259
    del primals_99
    buf266 = aten.convolution(buf265, primals_100, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf266, (8, 1024, 2, 2), (4096, 4, 2, 1))
    buf267 = buf245; del buf245  # reuse
    buf268 = buf267; del buf267  # reuse
    buf271 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf269 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf270 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf272 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf273 = empty_strided((8, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_33(c_void_p(buf268.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()))
    del primals_102
    del primals_261
    del primals_262
    buf274 = aten.convolution(buf273, primals_103, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf274, (8, 256, 2, 2), (1024, 4, 2, 1))
    buf275 = buf261; del buf261  # reuse
    buf276 = buf275; del buf275  # reuse
    buf279 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf277 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf278 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf280 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf281 = empty_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_34(c_void_p(buf276.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()))
    del primals_105
    del primals_264
    del primals_265
    buf282 = aten.convolution(buf281, primals_106, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf282, (8, 256, 2, 2), (1024, 4, 2, 1))
    buf283 = buf277; del buf277  # reuse
    buf284 = buf283; del buf283  # reuse
    buf287 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf285 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf286 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf288 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf289 = empty_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_35(c_void_p(buf284.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()))
    del primals_108
    del primals_267
    del primals_268
    buf290 = aten.convolution(buf289, primals_109, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf290, (8, 1024, 2, 2), (4096, 4, 2, 1))
    buf291 = buf269; del buf269  # reuse
    buf292 = buf291; del buf291  # reuse
    buf295 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf293 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf294 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf296 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf297 = empty_strided((8, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_36(c_void_p(buf292.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()))
    del primals_111
    del primals_270
    del primals_271
    buf298 = aten.convolution(buf297, primals_112, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf298, (8, 256, 2, 2), (1024, 4, 2, 1))
    buf299 = buf285; del buf285  # reuse
    buf300 = buf299; del buf299  # reuse
    buf303 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf301 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf302 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf304 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf305 = empty_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_37(c_void_p(buf300.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()))
    del primals_114
    del primals_273
    del primals_274
    buf306 = aten.convolution(buf305, primals_115, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf306, (8, 256, 2, 2), (1024, 4, 2, 1))
    buf307 = buf301; del buf301  # reuse
    buf308 = buf307; del buf307  # reuse
    buf311 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf309 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf310 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf312 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf313 = empty_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_38(c_void_p(buf308.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()))
    del primals_117
    del primals_276
    del primals_277
    buf314 = aten.convolution(buf313, primals_118, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf314, (8, 1024, 2, 2), (4096, 4, 2, 1))
    buf315 = buf293; del buf293  # reuse
    buf316 = buf315; del buf315  # reuse
    buf319 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf317 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf318 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf320 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf321 = empty_strided((8, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_39(c_void_p(buf316.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()))
    del primals_120
    del primals_279
    del primals_280
    buf322 = aten.convolution(buf321, primals_121, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf322, (8, 256, 2, 2), (1024, 4, 2, 1))
    buf323 = buf309; del buf309  # reuse
    buf324 = buf323; del buf323  # reuse
    buf327 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf325 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf326 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf328 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf329 = empty_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_40(c_void_p(buf324.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()))
    del primals_123
    del primals_282
    del primals_283
    buf330 = aten.convolution(buf329, primals_124, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf330, (8, 256, 2, 2), (1024, 4, 2, 1))
    buf331 = buf325; del buf325  # reuse
    buf332 = buf331; del buf331  # reuse
    buf335 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf333 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf334 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf336 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf337 = empty_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_41(c_void_p(buf332.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf337.data_ptr()))
    del buf333
    del primals_126
    del primals_285
    del primals_286
    buf338 = aten.convolution(buf337, primals_127, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf338, (8, 1024, 2, 2), (4096, 4, 2, 1))
    buf339 = buf317; del buf317  # reuse
    buf340 = buf339; del buf339  # reuse
    buf343 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf341 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf342 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf344 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf345 = empty_strided((8, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_42(c_void_p(buf340.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf345.data_ptr()))
    del buf341
    del primals_129
    del primals_288
    del primals_289
    buf346 = aten.convolution(buf345, primals_130, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf346, (8, 512, 2, 2), (2048, 4, 2, 1))
    buf347 = buf189; del buf189  # reuse
    buf348 = buf347; del buf347  # reuse
    buf351 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf349 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf350 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf352 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf353 = empty_strided((8, 512, 2, 2), (2048, 4, 2, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_43(c_void_p(buf348.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(primals_132.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()))
    del primals_132
    del primals_291
    del primals_292
    buf354 = aten.convolution(buf353, primals_133, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf354, (8, 512, 1, 1), (512, 1, 1, 1))
    buf355 = buf349; del buf349  # reuse
    buf356 = buf355; del buf355  # reuse
    buf358 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf357 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf359 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf360 = empty_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_44(c_void_p(buf356.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf360.data_ptr()))
    del primals_135
    del primals_294
    del primals_295
    buf361 = aten.convolution(buf360, primals_136, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf361, (8, 2048, 1, 1), (2048, 1, 1, 1))
    buf362 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf363 = buf362; del buf362  # reuse
    buf365 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf364 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf366 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    kernel_cpp_45(c_void_p(buf363.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf366.data_ptr()))
    del primals_297
    del primals_298
    buf367 = aten.convolution(buf345, primals_139, None, (2, 2), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf367, (8, 2048, 1, 1), (2048, 1, 1, 1))
    buf368 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf369 = buf368; del buf368  # reuse
    buf371 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf370 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf372 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf373 = empty_strided((8, 2048, 1, 1), (2048, 1, 16384, 16384), device='cpu', dtype=torch.float32)
    buf374 = as_strided(buf373, (8, 2048, 1, 1), (2048, 1, 1, 1)); del buf373  # reuse
    kernel_cpp_46(c_void_p(buf369.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf372.data_ptr()))
    del primals_138
    del primals_141
    del primals_300
    del primals_301
    buf375 = aten.convolution(buf374, primals_142, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf375, (8, 512, 1, 1), (512, 1, 1, 1))
    buf376 = buf357; del buf357  # reuse
    buf377 = buf376; del buf376  # reuse
    buf379 = as_strided(buf356, (512, ), (1, )); del buf356  # reuse
    buf378 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf380 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf381 = empty_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_47(c_void_p(buf377.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(primals_144.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf381.data_ptr()))
    del primals_144
    del primals_303
    del primals_304
    buf382 = aten.convolution(buf381, primals_145, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf382, (8, 512, 1, 1), (512, 1, 1, 1))
    buf383 = buf378; del buf378  # reuse
    buf384 = buf383; del buf383  # reuse
    buf386 = as_strided(buf377, (512, ), (1, )); del buf377  # reuse
    buf385 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf387 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf388 = empty_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_48(c_void_p(buf384.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf388.data_ptr()))
    del primals_147
    del primals_306
    del primals_307
    buf389 = aten.convolution(buf388, primals_148, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf389, (8, 2048, 1, 1), (2048, 1, 1, 1))
    buf390 = buf370; del buf370  # reuse
    buf391 = buf390; del buf390  # reuse
    buf393 = as_strided(buf369, (2048, ), (1, )); del buf369  # reuse
    buf392 = buf364; del buf364  # reuse
    buf394 = as_strided(buf363, (2048, ), (1, )); del buf363  # reuse
    buf395 = empty_strided((8, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_49(c_void_p(buf391.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(primals_150.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf395.data_ptr()))
    del primals_150
    del primals_309
    del primals_310
    buf396 = aten.convolution(buf395, primals_151, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf396, (8, 512, 1, 1), (512, 1, 1, 1))
    buf397 = buf385; del buf385  # reuse
    buf398 = buf397; del buf397  # reuse
    buf400 = as_strided(buf384, (512, ), (1, )); del buf384  # reuse
    buf399 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf401 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf402 = empty_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_50(c_void_p(buf398.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf402.data_ptr()))
    del primals_153
    del primals_312
    del primals_313
    buf403 = aten.convolution(buf402, primals_154, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf403, (8, 512, 1, 1), (512, 1, 1, 1))
    buf404 = buf399; del buf399  # reuse
    buf405 = buf404; del buf404  # reuse
    buf407 = as_strided(buf398, (512, ), (1, )); del buf398  # reuse
    buf406 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf408 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf409 = empty_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_51(c_void_p(buf405.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(primals_316.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(primals_156.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()))
    del buf405
    del buf406
    del primals_156
    del primals_315
    del primals_316
    buf410 = aten.convolution(buf409, primals_157, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf410, (8, 2048, 1, 1), (2048, 1, 1, 1))
    buf411 = buf392; del buf392  # reuse
    buf412 = buf411; del buf411  # reuse
    buf414 = as_strided(buf391, (2048, ), (1, )); del buf391  # reuse
    buf413 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf415 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf416 = empty_strided((8, 2048, 1, 1), (2048, 1, 16384, 16384), device='cpu', dtype=torch.float32)
    buf417 = empty_strided((8, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    buf419 = empty_strided((8, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.bool)
    kernel_cpp_52(c_void_p(buf412.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(primals_318.data_ptr()), c_void_p(primals_319.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf419.data_ptr()))
    del buf412
    del buf413
    del buf416
    del primals_159
    del primals_318
    del primals_319
    buf418 = empty_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    extern_kernels.addmm(primals_161, buf417, as_strided(primals_160, (2048, 1000), (1, 2048)), alpha=1, beta=1, out=buf418)
    del primals_161
    buf420 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf421 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf422 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf423 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf424 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf425 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf426 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf427 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf428 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf429 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf430 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf431 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf432 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf433 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf434 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf435 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf436 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf437 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf438 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf439 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf440 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf441 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf442 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf443 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf444 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf445 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf446 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf447 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf448 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf449 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf450 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf451 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf452 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf453 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf454 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf455 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf456 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf457 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf458 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf459 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf460 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf461 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf462 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf463 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf464 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf465 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf466 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf467 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf468 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf469 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf470 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf471 = empty_strided((), (), device='cpu', dtype=torch.int64)
    buf472 = empty_strided((), (), device='cpu', dtype=torch.int64)
    kernel_cpp_53(c_void_p(primals_164.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(primals_182.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(primals_188.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(primals_200.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(primals_206.data_ptr()), c_void_p(primals_209.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(primals_215.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(primals_221.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(primals_233.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(primals_239.data_ptr()), c_void_p(primals_242.data_ptr()), c_void_p(primals_245.data_ptr()), c_void_p(primals_248.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(primals_254.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(primals_266.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(primals_272.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_278.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(primals_311.data_ptr()), c_void_p(primals_314.data_ptr()), c_void_p(primals_317.data_ptr()), c_void_p(primals_320.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf472.data_ptr()))
    del primals_164
    del primals_167
    del primals_170
    del primals_173
    del primals_176
    del primals_179
    del primals_182
    del primals_185
    del primals_188
    del primals_191
    del primals_194
    del primals_197
    del primals_200
    del primals_203
    del primals_206
    del primals_209
    del primals_212
    del primals_215
    del primals_218
    del primals_221
    del primals_224
    del primals_227
    del primals_230
    del primals_233
    del primals_236
    del primals_239
    del primals_242
    del primals_245
    del primals_248
    del primals_251
    del primals_254
    del primals_257
    del primals_260
    del primals_263
    del primals_266
    del primals_269
    del primals_272
    del primals_275
    del primals_278
    del primals_281
    del primals_284
    del primals_287
    del primals_290
    del primals_293
    del primals_296
    del primals_299
    del primals_302
    del primals_305
    del primals_308
    del primals_311
    del primals_314
    del primals_317
    del primals_320
    return (buf5, buf6, buf420, buf15, buf16, buf421, buf23, buf24, buf422, buf31, buf32, buf423, buf38, buf39, buf424, buf47, buf48, buf425, buf55, buf56, buf426, buf63, buf64, buf427, buf71, buf72, buf428, buf79, buf80, buf429, buf87, buf88, buf430, buf95, buf96, buf431, buf103, buf104, buf432, buf111, buf112, buf433, buf118, buf119, buf434, buf127, buf128, buf435, buf135, buf136, buf436, buf143, buf144, buf437, buf151, buf152, buf438, buf159, buf160, buf439, buf167, buf168, buf440, buf175, buf176, buf441, buf183, buf184, buf442, buf191, buf192, buf443, buf199, buf200, buf444, buf207, buf208, buf445, buf215, buf216, buf446, buf222, buf223, buf447, buf231, buf232, buf448, buf239, buf240, buf449, buf247, buf248, buf450, buf255, buf256, buf451, buf263, buf264, buf452, buf271, buf272, buf453, buf279, buf280, buf454, buf287, buf288, buf455, buf295, buf296, buf456, buf303, buf304, buf457, buf311, buf312, buf458, buf319, buf320, buf459, buf327, buf328, buf460, buf335, buf336, buf461, buf343, buf344, buf462, buf351, buf352, buf463, buf358, buf359, buf464, buf365, buf366, buf465, buf371, buf372, buf466, buf379, buf380, buf467, buf386, buf387, buf468, buf393, buf394, buf469, buf400, buf401, buf470, buf407, buf408, buf471, buf414, buf415, buf472, buf418, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_321, buf0, buf4, buf7, buf8, buf9, buf10, buf14, buf17, buf18, buf22, buf25, buf26, buf30, buf33, buf37, buf41, buf42, buf46, buf49, buf50, buf54, buf57, buf58, buf62, buf65, buf66, buf70, buf73, buf74, buf78, buf81, buf82, buf86, buf89, buf90, buf94, buf97, buf98, buf102, buf105, buf106, buf110, buf113, buf117, buf121, buf122, buf126, buf129, buf130, buf134, buf137, buf138, buf142, buf145, buf146, buf150, buf153, buf154, buf158, buf161, buf162, buf166, buf169, buf170, buf174, buf177, buf178, buf182, buf185, buf186, buf190, buf193, buf194, buf198, buf201, buf202, buf206, buf209, buf210, buf214, buf217, buf221, buf225, buf226, buf230, buf233, buf234, buf238, buf241, buf242, buf246, buf249, buf250, buf254, buf257, buf258, buf262, buf265, buf266, buf270, buf273, buf274, buf278, buf281, buf282, buf286, buf289, buf290, buf294, buf297, buf298, buf302, buf305, buf306, buf310, buf313, buf314, buf318, buf321, buf322, buf326, buf329, buf330, buf334, buf337, buf338, buf342, buf345, buf346, buf350, buf353, buf354, buf360, buf361, buf367, buf374, buf375, buf381, buf382, buf388, buf389, buf395, buf396, buf402, buf403, buf409, buf410, buf417, as_strided(primals_160, (1000, 2048), (2048, 1)), buf419, as_strided(buf348, (1, 512, 1, 1), (512, 1, 1, 1)), as_strided(buf340, (1, 1024, 1, 1), (1024, 1, 1, 1)), as_strided(buf332, (1, 256, 1, 1), (256, 1, 1, 1)), as_strided(buf324, (1, 256, 1, 1), (256, 1, 1, 1)), as_strided(buf316, (1, 1024, 1, 1), (1024, 1, 1, 1)), as_strided(buf308, (1, 256, 1, 1), (256, 1, 1, 1)), as_strided(buf300, (1, 256, 1, 1), (256, 1, 1, 1)), as_strided(buf292, (1, 1024, 1, 1), (1024, 1, 1, 1)), as_strided(buf284, (1, 256, 1, 1), (256, 1, 1, 1)), as_strided(buf276, (1, 256, 1, 1), (256, 1, 1, 1)), as_strided(buf268, (1, 1024, 1, 1), (1024, 1, 1, 1)), as_strided(buf260, (1, 256, 1, 1), (256, 1, 1, 1)), as_strided(buf252, (1, 256, 1, 1), (256, 1, 1, 1)), as_strided(buf244, (1, 1024, 1, 1), (1024, 1, 1, 1)), as_strided(buf236, (1, 256, 1, 1), (256, 1, 1, 1)), as_strided(buf228, (1, 256, 1, 1), (256, 1, 1, 1)), as_strided(buf219, (1, 1024, 1, 1), (1024, 1, 1, 1)), as_strided(buf212, (1, 1024, 1, 1), (1024, 1, 1, 1)), as_strided(buf204, (1, 256, 1, 1), (256, 1, 1, 1)), as_strided(buf196, (1, 256, 1, 1), (256, 1, 1, 1)), as_strided(buf188, (1, 512, 1, 1), (512, 1, 1, 1)), as_strided(buf180, (1, 128, 1, 1), (128, 1, 1, 1)), as_strided(buf172, (1, 128, 1, 1), (128, 1, 1, 1)), as_strided(buf164, (1, 512, 1, 1), (512, 1, 1, 1)), as_strided(buf156, (1, 128, 1, 1), (128, 1, 1, 1)), as_strided(buf148, (1, 128, 1, 1), (128, 1, 1, 1)), as_strided(buf140, (1, 512, 1, 1), (512, 1, 1, 1)), as_strided(buf132, (1, 128, 1, 1), (128, 1, 1, 1)), as_strided(buf124, (1, 128, 1, 1), (128, 1, 1, 1)), as_strided(buf115, (1, 512, 1, 1), (512, 1, 1, 1)), as_strided(buf108, (1, 512, 1, 1), (512, 1, 1, 1)), as_strided(buf100, (1, 128, 1, 1), (128, 1, 1, 1)), as_strided(buf92, (1, 128, 1, 1), (128, 1, 1, 1)), as_strided(buf84, (1, 256, 1, 1), (256, 1, 1, 1)), as_strided(buf76, (1, 64, 1, 1), (64, 1, 1, 1)), as_strided(buf68, (1, 64, 1, 1), (64, 1, 1, 1)), as_strided(buf60, (1, 256, 1, 1), (256, 1, 1, 1)), as_strided(buf52, (1, 64, 1, 1), (64, 1, 1, 1)), as_strided(buf44, (1, 64, 1, 1), (64, 1, 1, 1)), as_strided(buf35, (1, 256, 1, 1), (256, 1, 1, 1)), as_strided(buf28, (1, 256, 1, 1), (256, 1, 1, 1)), as_strided(buf20, (1, 64, 1, 1), (64, 1, 1, 1)), as_strided(buf12, (1, 64, 1, 1), (64, 1, 1, 1)), as_strided(buf2, (1, 64, 1, 1), (64, 1, 1, 1)), )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((1000, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_165 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_168 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_171 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_174 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_177 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_180 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_183 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_186 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_189 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_192 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_195 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_198 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_201 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_204 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_207 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_210 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_213 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_216 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_219 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_222 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_225 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_228 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_231 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_234 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_237 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_240 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_243 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_246 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_249 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_252 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_255 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_258 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_261 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_264 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_267 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_270 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_273 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_276 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_279 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_282 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_285 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_288 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_291 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_294 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_297 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_300 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_303 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_306 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_309 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_312 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_314 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_315 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_316 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_317 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_318 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_319 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_320 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_321 = rand_strided((8, 3, 32, 32), (3072, 1024, 32, 1), device='cpu', dtype=torch.float32)
    print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321]))
