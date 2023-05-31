
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
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long i1=0; i1<16; i1+=1)
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
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(4096));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=64; i0<64; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(4096);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
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
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<16; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<64; i1+=1)
                {
                    for(long i2=0; i2<16; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i2) + (256*i1) + (16384*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(4096));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                        tmp14.store(in_out_ptr1 + (16*i2) + (256*i1) + (16384*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=256; i2<256; i2+=1)
                    {
                        auto tmp0 = in_out_ptr1[i2 + (256*i1) + (16384*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr3[i1];
                        auto tmp10 = in_ptr2[i1];
                        auto tmp12 = in_ptr3[i1];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(4096);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = tmp13 * (tmp13>0);
                        in_out_ptr1[i2 + (256*i1) + (16384*i0)] = tmp14;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<1024; i0+=1)
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
                            auto tmp12 = in_out_ptr1[(-17) + (2*i2) + (32*i1) + (256*i0)];
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
                            auto tmp20 = in_out_ptr1[(-16) + (2*i2) + (32*i1) + (256*i0)];
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
                            auto tmp29 = in_out_ptr1[(-15) + (2*i2) + (32*i1) + (256*i0)];
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
                            auto tmp38 = in_out_ptr1[(-1) + (2*i2) + (32*i1) + (256*i0)];
                            return tmp38;
                        }
                        ;
                        auto tmp39 = tmp36 ? tmp37() : -std::numeric_limits<decltype(tmp37())>::infinity();
                        auto tmp40 = (tmp31 != tmp31) ? tmp31 : std::max(tmp39, tmp31);
                        auto tmp41 = tmp35 & tmp17;
                        auto tmp42 = [&]
                        {
                            auto tmp43 = in_out_ptr1[(2*i2) + (32*i1) + (256*i0)];
                            return tmp43;
                        }
                        ;
                        auto tmp44 = tmp41 ? tmp42() : -std::numeric_limits<decltype(tmp42())>::infinity();
                        auto tmp45 = (tmp40 != tmp40) ? tmp40 : std::max(tmp44, tmp40);
                        auto tmp46 = tmp35 & tmp26;
                        auto tmp47 = [&]
                        {
                            auto tmp48 = in_out_ptr1[1 + (2*i2) + (32*i1) + (256*i0)];
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
                            auto tmp57 = in_out_ptr1[15 + (2*i2) + (32*i1) + (256*i0)];
                            return tmp57;
                        }
                        ;
                        auto tmp58 = tmp55 ? tmp56() : -std::numeric_limits<decltype(tmp56())>::infinity();
                        auto tmp59 = (tmp50 != tmp50) ? tmp50 : std::max(tmp58, tmp50);
                        auto tmp60 = tmp54 & tmp17;
                        auto tmp61 = [&]
                        {
                            auto tmp62 = in_out_ptr1[16 + (2*i2) + (32*i1) + (256*i0)];
                            return tmp62;
                        }
                        ;
                        auto tmp63 = tmp60 ? tmp61() : -std::numeric_limits<decltype(tmp61())>::infinity();
                        auto tmp64 = (tmp59 != tmp59) ? tmp59 : std::max(tmp63, tmp59);
                        auto tmp65 = tmp54 & tmp26;
                        auto tmp66 = [&]
                        {
                            auto tmp67 = in_out_ptr1[17 + (2*i2) + (32*i1) + (256*i0)];
                            return tmp67;
                        }
                        ;
                        auto tmp68 = tmp65 ? tmp66() : -std::numeric_limits<decltype(tmp66())>::infinity();
                        auto tmp69 = (tmp64 != tmp64) ? tmp64 : std::max(tmp68, tmp64);
                        out_ptr4[i2 + (8*i1) + (64*i0)] = tmp69;
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
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long i1=0; i1<16; i1+=1)
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
        #pragma omp single
        {
            {
                for(long i0=0; i0<4; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(1024));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=64; i0<64; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(1024);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
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
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<16; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<64; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1024));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                        tmp14.store(in_out_ptr1 + (16*i2) + (64*i1) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_out_ptr1[i2 + (64*i1) + (4096*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr3[i1];
                        auto tmp10 = in_ptr2[i1];
                        auto tmp12 = in_ptr3[i1];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1024);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = tmp13 * (tmp13>0);
                        in_out_ptr1[i2 + (64*i1) + (4096*i0)] = tmp14;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_2 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long i1=0; i1<16; i1+=1)
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
        #pragma omp single
        {
            {
                for(long i0=0; i0<4; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(1024));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=64; i0<64; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(1024);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
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
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<16; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<64; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1024));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                        tmp14.store(in_out_ptr1 + (16*i2) + (64*i1) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_out_ptr1[i2 + (64*i1) + (4096*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr3[i1];
                        auto tmp10 = in_ptr2[i1];
                        auto tmp12 = in_ptr3[i1];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1024);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = tmp13 * (tmp13>0);
                        in_out_ptr1[i2 + (64*i1) + (4096*i0)] = tmp14;
                    }
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
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
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
                    for(long i1=0; i1<16; i1+=1)
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
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(1024));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(1024);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
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
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
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
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
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
                    for(long i1=0; i1<16; i1+=1)
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
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(1024));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(1024);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
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
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<16; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<256; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i2) + (64*i1) + (16384*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_ptr2[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i1) + (16384*i0));
                        auto tmp15 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr3[i1]);
                        auto tmp22 = at::vec::Vectorized<float>(in_ptr6[i1]);
                        auto tmp24 = at::vec::Vectorized<float>(in_ptr7[i1]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1024));
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
                        tmp26.store(in_out_ptr1 + (16*i2) + (64*i1) + (16384*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_out_ptr1[i2 + (64*i1) + (16384*i0)];
                        auto tmp1 = in_ptr2[i1];
                        auto tmp3 = in_ptr3[i1];
                        auto tmp10 = in_ptr4[i1];
                        auto tmp12 = in_ptr5[i1];
                        auto tmp14 = in_ptr0[i2 + (64*i1) + (16384*i0)];
                        auto tmp15 = in_out_ptr0[i1];
                        auto tmp17 = out_ptr3[i1];
                        auto tmp22 = in_ptr6[i1];
                        auto tmp24 = in_ptr7[i1];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1024);
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
                        in_out_ptr1[i2 + (64*i1) + (16384*i0)] = tmp26;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<16384; i0+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + 16*i0);
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr1 + 16*i0);
            }
            #pragma omp for simd simdlen(8) 
            for(long i0=262144; i0<262144; i0+=1)
            {
                auto tmp0 = in_out_ptr1[i0];
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
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long i1=0; i1<16; i1+=1)
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
        #pragma omp single
        {
            {
                for(long i0=0; i0<4; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(1024));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=64; i0<64; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(1024);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
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
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<16; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<64; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1024));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                        tmp14.store(in_out_ptr1 + (16*i2) + (64*i1) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_out_ptr1[i2 + (64*i1) + (4096*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr3[i1];
                        auto tmp10 = in_ptr2[i1];
                        auto tmp12 = in_ptr3[i1];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1024);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = tmp13 * (tmp13>0);
                        in_out_ptr1[i2 + (64*i1) + (4096*i0)] = tmp14;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_6 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long i1=0; i1<16; i1+=1)
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
        #pragma omp single
        {
            {
                for(long i0=0; i0<4; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(1024));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=64; i0<64; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(1024);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
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
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<16; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<64; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1024));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                        tmp14.store(in_out_ptr1 + (16*i2) + (64*i1) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_out_ptr1[i2 + (64*i1) + (4096*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr3[i1];
                        auto tmp10 = in_ptr2[i1];
                        auto tmp12 = in_ptr3[i1];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1024);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = tmp13 * (tmp13>0);
                        in_out_ptr1[i2 + (64*i1) + (4096*i0)] = tmp14;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_7 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
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
                    for(long i1=0; i1<16; i1+=1)
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
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(1024));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(1024);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
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
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<16; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<256; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i1) + (16384*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i2) + (64*i1) + (16384*i0));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1024));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                        tmp16.store(in_out_ptr1 + (16*i2) + (64*i1) + (16384*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i1) + (16384*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr3[i1];
                        auto tmp10 = in_ptr2[i1];
                        auto tmp12 = in_ptr3[i1];
                        auto tmp14 = in_out_ptr1[i2 + (64*i1) + (16384*i0)];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1024);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp15 * (tmp15>0);
                        in_out_ptr1[i2 + (64*i1) + (16384*i0)] = tmp16;
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
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long i1=0; i1<16; i1+=1)
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
        #pragma omp single
        {
            {
                for(long i0=0; i0<4; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(1024));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=64; i0<64; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(1024);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
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
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<16; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<64; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1024));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                        tmp14.store(in_out_ptr1 + (16*i2) + (64*i1) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_out_ptr1[i2 + (64*i1) + (4096*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr3[i1];
                        auto tmp10 = in_ptr2[i1];
                        auto tmp12 = in_ptr3[i1];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1024);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = tmp13 * (tmp13>0);
                        in_out_ptr1[i2 + (64*i1) + (4096*i0)] = tmp14;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_9 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long i1=0; i1<16; i1+=1)
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
        #pragma omp single
        {
            {
                for(long i0=0; i0<4; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(1024));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=64; i0<64; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(1024);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
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
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<16; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<64; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1024));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                        tmp14.store(in_out_ptr1 + (16*i2) + (64*i1) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_out_ptr1[i2 + (64*i1) + (4096*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr3[i1];
                        auto tmp10 = in_ptr2[i1];
                        auto tmp12 = in_ptr3[i1];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1024);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = tmp13 * (tmp13>0);
                        in_out_ptr1[i2 + (64*i1) + (4096*i0)] = tmp14;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_10 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
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
                    for(long i1=0; i1<16; i1+=1)
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
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(1024));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(1024);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
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
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<16; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<256; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i1) + (16384*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i2) + (64*i1) + (16384*i0));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1024));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                        tmp16.store(in_out_ptr1 + (16*i2) + (64*i1) + (16384*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i1) + (16384*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr3[i1];
                        auto tmp10 = in_ptr2[i1];
                        auto tmp12 = in_ptr3[i1];
                        auto tmp14 = in_out_ptr1[i2 + (64*i1) + (16384*i0)];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1024);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp15 * (tmp15>0);
                        in_out_ptr1[i2 + (64*i1) + (16384*i0)] = tmp16;
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
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long i1=0; i1<16; i1+=1)
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
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(1024));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=128; i0<128; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(1024);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
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
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<16; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<128; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i2) + (64*i1) + (8192*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1024));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                        tmp14.store(in_out_ptr1 + (16*i2) + (64*i1) + (8192*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_out_ptr1[i2 + (64*i1) + (8192*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr3[i1];
                        auto tmp10 = in_ptr2[i1];
                        auto tmp12 = in_ptr3[i1];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1024);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = tmp13 * (tmp13>0);
                        in_out_ptr1[i2 + (64*i1) + (8192*i0)] = tmp14;
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
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(256);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
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
                for(long i1=0; i1<16; i1+=1)
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
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<128; i1+=1)
            {
                for(long i2=0; i2<1; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i1) + (16*i2) + (2048*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(256));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(in_out_ptr1 + (16*i1) + (16*i2) + (2048*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=16; i2<16; i2+=1)
                {
                    auto tmp0 = in_out_ptr1[i2 + (16*i1) + (2048*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr3[i1];
                    auto tmp10 = in_ptr2[i1];
                    auto tmp12 = in_ptr3[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(256);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    in_out_ptr1[i2 + (16*i1) + (2048*i0)] = tmp14;
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
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
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
                    for(long i1=0; i1<16; i1+=1)
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
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=512; i0<512; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(256);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
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
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
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
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
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
                    for(long i1=0; i1<16; i1+=1)
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
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=512; i0<512; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(256);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
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
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<16; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<512; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_ptr2[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp15 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr3[i1]);
                        auto tmp22 = at::vec::Vectorized<float>(in_ptr6[i1]);
                        auto tmp24 = at::vec::Vectorized<float>(in_ptr7[i1]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(256));
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
                        tmp26.store(in_out_ptr1 + (16*i1) + (16*i2) + (8192*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_out_ptr1[i2 + (16*i1) + (8192*i0)];
                        auto tmp1 = in_ptr2[i1];
                        auto tmp3 = in_ptr3[i1];
                        auto tmp10 = in_ptr4[i1];
                        auto tmp12 = in_ptr5[i1];
                        auto tmp14 = in_ptr0[i2 + (16*i1) + (8192*i0)];
                        auto tmp15 = in_out_ptr0[i1];
                        auto tmp17 = out_ptr3[i1];
                        auto tmp22 = in_ptr6[i1];
                        auto tmp24 = in_ptr7[i1];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(256);
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
                        in_out_ptr1[i2 + (16*i1) + (8192*i0)] = tmp26;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<8192; i0+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + 16*i0);
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr1 + 16*i0);
            }
            #pragma omp for simd simdlen(8) 
            for(long i0=131072; i0<131072; i0+=1)
            {
                auto tmp0 = in_out_ptr1[i0];
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
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(256);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
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
                for(long i1=0; i1<16; i1+=1)
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
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<128; i1+=1)
            {
                for(long i2=0; i2<1; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i1) + (16*i2) + (2048*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(256));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(in_out_ptr1 + (16*i1) + (16*i2) + (2048*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=16; i2<16; i2+=1)
                {
                    auto tmp0 = in_out_ptr1[i2 + (16*i1) + (2048*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr3[i1];
                    auto tmp10 = in_ptr2[i1];
                    auto tmp12 = in_ptr3[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(256);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    in_out_ptr1[i2 + (16*i1) + (2048*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_16 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(256);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
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
                for(long i1=0; i1<16; i1+=1)
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
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<128; i1+=1)
            {
                for(long i2=0; i2<1; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i1) + (16*i2) + (2048*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(256));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(in_out_ptr1 + (16*i1) + (16*i2) + (2048*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=16; i2<16; i2+=1)
                {
                    auto tmp0 = in_out_ptr1[i2 + (16*i1) + (2048*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr3[i1];
                    auto tmp10 = in_ptr2[i1];
                    auto tmp12 = in_ptr3[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(256);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    in_out_ptr1[i2 + (16*i1) + (2048*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_17 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long i1=0; i1<16; i1+=1)
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
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=512; i0<512; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(256);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
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
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<16; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<512; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(256));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                        tmp16.store(in_out_ptr1 + (16*i1) + (16*i2) + (8192*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_out_ptr1[i2 + (16*i1) + (8192*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr3[i1];
                        auto tmp10 = in_ptr2[i1];
                        auto tmp12 = in_ptr3[i1];
                        auto tmp14 = in_ptr4[i2 + (16*i1) + (8192*i0)];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(256);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp15 * (tmp15>0);
                        in_out_ptr1[i2 + (16*i1) + (8192*i0)] = tmp16;
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
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(256);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
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
                for(long i1=0; i1<16; i1+=1)
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
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<128; i1+=1)
            {
                for(long i2=0; i2<1; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i1) + (16*i2) + (2048*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(256));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(in_out_ptr1 + (16*i1) + (16*i2) + (2048*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=16; i2<16; i2+=1)
                {
                    auto tmp0 = in_out_ptr1[i2 + (16*i1) + (2048*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr3[i1];
                    auto tmp10 = in_ptr2[i1];
                    auto tmp12 = in_ptr3[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(256);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    in_out_ptr1[i2 + (16*i1) + (2048*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_19 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(256);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
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
                for(long i1=0; i1<16; i1+=1)
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
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<128; i1+=1)
            {
                for(long i2=0; i2<1; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i1) + (16*i2) + (2048*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(256));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(in_out_ptr1 + (16*i1) + (16*i2) + (2048*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=16; i2<16; i2+=1)
                {
                    auto tmp0 = in_out_ptr1[i2 + (16*i1) + (2048*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr3[i1];
                    auto tmp10 = in_ptr2[i1];
                    auto tmp12 = in_ptr3[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(256);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    in_out_ptr1[i2 + (16*i1) + (2048*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_20 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
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
                    for(long i1=0; i1<16; i1+=1)
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
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=512; i0<512; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(256);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
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
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<16; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<512; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(256));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                        tmp16.store(in_out_ptr1 + (16*i1) + (16*i2) + (8192*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i1) + (8192*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr3[i1];
                        auto tmp10 = in_ptr2[i1];
                        auto tmp12 = in_ptr3[i1];
                        auto tmp14 = in_out_ptr1[i2 + (16*i1) + (8192*i0)];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(256);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp15 * (tmp15>0);
                        in_out_ptr1[i2 + (16*i1) + (8192*i0)] = tmp16;
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
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(256);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
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
                for(long i1=0; i1<16; i1+=1)
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
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<128; i1+=1)
            {
                for(long i2=0; i2<1; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i1) + (16*i2) + (2048*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(256));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(in_out_ptr1 + (16*i1) + (16*i2) + (2048*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=16; i2<16; i2+=1)
                {
                    auto tmp0 = in_out_ptr1[i2 + (16*i1) + (2048*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr3[i1];
                    auto tmp10 = in_ptr2[i1];
                    auto tmp12 = in_ptr3[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(256);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    in_out_ptr1[i2 + (16*i1) + (2048*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_22 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(256);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
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
                for(long i1=0; i1<16; i1+=1)
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
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<128; i1+=1)
            {
                for(long i2=0; i2<1; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i1) + (16*i2) + (2048*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(256));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(in_out_ptr1 + (16*i1) + (16*i2) + (2048*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=16; i2<16; i2+=1)
                {
                    auto tmp0 = in_out_ptr1[i2 + (16*i1) + (2048*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr3[i1];
                    auto tmp10 = in_ptr2[i1];
                    auto tmp12 = in_ptr3[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(256);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    in_out_ptr1[i2 + (16*i1) + (2048*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_23 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
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
                    for(long i1=0; i1<16; i1+=1)
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
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=512; i0<512; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(256);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
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
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<16; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<512; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(256));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                        tmp16.store(in_out_ptr1 + (16*i1) + (16*i2) + (8192*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i1) + (8192*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr3[i1];
                        auto tmp10 = in_ptr2[i1];
                        auto tmp12 = in_ptr3[i1];
                        auto tmp14 = in_out_ptr1[i2 + (16*i1) + (8192*i0)];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(256);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp15 * (tmp15>0);
                        in_out_ptr1[i2 + (16*i1) + (8192*i0)] = tmp16;
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
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long i1=0; i1<16; i1+=1)
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
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(256);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
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
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<16; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<256; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i1) + (16*i2) + (4096*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(256));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                        tmp14.store(in_out_ptr1 + (16*i1) + (16*i2) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_out_ptr1[i2 + (16*i1) + (4096*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr3[i1];
                        auto tmp10 = in_ptr2[i1];
                        auto tmp12 = in_ptr3[i1];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(256);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = tmp13 * (tmp13>0);
                        in_out_ptr1[i2 + (16*i1) + (4096*i0)] = tmp14;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_25 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
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
                for(long i1=0; i1<16; i1+=1)
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
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (4*i1) + (16*i2) + (1024*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(64));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(in_out_ptr1 + (4*i1) + (16*i2) + (1024*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_out_ptr1[i2 + (4*i1) + (1024*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr3[i1];
                    auto tmp10 = in_ptr2[i1];
                    auto tmp12 = in_ptr3[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(64);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    in_out_ptr1[i2 + (4*i1) + (1024*i0)] = tmp14;
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
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<1024; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    for(long i1=0; i1<16; i1+=1)
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
        #pragma omp single
        {
            {
                for(long i0=0; i0<64; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=1024; i0<1024; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(64);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<1024; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp4 = 0;
                    auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
                }
            }
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
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<1024; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    for(long i1=0; i1<16; i1+=1)
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
        #pragma omp single
        {
            {
                for(long i0=0; i0<64; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=1024; i0<1024; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(64);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<1024; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp4 = 0;
                    auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<16; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<1024; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_ptr2[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp15 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr3[i1]);
                        auto tmp22 = at::vec::Vectorized<float>(in_ptr6[i1]);
                        auto tmp24 = at::vec::Vectorized<float>(in_ptr7[i1]);
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(64));
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
                        tmp26.store(in_out_ptr1 + (4*i1) + (16*i2) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_out_ptr1[i2 + (4*i1) + (4096*i0)];
                        auto tmp1 = in_ptr2[i1];
                        auto tmp3 = in_ptr3[i1];
                        auto tmp10 = in_ptr4[i1];
                        auto tmp12 = in_ptr5[i1];
                        auto tmp14 = in_ptr0[i2 + (4*i1) + (4096*i0)];
                        auto tmp15 = in_out_ptr0[i1];
                        auto tmp17 = out_ptr3[i1];
                        auto tmp22 = in_ptr6[i1];
                        auto tmp24 = in_ptr7[i1];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(64);
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
                        in_out_ptr1[i2 + (4*i1) + (4096*i0)] = tmp26;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<4096; i0+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + 16*i0);
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr1 + 16*i0);
            }
            #pragma omp for simd simdlen(8) 
            for(long i0=65536; i0<65536; i0+=1)
            {
                auto tmp0 = in_out_ptr1[i0];
                auto tmp1 = tmp0 * (tmp0>0);
                in_out_ptr1[i0] = tmp1;
            }
        }
    }
}
''')


kernel_cpp_28 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
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
                for(long i1=0; i1<16; i1+=1)
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
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (4*i1) + (16*i2) + (1024*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(64));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(in_out_ptr1 + (4*i1) + (16*i2) + (1024*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_out_ptr1[i2 + (4*i1) + (1024*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr3[i1];
                    auto tmp10 = in_ptr2[i1];
                    auto tmp12 = in_ptr3[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(64);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    in_out_ptr1[i2 + (4*i1) + (1024*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_29 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
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
                for(long i1=0; i1<16; i1+=1)
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
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (4*i1) + (16*i2) + (1024*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(64));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(in_out_ptr1 + (4*i1) + (16*i2) + (1024*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_out_ptr1[i2 + (4*i1) + (1024*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr3[i1];
                    auto tmp10 = in_ptr2[i1];
                    auto tmp12 = in_ptr3[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(64);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    in_out_ptr1[i2 + (4*i1) + (1024*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_30 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<1024; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    for(long i1=0; i1<16; i1+=1)
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
        #pragma omp single
        {
            {
                for(long i0=0; i0<64; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=1024; i0<1024; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(64);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<1024; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp4 = 0;
                    auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<16; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<1024; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(64));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                        tmp16.store(in_out_ptr1 + (4*i1) + (16*i2) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i1) + (4096*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr3[i1];
                        auto tmp10 = in_ptr2[i1];
                        auto tmp12 = in_ptr3[i1];
                        auto tmp14 = in_out_ptr1[i2 + (4*i1) + (4096*i0)];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(64);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp15 * (tmp15>0);
                        in_out_ptr1[i2 + (4*i1) + (4096*i0)] = tmp16;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_31 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
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
                for(long i1=0; i1<16; i1+=1)
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
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (4*i1) + (16*i2) + (1024*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(64));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(in_out_ptr1 + (4*i1) + (16*i2) + (1024*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_out_ptr1[i2 + (4*i1) + (1024*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr3[i1];
                    auto tmp10 = in_ptr2[i1];
                    auto tmp12 = in_ptr3[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(64);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    in_out_ptr1[i2 + (4*i1) + (1024*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_32 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
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
                for(long i1=0; i1<16; i1+=1)
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
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (4*i1) + (16*i2) + (1024*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(64));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(in_out_ptr1 + (4*i1) + (16*i2) + (1024*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_out_ptr1[i2 + (4*i1) + (1024*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr3[i1];
                    auto tmp10 = in_ptr2[i1];
                    auto tmp12 = in_ptr3[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(64);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    in_out_ptr1[i2 + (4*i1) + (1024*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_33 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<1024; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    for(long i1=0; i1<16; i1+=1)
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
        #pragma omp single
        {
            {
                for(long i0=0; i0<64; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=1024; i0<1024; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(64);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<1024; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp4 = 0;
                    auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<16; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<1024; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(64));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                        tmp16.store(in_out_ptr1 + (4*i1) + (16*i2) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i1) + (4096*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr3[i1];
                        auto tmp10 = in_ptr2[i1];
                        auto tmp12 = in_ptr3[i1];
                        auto tmp14 = in_out_ptr1[i2 + (4*i1) + (4096*i0)];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(64);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp15 * (tmp15>0);
                        in_out_ptr1[i2 + (4*i1) + (4096*i0)] = tmp16;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_34 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
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
                for(long i1=0; i1<16; i1+=1)
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
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (4*i1) + (16*i2) + (1024*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(64));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(in_out_ptr1 + (4*i1) + (16*i2) + (1024*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_out_ptr1[i2 + (4*i1) + (1024*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr3[i1];
                    auto tmp10 = in_ptr2[i1];
                    auto tmp12 = in_ptr3[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(64);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    in_out_ptr1[i2 + (4*i1) + (1024*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_35 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
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
                for(long i1=0; i1<16; i1+=1)
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
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (4*i1) + (16*i2) + (1024*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(64));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(in_out_ptr1 + (4*i1) + (16*i2) + (1024*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_out_ptr1[i2 + (4*i1) + (1024*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr3[i1];
                    auto tmp10 = in_ptr2[i1];
                    auto tmp12 = in_ptr3[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(64);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    in_out_ptr1[i2 + (4*i1) + (1024*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_36 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<1024; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    for(long i1=0; i1<16; i1+=1)
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
        #pragma omp single
        {
            {
                for(long i0=0; i0<64; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=1024; i0<1024; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(64);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<1024; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp4 = 0;
                    auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<16; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<1024; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(64));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                        tmp16.store(in_out_ptr1 + (4*i1) + (16*i2) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i1) + (4096*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr3[i1];
                        auto tmp10 = in_ptr2[i1];
                        auto tmp12 = in_ptr3[i1];
                        auto tmp14 = in_out_ptr1[i2 + (4*i1) + (4096*i0)];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(64);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp15 * (tmp15>0);
                        in_out_ptr1[i2 + (4*i1) + (4096*i0)] = tmp16;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_37 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
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
                for(long i1=0; i1<16; i1+=1)
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
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (4*i1) + (16*i2) + (1024*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(64));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(in_out_ptr1 + (4*i1) + (16*i2) + (1024*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_out_ptr1[i2 + (4*i1) + (1024*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr3[i1];
                    auto tmp10 = in_ptr2[i1];
                    auto tmp12 = in_ptr3[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(64);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    in_out_ptr1[i2 + (4*i1) + (1024*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_38 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
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
                for(long i1=0; i1<16; i1+=1)
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
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (4*i1) + (16*i2) + (1024*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(64));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(in_out_ptr1 + (4*i1) + (16*i2) + (1024*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_out_ptr1[i2 + (4*i1) + (1024*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr3[i1];
                    auto tmp10 = in_ptr2[i1];
                    auto tmp12 = in_ptr3[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(64);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    in_out_ptr1[i2 + (4*i1) + (1024*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_39 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<1024; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    for(long i1=0; i1<16; i1+=1)
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
        #pragma omp single
        {
            {
                for(long i0=0; i0<64; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=1024; i0<1024; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(64);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<1024; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp4 = 0;
                    auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<16; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<1024; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(64));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                        tmp16.store(in_out_ptr1 + (4*i1) + (16*i2) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i1) + (4096*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr3[i1];
                        auto tmp10 = in_ptr2[i1];
                        auto tmp12 = in_ptr3[i1];
                        auto tmp14 = in_out_ptr1[i2 + (4*i1) + (4096*i0)];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(64);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp15 * (tmp15>0);
                        in_out_ptr1[i2 + (4*i1) + (4096*i0)] = tmp16;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_40 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
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
                for(long i1=0; i1<16; i1+=1)
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
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (4*i1) + (16*i2) + (1024*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(64));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(in_out_ptr1 + (4*i1) + (16*i2) + (1024*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_out_ptr1[i2 + (4*i1) + (1024*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr3[i1];
                    auto tmp10 = in_ptr2[i1];
                    auto tmp12 = in_ptr3[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(64);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    in_out_ptr1[i2 + (4*i1) + (1024*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_41 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
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
                for(long i1=0; i1<16; i1+=1)
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
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (4*i1) + (16*i2) + (1024*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(64));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(in_out_ptr1 + (4*i1) + (16*i2) + (1024*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_out_ptr1[i2 + (4*i1) + (1024*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr3[i1];
                    auto tmp10 = in_ptr2[i1];
                    auto tmp12 = in_ptr3[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(64);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    in_out_ptr1[i2 + (4*i1) + (1024*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_42 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<1024; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    for(long i1=0; i1<16; i1+=1)
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
        #pragma omp single
        {
            {
                for(long i0=0; i0<64; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    tmp2.store(in_out_ptr0 + 16*i0);
                    tmp8.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=1024; i0<1024; i0+=1)
                {
                    auto tmp0 = out_ptr0[i0];
                    auto tmp5 = in_ptr1[i0];
                    auto tmp1 = static_cast<float>(64);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(0.1);
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = static_cast<float>(0.9);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp4 + tmp7;
                    in_out_ptr0[i0] = tmp2;
                    out_ptr2[i0] = tmp8;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<1024; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp4 = 0;
                    auto tmp4_vec = at::vec::Vectorized<float>(tmp4);
                    for(long i1=0; i1<16; i1+=1)
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
                    out_ptr3[i0] = tmp4;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<16; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<1024; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                        auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                        auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(64));
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp7.rsqrt();
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                        tmp16.store(in_out_ptr1 + (4*i1) + (16*i2) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i1) + (4096*i0)];
                        auto tmp1 = in_out_ptr0[i1];
                        auto tmp3 = out_ptr3[i1];
                        auto tmp10 = in_ptr2[i1];
                        auto tmp12 = in_ptr3[i1];
                        auto tmp14 = in_out_ptr1[i2 + (4*i1) + (4096*i0)];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(64);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = tmp2 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp15 * (tmp15>0);
                        in_out_ptr1[i2 + (4*i1) + (4096*i0)] = tmp16;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_43 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
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
                for(long i1=0; i1<16; i1+=1)
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
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<512; i1+=1)
            {
                for(long i2=0; i2<0; i2+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (4*i1) + (16*i2) + (2048*i0));
                    auto tmp1 = at::vec::Vectorized<float>(in_out_ptr0[i1]);
                    auto tmp3 = at::vec::Vectorized<float>(out_ptr3[i1]);
                    auto tmp10 = at::vec::Vectorized<float>(in_ptr2[i1]);
                    auto tmp12 = at::vec::Vectorized<float>(in_ptr3[i1]);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(64));
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                    tmp14.store(in_out_ptr1 + (4*i1) + (16*i2) + (2048*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i2=0; i2<4; i2+=1)
                {
                    auto tmp0 = in_out_ptr1[i2 + (4*i1) + (2048*i0)];
                    auto tmp1 = in_out_ptr0[i1];
                    auto tmp3 = out_ptr3[i1];
                    auto tmp10 = in_ptr2[i1];
                    auto tmp12 = in_ptr3[i1];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(64);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = tmp2 * tmp8;
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp11 + tmp12;
                    auto tmp14 = tmp13 * (tmp13>0);
                    in_out_ptr1[i2 + (4*i1) + (2048*i0)] = tmp14;
                }
            }
        }
    }
}
''')


kernel_cpp_44 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(16));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(16);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp4 = 0;
                for(long i1=0; i1<16; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    auto tmp1 = in_out_ptr0[i0];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    tmp4 += tmp3;
                }
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i1);
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr3 + 16*i1);
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(16));
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp7.rsqrt();
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                tmp14.store(in_out_ptr1 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = in_out_ptr1[i1 + (512*i0)];
                auto tmp1 = in_out_ptr0[i1];
                auto tmp3 = out_ptr3[i1];
                auto tmp10 = in_ptr2[i1];
                auto tmp12 = in_ptr3[i1];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                in_out_ptr1[i1 + (512*i0)] = tmp14;
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
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(16));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(16);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<2048; i0+=1)
        {
            {
                float tmp4 = 0;
                for(long i1=0; i1<16; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (2048*i1)];
                    auto tmp1 = in_out_ptr0[i0];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    tmp4 += tmp3;
                }
                out_ptr3[i0] = tmp4;
            }
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
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(16));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(16);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<2048; i0+=1)
        {
            {
                float tmp4 = 0;
                for(long i1=0; i1<16; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (2048*i1)];
                    auto tmp1 = in_out_ptr0[i0];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    tmp4 += tmp3;
                }
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            for(long i1=0; i1<128; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i1) + (2048*i0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (2048*i0));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i1);
                auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr3 + 16*i1);
                auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr6 + 16*i1);
                auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr7 + 16*i1);
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(16));
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
                tmp26.store(in_out_ptr1 + (16*i1) + (2048*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=2048; i1<2048; i1+=1)
            {
                auto tmp0 = in_out_ptr1[i1 + (2048*i0)];
                auto tmp1 = in_ptr2[i1];
                auto tmp3 = in_ptr3[i1];
                auto tmp10 = in_ptr4[i1];
                auto tmp12 = in_ptr5[i1];
                auto tmp14 = in_ptr0[i1 + (2048*i0)];
                auto tmp15 = in_out_ptr0[i1];
                auto tmp17 = out_ptr3[i1];
                auto tmp22 = in_ptr6[i1];
                auto tmp24 = in_ptr7[i1];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16);
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
                in_out_ptr1[i1 + (2048*i0)] = tmp26;
            }
        }
    }
    {
        for(long i0=0; i0<2048; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + 16*i0);
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=32768; i0<32768; i0+=1)
        {
            auto tmp0 = in_out_ptr1[i0];
            auto tmp1 = tmp0 * (tmp0>0);
            in_out_ptr1[i0] = tmp1;
        }
    }
}
''')


kernel_cpp_47 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(16));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(16);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp4 = 0;
                for(long i1=0; i1<16; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    auto tmp1 = in_out_ptr0[i0];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    tmp4 += tmp3;
                }
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i1);
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr3 + 16*i1);
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(16));
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp7.rsqrt();
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                tmp14.store(in_out_ptr1 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = in_out_ptr1[i1 + (512*i0)];
                auto tmp1 = in_out_ptr0[i1];
                auto tmp3 = out_ptr3[i1];
                auto tmp10 = in_ptr2[i1];
                auto tmp12 = in_ptr3[i1];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                in_out_ptr1[i1 + (512*i0)] = tmp14;
            }
        }
    }
}
''')


kernel_cpp_48 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(16));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(16);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp4 = 0;
                for(long i1=0; i1<16; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    auto tmp1 = in_out_ptr0[i0];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    tmp4 += tmp3;
                }
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i1);
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr3 + 16*i1);
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(16));
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp7.rsqrt();
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                tmp14.store(in_out_ptr1 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = in_out_ptr1[i1 + (512*i0)];
                auto tmp1 = in_out_ptr0[i1];
                auto tmp3 = out_ptr3[i1];
                auto tmp10 = in_ptr2[i1];
                auto tmp12 = in_ptr3[i1];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                in_out_ptr1[i1 + (512*i0)] = tmp14;
            }
        }
    }
}
''')


kernel_cpp_49 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
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
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(16));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(16);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<2048; i0+=1)
        {
            {
                float tmp4 = 0;
                for(long i1=0; i1<16; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (2048*i1)];
                    auto tmp1 = in_out_ptr0[i0];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    tmp4 += tmp3;
                }
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            for(long i1=0; i1<128; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (2048*i0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i1);
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr3 + 16*i1);
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i1) + (2048*i0));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(16));
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp7.rsqrt();
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                tmp16.store(in_out_ptr1 + (16*i1) + (2048*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=2048; i1<2048; i1+=1)
            {
                auto tmp0 = in_ptr0[i1 + (2048*i0)];
                auto tmp1 = in_out_ptr0[i1];
                auto tmp3 = out_ptr3[i1];
                auto tmp10 = in_ptr2[i1];
                auto tmp12 = in_ptr3[i1];
                auto tmp14 = in_out_ptr1[i1 + (2048*i0)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = tmp15 * (tmp15>0);
                in_out_ptr1[i1 + (2048*i0)] = tmp16;
            }
        }
    }
}
''')


kernel_cpp_50 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(16));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(16);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp4 = 0;
                for(long i1=0; i1<16; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    auto tmp1 = in_out_ptr0[i0];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    tmp4 += tmp3;
                }
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i1);
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr3 + 16*i1);
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(16));
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp7.rsqrt();
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                tmp14.store(in_out_ptr1 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = in_out_ptr1[i1 + (512*i0)];
                auto tmp1 = in_out_ptr0[i1];
                auto tmp3 = out_ptr3[i1];
                auto tmp10 = in_ptr2[i1];
                auto tmp12 = in_ptr3[i1];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                in_out_ptr1[i1 + (512*i0)] = tmp14;
            }
        }
    }
}
''')


kernel_cpp_51 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(16));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(16);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp4 = 0;
                for(long i1=0; i1<16; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    auto tmp1 = in_out_ptr0[i0];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    tmp4 += tmp3;
                }
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i1);
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr3 + 16*i1);
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(16));
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp7.rsqrt();
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                tmp14.store(in_out_ptr1 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = in_out_ptr1[i1 + (512*i0)];
                auto tmp1 = in_out_ptr0[i1];
                auto tmp3 = out_ptr3[i1];
                auto tmp10 = in_ptr2[i1];
                auto tmp12 = in_ptr3[i1];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp14 = tmp13 * (tmp13>0);
                in_out_ptr1[i1 + (512*i0)] = tmp14;
            }
        }
    }
}
''')


kernel_cpp_52 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
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
                for(long i1=0; i1<16; i1+=1)
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
            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(16));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            tmp2.store(in_out_ptr0 + 16*i0);
            tmp8.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp5 = in_ptr1[i0];
            auto tmp1 = static_cast<float>(16);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(0.1);
            auto tmp4 = tmp2 * tmp3;
            auto tmp6 = static_cast<float>(0.9);
            auto tmp7 = tmp5 * tmp6;
            auto tmp8 = tmp4 + tmp7;
            in_out_ptr0[i0] = tmp2;
            out_ptr2[i0] = tmp8;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<2048; i0+=1)
        {
            {
                float tmp4 = 0;
                for(long i1=0; i1<16; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (2048*i1)];
                    auto tmp1 = in_out_ptr0[i0];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    tmp4 += tmp3;
                }
                out_ptr3[i0] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<16; i0+=1)
        {
            for(long i1=0; i1<128; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (2048*i0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i1);
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr3 + 16*i1);
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + (16*i1) + (2048*i0));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(16));
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp7.rsqrt();
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                auto tmp17 = at::vec::Vectorized<float>(static_cast<float>(1));
                auto tmp18 = tmp16 / tmp17;
                tmp18.store(in_out_ptr1 + (16*i1) + (2048*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=2048; i1<2048; i1+=1)
            {
                auto tmp0 = in_ptr0[i1 + (2048*i0)];
                auto tmp1 = in_out_ptr0[i1];
                auto tmp3 = out_ptr3[i1];
                auto tmp10 = in_ptr2[i1];
                auto tmp12 = in_ptr3[i1];
                auto tmp14 = in_out_ptr1[i1 + (2048*i0)];
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(16);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = tmp2 * tmp8;
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp11 + tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = tmp15 * (tmp15>0);
                auto tmp17 = static_cast<float>(1);
                auto tmp18 = tmp16 / tmp17;
                in_out_ptr1[i1 + (2048*i0)] = tmp18;
            }
        }
    }
}
''')


kernel_cpp_53 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       float* __restrict__ in_out_ptr2,
                       float* __restrict__ in_out_ptr3,
                       float* __restrict__ in_out_ptr4,
                       float* __restrict__ in_out_ptr5,
                       float* __restrict__ in_out_ptr6,
                       float* __restrict__ in_out_ptr7,
                       float* __restrict__ in_out_ptr8,
                       float* __restrict__ in_out_ptr9,
                       float* __restrict__ in_out_ptr10,
                       float* __restrict__ in_out_ptr11,
                       float* __restrict__ in_out_ptr12,
                       float* __restrict__ in_out_ptr13,
                       float* __restrict__ in_out_ptr14,
                       float* __restrict__ in_out_ptr15,
                       float* __restrict__ in_out_ptr16,
                       float* __restrict__ in_out_ptr17,
                       float* __restrict__ in_out_ptr18,
                       float* __restrict__ in_out_ptr19,
                       float* __restrict__ in_out_ptr20,
                       float* __restrict__ in_out_ptr21,
                       float* __restrict__ in_out_ptr22,
                       float* __restrict__ in_out_ptr23,
                       float* __restrict__ in_out_ptr24,
                       float* __restrict__ in_out_ptr25,
                       float* __restrict__ in_out_ptr26,
                       float* __restrict__ in_out_ptr27,
                       float* __restrict__ in_out_ptr28,
                       float* __restrict__ in_out_ptr29,
                       float* __restrict__ in_out_ptr30,
                       float* __restrict__ in_out_ptr31,
                       float* __restrict__ in_out_ptr32,
                       float* __restrict__ in_out_ptr33,
                       float* __restrict__ in_out_ptr34,
                       float* __restrict__ in_out_ptr35,
                       float* __restrict__ in_out_ptr36,
                       float* __restrict__ in_out_ptr37,
                       float* __restrict__ in_out_ptr38,
                       float* __restrict__ in_out_ptr39,
                       float* __restrict__ in_out_ptr40,
                       float* __restrict__ in_out_ptr41,
                       float* __restrict__ in_out_ptr42,
                       float* __restrict__ in_out_ptr43,
                       float* __restrict__ in_out_ptr44,
                       float* __restrict__ in_out_ptr45,
                       float* __restrict__ in_out_ptr46,
                       float* __restrict__ in_out_ptr47,
                       float* __restrict__ in_out_ptr48,
                       float* __restrict__ in_out_ptr49,
                       float* __restrict__ in_out_ptr50,
                       float* __restrict__ in_out_ptr51,
                       float* __restrict__ in_out_ptr52,
                       const float* __restrict__ in_ptr0,
                       const long* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const long* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const long* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       const long* __restrict__ in_ptr7,
                       const float* __restrict__ in_ptr8,
                       const long* __restrict__ in_ptr9,
                       const float* __restrict__ in_ptr10,
                       const long* __restrict__ in_ptr11,
                       const float* __restrict__ in_ptr12,
                       const long* __restrict__ in_ptr13,
                       const float* __restrict__ in_ptr14,
                       const long* __restrict__ in_ptr15,
                       const float* __restrict__ in_ptr16,
                       const long* __restrict__ in_ptr17,
                       const float* __restrict__ in_ptr18,
                       const long* __restrict__ in_ptr19,
                       const float* __restrict__ in_ptr20,
                       const long* __restrict__ in_ptr21,
                       const float* __restrict__ in_ptr22,
                       const long* __restrict__ in_ptr23,
                       const float* __restrict__ in_ptr24,
                       const long* __restrict__ in_ptr25,
                       const float* __restrict__ in_ptr26,
                       const long* __restrict__ in_ptr27,
                       const float* __restrict__ in_ptr28,
                       const long* __restrict__ in_ptr29,
                       const float* __restrict__ in_ptr30,
                       const long* __restrict__ in_ptr31,
                       const float* __restrict__ in_ptr32,
                       const long* __restrict__ in_ptr33,
                       const float* __restrict__ in_ptr34,
                       const long* __restrict__ in_ptr35,
                       const float* __restrict__ in_ptr36,
                       const long* __restrict__ in_ptr37,
                       const float* __restrict__ in_ptr38,
                       const long* __restrict__ in_ptr39,
                       const float* __restrict__ in_ptr40,
                       const long* __restrict__ in_ptr41,
                       const float* __restrict__ in_ptr42,
                       const long* __restrict__ in_ptr43,
                       const float* __restrict__ in_ptr44,
                       const long* __restrict__ in_ptr45,
                       const float* __restrict__ in_ptr46,
                       const long* __restrict__ in_ptr47,
                       const float* __restrict__ in_ptr48,
                       const long* __restrict__ in_ptr49,
                       const float* __restrict__ in_ptr50,
                       const long* __restrict__ in_ptr51,
                       const float* __restrict__ in_ptr52,
                       const long* __restrict__ in_ptr53,
                       const float* __restrict__ in_ptr54,
                       const long* __restrict__ in_ptr55,
                       const float* __restrict__ in_ptr56,
                       const long* __restrict__ in_ptr57,
                       const float* __restrict__ in_ptr58,
                       const long* __restrict__ in_ptr59,
                       const float* __restrict__ in_ptr60,
                       const long* __restrict__ in_ptr61,
                       const float* __restrict__ in_ptr62,
                       const long* __restrict__ in_ptr63,
                       const float* __restrict__ in_ptr64,
                       const long* __restrict__ in_ptr65,
                       const float* __restrict__ in_ptr66,
                       const long* __restrict__ in_ptr67,
                       const float* __restrict__ in_ptr68,
                       const long* __restrict__ in_ptr69,
                       const float* __restrict__ in_ptr70,
                       const long* __restrict__ in_ptr71,
                       const float* __restrict__ in_ptr72,
                       const long* __restrict__ in_ptr73,
                       const float* __restrict__ in_ptr74,
                       const long* __restrict__ in_ptr75,
                       const float* __restrict__ in_ptr76,
                       const long* __restrict__ in_ptr77,
                       const float* __restrict__ in_ptr78,
                       const long* __restrict__ in_ptr79,
                       const float* __restrict__ in_ptr80,
                       const long* __restrict__ in_ptr81,
                       const float* __restrict__ in_ptr82,
                       const long* __restrict__ in_ptr83,
                       const float* __restrict__ in_ptr84,
                       const long* __restrict__ in_ptr85,
                       const float* __restrict__ in_ptr86,
                       const long* __restrict__ in_ptr87,
                       const float* __restrict__ in_ptr88,
                       const long* __restrict__ in_ptr89,
                       const float* __restrict__ in_ptr90,
                       const long* __restrict__ in_ptr91,
                       const float* __restrict__ in_ptr92,
                       const long* __restrict__ in_ptr93,
                       const float* __restrict__ in_ptr94,
                       const long* __restrict__ in_ptr95,
                       const float* __restrict__ in_ptr96,
                       const long* __restrict__ in_ptr97,
                       const float* __restrict__ in_ptr98,
                       const long* __restrict__ in_ptr99,
                       const float* __restrict__ in_ptr100,
                       const long* __restrict__ in_ptr101,
                       const float* __restrict__ in_ptr102,
                       const long* __restrict__ in_ptr103,
                       const float* __restrict__ in_ptr104,
                       const long* __restrict__ in_ptr105,
                       float* __restrict__ out_ptr0,
                       long* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       long* __restrict__ out_ptr5,
                       float* __restrict__ out_ptr6,
                       long* __restrict__ out_ptr8,
                       float* __restrict__ out_ptr9,
                       long* __restrict__ out_ptr11,
                       float* __restrict__ out_ptr12,
                       long* __restrict__ out_ptr14,
                       float* __restrict__ out_ptr15,
                       long* __restrict__ out_ptr17,
                       float* __restrict__ out_ptr18,
                       long* __restrict__ out_ptr20,
                       float* __restrict__ out_ptr21,
                       long* __restrict__ out_ptr23,
                       float* __restrict__ out_ptr24,
                       long* __restrict__ out_ptr26,
                       float* __restrict__ out_ptr27,
                       long* __restrict__ out_ptr29,
                       float* __restrict__ out_ptr30,
                       long* __restrict__ out_ptr32,
                       float* __restrict__ out_ptr33,
                       long* __restrict__ out_ptr35,
                       float* __restrict__ out_ptr36,
                       long* __restrict__ out_ptr38,
                       float* __restrict__ out_ptr39,
                       long* __restrict__ out_ptr41,
                       float* __restrict__ out_ptr42,
                       long* __restrict__ out_ptr44,
                       float* __restrict__ out_ptr45,
                       long* __restrict__ out_ptr47,
                       float* __restrict__ out_ptr48,
                       long* __restrict__ out_ptr50,
                       float* __restrict__ out_ptr51,
                       long* __restrict__ out_ptr53,
                       float* __restrict__ out_ptr54,
                       long* __restrict__ out_ptr56,
                       float* __restrict__ out_ptr57,
                       long* __restrict__ out_ptr59,
                       float* __restrict__ out_ptr60,
                       long* __restrict__ out_ptr62,
                       float* __restrict__ out_ptr63,
                       long* __restrict__ out_ptr65,
                       float* __restrict__ out_ptr66,
                       long* __restrict__ out_ptr68,
                       float* __restrict__ out_ptr69,
                       long* __restrict__ out_ptr71,
                       float* __restrict__ out_ptr72,
                       long* __restrict__ out_ptr74,
                       float* __restrict__ out_ptr75,
                       long* __restrict__ out_ptr77,
                       float* __restrict__ out_ptr78,
                       long* __restrict__ out_ptr80,
                       float* __restrict__ out_ptr81,
                       long* __restrict__ out_ptr83,
                       float* __restrict__ out_ptr84,
                       long* __restrict__ out_ptr86,
                       float* __restrict__ out_ptr87,
                       long* __restrict__ out_ptr89,
                       float* __restrict__ out_ptr90,
                       long* __restrict__ out_ptr92,
                       float* __restrict__ out_ptr93,
                       long* __restrict__ out_ptr95,
                       float* __restrict__ out_ptr96,
                       long* __restrict__ out_ptr98,
                       float* __restrict__ out_ptr99,
                       long* __restrict__ out_ptr101,
                       float* __restrict__ out_ptr102,
                       long* __restrict__ out_ptr104,
                       float* __restrict__ out_ptr105,
                       long* __restrict__ out_ptr107,
                       float* __restrict__ out_ptr108,
                       long* __restrict__ out_ptr110,
                       float* __restrict__ out_ptr111,
                       long* __restrict__ out_ptr113,
                       float* __restrict__ out_ptr114,
                       long* __restrict__ out_ptr116,
                       float* __restrict__ out_ptr117,
                       long* __restrict__ out_ptr119,
                       float* __restrict__ out_ptr120,
                       long* __restrict__ out_ptr122,
                       float* __restrict__ out_ptr123,
                       long* __restrict__ out_ptr125,
                       float* __restrict__ out_ptr126,
                       long* __restrict__ out_ptr128,
                       float* __restrict__ out_ptr129,
                       long* __restrict__ out_ptr131,
                       float* __restrict__ out_ptr132,
                       long* __restrict__ out_ptr134,
                       float* __restrict__ out_ptr135,
                       long* __restrict__ out_ptr137,
                       float* __restrict__ out_ptr138,
                       long* __restrict__ out_ptr140,
                       float* __restrict__ out_ptr141,
                       long* __restrict__ out_ptr143,
                       float* __restrict__ out_ptr144,
                       long* __restrict__ out_ptr146,
                       float* __restrict__ out_ptr147,
                       long* __restrict__ out_ptr149,
                       float* __restrict__ out_ptr150,
                       long* __restrict__ out_ptr152,
                       float* __restrict__ out_ptr153,
                       long* __restrict__ out_ptr155,
                       float* __restrict__ out_ptr156,
                       long* __restrict__ out_ptr158)
{
    {
        for(long i0=0; i0<4; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(4096));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0002442002442002));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=64; i0<64; i0+=1)
        {
            auto tmp0 = in_out_ptr0[i0];
            auto tmp7 = in_ptr0[i0];
            auto tmp1 = static_cast<float>(4096);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0002442002442002);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr0[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr1[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr2[0] = tmp2;
    }
    {
        for(long i0=0; i0<4; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr3 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(1024));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0009775171065494));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr3 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=64; i0<64; i0+=1)
        {
            auto tmp0 = in_out_ptr1[i0];
            auto tmp7 = in_ptr2[i0];
            auto tmp1 = static_cast<float>(1024);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0009775171065494);
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
        auto tmp0 = in_ptr3[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr5[0] = tmp2;
    }
    {
        for(long i0=0; i0<4; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr6 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(1024));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0009775171065494));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr6 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=64; i0<64; i0+=1)
        {
            auto tmp0 = in_out_ptr2[i0];
            auto tmp7 = in_ptr4[i0];
            auto tmp1 = static_cast<float>(1024);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0009775171065494);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr6[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr5[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr8[0] = tmp2;
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr9 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(1024));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0009775171065494));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr9 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = in_out_ptr3[i0];
            auto tmp7 = in_ptr6[i0];
            auto tmp1 = static_cast<float>(1024);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0009775171065494);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr9[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr7[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr11[0] = tmp2;
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr12 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(1024));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0009775171065494));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr12 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = in_out_ptr4[i0];
            auto tmp7 = in_ptr8[i0];
            auto tmp1 = static_cast<float>(1024);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0009775171065494);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr12[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr9[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr14[0] = tmp2;
    }
    {
        for(long i0=0; i0<4; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr15 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(1024));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0009775171065494));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr15 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=64; i0<64; i0+=1)
        {
            auto tmp0 = in_out_ptr5[i0];
            auto tmp7 = in_ptr10[i0];
            auto tmp1 = static_cast<float>(1024);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0009775171065494);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr15[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr11[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr17[0] = tmp2;
    }
    {
        for(long i0=0; i0<4; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr18 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(1024));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0009775171065494));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr18 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=64; i0<64; i0+=1)
        {
            auto tmp0 = in_out_ptr6[i0];
            auto tmp7 = in_ptr12[i0];
            auto tmp1 = static_cast<float>(1024);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0009775171065494);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr18[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr13[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr20[0] = tmp2;
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr21 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(1024));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0009775171065494));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr21 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = in_out_ptr7[i0];
            auto tmp7 = in_ptr14[i0];
            auto tmp1 = static_cast<float>(1024);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0009775171065494);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr21[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr15[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr23[0] = tmp2;
    }
    {
        for(long i0=0; i0<4; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr24 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(1024));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0009775171065494));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr24 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=64; i0<64; i0+=1)
        {
            auto tmp0 = in_out_ptr8[i0];
            auto tmp7 = in_ptr16[i0];
            auto tmp1 = static_cast<float>(1024);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0009775171065494);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr24[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr17[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr26[0] = tmp2;
    }
    {
        for(long i0=0; i0<4; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr27 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(1024));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0009775171065494));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr27 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=64; i0<64; i0+=1)
        {
            auto tmp0 = in_out_ptr9[i0];
            auto tmp7 = in_ptr18[i0];
            auto tmp1 = static_cast<float>(1024);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0009775171065494);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr27[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr19[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr29[0] = tmp2;
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr30 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(1024));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0009775171065494));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr30 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = in_out_ptr10[i0];
            auto tmp7 = in_ptr20[i0];
            auto tmp1 = static_cast<float>(1024);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0009775171065494);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr30[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr21[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr32[0] = tmp2;
    }
    {
        for(long i0=0; i0<8; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr33 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(1024));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0009775171065494));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr33 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = in_out_ptr11[i0];
            auto tmp7 = in_ptr22[i0];
            auto tmp1 = static_cast<float>(1024);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0009775171065494);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr33[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr23[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr35[0] = tmp2;
    }
    {
        for(long i0=0; i0<8; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr36 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.003921568627451));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr36 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = in_out_ptr12[i0];
            auto tmp7 = in_ptr24[i0];
            auto tmp1 = static_cast<float>(256);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.003921568627451);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr36[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr25[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr38[0] = tmp2;
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr39 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.003921568627451));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr39 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = in_out_ptr13[i0];
            auto tmp7 = in_ptr26[i0];
            auto tmp1 = static_cast<float>(256);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.003921568627451);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr39[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr27[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr41[0] = tmp2;
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr42 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.003921568627451));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr42 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = in_out_ptr14[i0];
            auto tmp7 = in_ptr28[i0];
            auto tmp1 = static_cast<float>(256);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.003921568627451);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr42[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr29[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr44[0] = tmp2;
    }
    {
        for(long i0=0; i0<8; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr15 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr45 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.003921568627451));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr45 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = in_out_ptr15[i0];
            auto tmp7 = in_ptr30[i0];
            auto tmp1 = static_cast<float>(256);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.003921568627451);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr45[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr31[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr47[0] = tmp2;
    }
    {
        for(long i0=0; i0<8; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr16 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr48 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.003921568627451));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr48 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = in_out_ptr16[i0];
            auto tmp7 = in_ptr32[i0];
            auto tmp1 = static_cast<float>(256);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.003921568627451);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr48[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr33[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr50[0] = tmp2;
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr17 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr51 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.003921568627451));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr51 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = in_out_ptr17[i0];
            auto tmp7 = in_ptr34[i0];
            auto tmp1 = static_cast<float>(256);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.003921568627451);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr51[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr35[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr53[0] = tmp2;
    }
    {
        for(long i0=0; i0<8; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr18 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr54 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.003921568627451));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr54 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = in_out_ptr18[i0];
            auto tmp7 = in_ptr36[i0];
            auto tmp1 = static_cast<float>(256);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.003921568627451);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr54[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr37[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr56[0] = tmp2;
    }
    {
        for(long i0=0; i0<8; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr19 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr57 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.003921568627451));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr57 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = in_out_ptr19[i0];
            auto tmp7 = in_ptr38[i0];
            auto tmp1 = static_cast<float>(256);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.003921568627451);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr57[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr39[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr59[0] = tmp2;
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr20 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr60 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.003921568627451));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr60 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = in_out_ptr20[i0];
            auto tmp7 = in_ptr40[i0];
            auto tmp1 = static_cast<float>(256);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.003921568627451);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr60[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr41[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr62[0] = tmp2;
    }
    {
        for(long i0=0; i0<8; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr21 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr63 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.003921568627451));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr63 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = in_out_ptr21[i0];
            auto tmp7 = in_ptr42[i0];
            auto tmp1 = static_cast<float>(256);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.003921568627451);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr63[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr43[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr65[0] = tmp2;
    }
    {
        for(long i0=0; i0<8; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr22 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr66 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.003921568627451));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr66 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=128; i0<128; i0+=1)
        {
            auto tmp0 = in_out_ptr22[i0];
            auto tmp7 = in_ptr44[i0];
            auto tmp1 = static_cast<float>(256);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.003921568627451);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr66[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr45[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr68[0] = tmp2;
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr23 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr69 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.003921568627451));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr69 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = in_out_ptr23[i0];
            auto tmp7 = in_ptr46[i0];
            auto tmp1 = static_cast<float>(256);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.003921568627451);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr69[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr47[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr71[0] = tmp2;
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr24 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr72 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(256));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.003921568627451));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr72 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = in_out_ptr24[i0];
            auto tmp7 = in_ptr48[i0];
            auto tmp1 = static_cast<float>(256);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.003921568627451);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr72[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr49[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr74[0] = tmp2;
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr25 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr75 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0158730158730158));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr75 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = in_out_ptr25[i0];
            auto tmp7 = in_ptr50[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0158730158730158);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr75[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr51[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr77[0] = tmp2;
    }
    {
        for(long i0=0; i0<64; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr26 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr78 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0158730158730158));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr78 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=1024; i0<1024; i0+=1)
        {
            auto tmp0 = in_out_ptr26[i0];
            auto tmp7 = in_ptr52[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0158730158730158);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr78[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr53[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr80[0] = tmp2;
    }
    {
        for(long i0=0; i0<64; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr27 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr81 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0158730158730158));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr81 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=1024; i0<1024; i0+=1)
        {
            auto tmp0 = in_out_ptr27[i0];
            auto tmp7 = in_ptr54[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0158730158730158);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr81[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr55[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr83[0] = tmp2;
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr28 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr84 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0158730158730158));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr84 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = in_out_ptr28[i0];
            auto tmp7 = in_ptr56[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0158730158730158);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr84[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr57[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr86[0] = tmp2;
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr29 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr87 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0158730158730158));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr87 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = in_out_ptr29[i0];
            auto tmp7 = in_ptr58[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0158730158730158);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr87[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr59[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr89[0] = tmp2;
    }
    {
        for(long i0=0; i0<64; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr30 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr90 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0158730158730158));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr90 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=1024; i0<1024; i0+=1)
        {
            auto tmp0 = in_out_ptr30[i0];
            auto tmp7 = in_ptr60[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0158730158730158);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr90[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr61[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr92[0] = tmp2;
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr31 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr93 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0158730158730158));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr93 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = in_out_ptr31[i0];
            auto tmp7 = in_ptr62[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0158730158730158);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr93[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr63[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr95[0] = tmp2;
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr32 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr96 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0158730158730158));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr96 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = in_out_ptr32[i0];
            auto tmp7 = in_ptr64[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0158730158730158);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr96[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr65[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr98[0] = tmp2;
    }
    {
        for(long i0=0; i0<64; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr33 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr99 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0158730158730158));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr99 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=1024; i0<1024; i0+=1)
        {
            auto tmp0 = in_out_ptr33[i0];
            auto tmp7 = in_ptr66[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0158730158730158);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr99[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr67[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr101[0] = tmp2;
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr34 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr102 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0158730158730158));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr102 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = in_out_ptr34[i0];
            auto tmp7 = in_ptr68[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0158730158730158);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr102[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr69[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr104[0] = tmp2;
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr35 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr105 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0158730158730158));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr105 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = in_out_ptr35[i0];
            auto tmp7 = in_ptr70[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0158730158730158);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr105[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr71[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr107[0] = tmp2;
    }
    {
        for(long i0=0; i0<64; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr36 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr108 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0158730158730158));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr108 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=1024; i0<1024; i0+=1)
        {
            auto tmp0 = in_out_ptr36[i0];
            auto tmp7 = in_ptr72[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0158730158730158);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr108[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr73[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr110[0] = tmp2;
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr37 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr111 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0158730158730158));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr111 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = in_out_ptr37[i0];
            auto tmp7 = in_ptr74[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0158730158730158);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr111[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr75[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr113[0] = tmp2;
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr38 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr114 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0158730158730158));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr114 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = in_out_ptr38[i0];
            auto tmp7 = in_ptr76[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0158730158730158);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr114[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr77[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr116[0] = tmp2;
    }
    {
        for(long i0=0; i0<64; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr39 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr117 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0158730158730158));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr117 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=1024; i0<1024; i0+=1)
        {
            auto tmp0 = in_out_ptr39[i0];
            auto tmp7 = in_ptr78[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0158730158730158);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr117[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr79[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr119[0] = tmp2;
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr40 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr120 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0158730158730158));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr120 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = in_out_ptr40[i0];
            auto tmp7 = in_ptr80[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0158730158730158);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr120[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr81[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr122[0] = tmp2;
    }
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr41 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr123 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0158730158730158));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr123 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = in_out_ptr41[i0];
            auto tmp7 = in_ptr82[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0158730158730158);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr123[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr83[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr125[0] = tmp2;
    }
    {
        for(long i0=0; i0<64; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr42 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr126 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0158730158730158));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr126 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=1024; i0<1024; i0+=1)
        {
            auto tmp0 = in_out_ptr42[i0];
            auto tmp7 = in_ptr84[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0158730158730158);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr126[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr85[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr128[0] = tmp2;
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr43 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr129 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(64));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0158730158730158));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr129 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = in_out_ptr43[i0];
            auto tmp7 = in_ptr86[i0];
            auto tmp1 = static_cast<float>(64);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0158730158730158);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr129[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr87[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr131[0] = tmp2;
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr44 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr132 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(16));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0666666666666667));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr132 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = in_out_ptr44[i0];
            auto tmp7 = in_ptr88[i0];
            auto tmp1 = static_cast<float>(16);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0666666666666667);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr132[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr89[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr134[0] = tmp2;
    }
    {
        for(long i0=0; i0<128; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr45 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr135 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(16));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0666666666666667));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr135 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = in_out_ptr45[i0];
            auto tmp7 = in_ptr90[i0];
            auto tmp1 = static_cast<float>(16);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0666666666666667);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr135[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr91[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr137[0] = tmp2;
    }
    {
        for(long i0=0; i0<128; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr46 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr138 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(16));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0666666666666667));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr138 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = in_out_ptr46[i0];
            auto tmp7 = in_ptr92[i0];
            auto tmp1 = static_cast<float>(16);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0666666666666667);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr138[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr93[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr140[0] = tmp2;
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr47 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr141 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(16));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0666666666666667));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr141 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = in_out_ptr47[i0];
            auto tmp7 = in_ptr94[i0];
            auto tmp1 = static_cast<float>(16);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0666666666666667);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr141[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr95[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr143[0] = tmp2;
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr48 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr144 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(16));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0666666666666667));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr144 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = in_out_ptr48[i0];
            auto tmp7 = in_ptr96[i0];
            auto tmp1 = static_cast<float>(16);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0666666666666667);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr144[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr97[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr146[0] = tmp2;
    }
    {
        for(long i0=0; i0<128; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr49 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr147 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(16));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0666666666666667));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr147 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = in_out_ptr49[i0];
            auto tmp7 = in_ptr98[i0];
            auto tmp1 = static_cast<float>(16);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0666666666666667);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr147[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr99[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr149[0] = tmp2;
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr50 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr150 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(16));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0666666666666667));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr150 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = in_out_ptr50[i0];
            auto tmp7 = in_ptr100[i0];
            auto tmp1 = static_cast<float>(16);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0666666666666667);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr150[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr101[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr152[0] = tmp2;
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr51 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr153 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(16));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0666666666666667));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr153 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = in_out_ptr51[i0];
            auto tmp7 = in_ptr102[i0];
            auto tmp1 = static_cast<float>(16);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0666666666666667);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr153[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr103[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr155[0] = tmp2;
    }
    {
        for(long i0=0; i0<128; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr52 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr156 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(16));
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1.0666666666666667));
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.1));
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.9));
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            tmp10.store(out_ptr156 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = in_out_ptr52[i0];
            auto tmp7 = in_ptr104[i0];
            auto tmp1 = static_cast<float>(16);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.0666666666666667);
            auto tmp4 = tmp2 * tmp3;
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = tmp4 * tmp5;
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = tmp7 * tmp8;
            auto tmp10 = tmp6 + tmp9;
            out_ptr156[i0] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr105[0];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = tmp0 + tmp1;
        out_ptr158[0] = tmp2;
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1 = args
    args.clear()
    buf0 = aten.convolution(arg320_1, arg0_1, None, (2, 2), (3, 3), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf0, (16, 64, 16, 16), (16384, 256, 16, 1))
    del arg0_1
    del arg320_1
    buf2 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf3 = buf2; del buf2  # reuse
    buf4 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf5 = buf0; del buf0  # reuse
    buf6 = empty_strided((16, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_0(c_void_p(buf3.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(arg161_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg161_1.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()))
    del arg161_1
    del arg1_1
    del arg2_1
    del buf5
    buf8 = aten.convolution(buf6, arg3_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf8, (16, 64, 8, 8), (4096, 64, 8, 1))
    del arg3_1
    buf10 = buf3; del buf3  # reuse
    buf11 = buf10; del buf10  # reuse
    buf12 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf13 = buf8; del buf8  # reuse
    kernel_cpp_1(c_void_p(buf11.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(arg164_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(arg164_1.data_ptr()), c_void_p(buf12.data_ptr()))
    del arg164_1
    del arg4_1
    del arg5_1
    buf14 = aten.convolution(buf13, arg6_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf14, (16, 64, 8, 8), (4096, 64, 8, 1))
    del arg6_1
    del buf13
    buf16 = buf11; del buf11  # reuse
    buf17 = buf16; del buf16  # reuse
    buf18 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf19 = buf14; del buf14  # reuse
    kernel_cpp_2(c_void_p(buf17.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(arg167_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(arg167_1.data_ptr()), c_void_p(buf18.data_ptr()))
    del arg167_1
    del arg7_1
    del arg8_1
    buf20 = aten.convolution(buf19, arg9_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf20, (16, 256, 8, 8), (16384, 64, 8, 1))
    del arg9_1
    del buf19
    buf22 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf23 = buf22; del buf22  # reuse
    buf24 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    kernel_cpp_3(c_void_p(buf23.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(arg170_1.data_ptr()), c_void_p(arg170_1.data_ptr()), c_void_p(buf24.data_ptr()))
    del arg170_1
    buf25 = aten.convolution(buf6, arg12_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf25, (16, 256, 8, 8), (16384, 64, 8, 1))
    del arg12_1
    del buf6
    buf27 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf28 = buf27; del buf27  # reuse
    buf29 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf30 = buf20; del buf20  # reuse
    buf31 = buf30; del buf30  # reuse
    kernel_cpp_4(c_void_p(buf28.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(arg173_1.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg173_1.data_ptr()), c_void_p(buf29.data_ptr()))
    del arg10_1
    del arg11_1
    del arg13_1
    del arg14_1
    del arg173_1
    del buf25
    buf32 = aten.convolution(buf31, arg15_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf32, (16, 64, 8, 8), (4096, 64, 8, 1))
    del arg15_1
    buf34 = buf17; del buf17  # reuse
    buf35 = buf34; del buf34  # reuse
    buf36 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf37 = buf32; del buf32  # reuse
    kernel_cpp_5(c_void_p(buf35.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(arg176_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(arg176_1.data_ptr()), c_void_p(buf36.data_ptr()))
    del arg16_1
    del arg176_1
    del arg17_1
    buf38 = aten.convolution(buf37, arg18_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf38, (16, 64, 8, 8), (4096, 64, 8, 1))
    del arg18_1
    del buf37
    buf40 = buf35; del buf35  # reuse
    buf41 = buf40; del buf40  # reuse
    buf42 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf43 = buf38; del buf38  # reuse
    kernel_cpp_6(c_void_p(buf41.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(arg179_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg179_1.data_ptr()), c_void_p(buf42.data_ptr()))
    del arg179_1
    del arg19_1
    del arg20_1
    buf44 = aten.convolution(buf43, arg21_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf44, (16, 256, 8, 8), (16384, 64, 8, 1))
    del arg21_1
    del buf43
    buf46 = buf28; del buf28  # reuse
    buf47 = buf46; del buf46  # reuse
    buf48 = buf23; del buf23  # reuse
    buf49 = buf31; del buf31  # reuse
    kernel_cpp_7(c_void_p(buf47.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(arg182_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(arg182_1.data_ptr()), c_void_p(buf48.data_ptr()))
    del arg182_1
    del arg22_1
    del arg23_1
    del buf44
    buf50 = aten.convolution(buf49, arg24_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf50, (16, 64, 8, 8), (4096, 64, 8, 1))
    del arg24_1
    buf52 = buf41; del buf41  # reuse
    buf53 = buf52; del buf52  # reuse
    buf54 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf55 = buf50; del buf50  # reuse
    kernel_cpp_8(c_void_p(buf53.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(arg185_1.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg185_1.data_ptr()), c_void_p(buf54.data_ptr()))
    del arg185_1
    del arg25_1
    del arg26_1
    buf56 = aten.convolution(buf55, arg27_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf56, (16, 64, 8, 8), (4096, 64, 8, 1))
    del arg27_1
    del buf55
    buf58 = buf53; del buf53  # reuse
    buf59 = buf58; del buf58  # reuse
    buf60 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf61 = buf56; del buf56  # reuse
    kernel_cpp_9(c_void_p(buf59.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(arg188_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(arg188_1.data_ptr()), c_void_p(buf60.data_ptr()))
    del arg188_1
    del arg28_1
    del arg29_1
    del buf59
    buf62 = aten.convolution(buf61, arg30_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf62, (16, 256, 8, 8), (16384, 64, 8, 1))
    del arg30_1
    del buf61
    buf64 = buf47; del buf47  # reuse
    buf65 = buf64; del buf64  # reuse
    buf66 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf67 = buf49; del buf49  # reuse
    kernel_cpp_10(c_void_p(buf65.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(arg191_1.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg191_1.data_ptr()), c_void_p(buf66.data_ptr()))
    del arg191_1
    del arg31_1
    del arg32_1
    del buf62
    buf68 = aten.convolution(buf67, arg33_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf68, (16, 128, 8, 8), (8192, 64, 8, 1))
    del arg33_1
    buf70 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf71 = buf70; del buf70  # reuse
    buf72 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf73 = buf68; del buf68  # reuse
    kernel_cpp_11(c_void_p(buf71.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(arg194_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(arg194_1.data_ptr()), c_void_p(buf72.data_ptr()))
    del arg194_1
    del arg34_1
    del arg35_1
    buf74 = aten.convolution(buf73, arg36_1, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf74, (16, 128, 4, 4), (2048, 16, 4, 1))
    del arg36_1
    del buf73
    buf76 = buf71; del buf71  # reuse
    buf77 = buf76; del buf76  # reuse
    buf78 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf79 = buf74; del buf74  # reuse
    kernel_cpp_12(c_void_p(buf77.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(arg197_1.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg197_1.data_ptr()), c_void_p(buf78.data_ptr()))
    del arg197_1
    del arg37_1
    del arg38_1
    buf80 = aten.convolution(buf79, arg39_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf80, (16, 512, 4, 4), (8192, 16, 4, 1))
    del arg39_1
    del buf79
    buf82 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf83 = buf82; del buf82  # reuse
    buf84 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    kernel_cpp_13(c_void_p(buf83.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(arg200_1.data_ptr()), c_void_p(arg200_1.data_ptr()), c_void_p(buf84.data_ptr()))
    del arg200_1
    buf85 = aten.convolution(buf67, arg42_1, None, (2, 2), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf85, (16, 512, 4, 4), (8192, 16, 4, 1))
    del arg42_1
    del buf67
    buf87 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf88 = buf87; del buf87  # reuse
    buf89 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf90 = buf80; del buf80  # reuse
    buf91 = buf90; del buf90  # reuse
    kernel_cpp_14(c_void_p(buf88.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(arg203_1.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg203_1.data_ptr()), c_void_p(buf89.data_ptr()))
    del arg203_1
    del arg40_1
    del arg41_1
    del arg43_1
    del arg44_1
    del buf85
    buf92 = aten.convolution(buf91, arg45_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf92, (16, 128, 4, 4), (2048, 16, 4, 1))
    del arg45_1
    buf94 = buf77; del buf77  # reuse
    buf95 = buf94; del buf94  # reuse
    buf96 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf97 = buf92; del buf92  # reuse
    kernel_cpp_15(c_void_p(buf95.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(arg206_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(arg206_1.data_ptr()), c_void_p(buf96.data_ptr()))
    del arg206_1
    del arg46_1
    del arg47_1
    buf98 = aten.convolution(buf97, arg48_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf98, (16, 128, 4, 4), (2048, 16, 4, 1))
    del arg48_1
    del buf97
    buf100 = buf95; del buf95  # reuse
    buf101 = buf100; del buf100  # reuse
    buf102 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf103 = buf98; del buf98  # reuse
    kernel_cpp_16(c_void_p(buf101.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(arg209_1.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg209_1.data_ptr()), c_void_p(buf102.data_ptr()))
    del arg209_1
    del arg49_1
    del arg50_1
    buf104 = aten.convolution(buf103, arg51_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf104, (16, 512, 4, 4), (8192, 16, 4, 1))
    del arg51_1
    del buf103
    buf106 = buf88; del buf88  # reuse
    buf107 = buf106; del buf106  # reuse
    buf108 = buf83; del buf83  # reuse
    buf109 = buf104; del buf104  # reuse
    kernel_cpp_17(c_void_p(buf107.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(arg212_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(arg212_1.data_ptr()), c_void_p(buf108.data_ptr()))
    del arg212_1
    del arg52_1
    del arg53_1
    del buf91
    buf110 = aten.convolution(buf109, arg54_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf110, (16, 128, 4, 4), (2048, 16, 4, 1))
    del arg54_1
    buf112 = buf101; del buf101  # reuse
    buf113 = buf112; del buf112  # reuse
    buf114 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf115 = buf110; del buf110  # reuse
    kernel_cpp_18(c_void_p(buf113.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(arg215_1.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg215_1.data_ptr()), c_void_p(buf114.data_ptr()))
    del arg215_1
    del arg55_1
    del arg56_1
    buf116 = aten.convolution(buf115, arg57_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf116, (16, 128, 4, 4), (2048, 16, 4, 1))
    del arg57_1
    del buf115
    buf118 = buf113; del buf113  # reuse
    buf119 = buf118; del buf118  # reuse
    buf120 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf121 = buf116; del buf116  # reuse
    kernel_cpp_19(c_void_p(buf119.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(arg218_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()), c_void_p(arg218_1.data_ptr()), c_void_p(buf120.data_ptr()))
    del arg218_1
    del arg58_1
    del arg59_1
    buf122 = aten.convolution(buf121, arg60_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf122, (16, 512, 4, 4), (8192, 16, 4, 1))
    del arg60_1
    del buf121
    buf124 = buf107; del buf107  # reuse
    buf125 = buf124; del buf124  # reuse
    buf126 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf127 = buf109; del buf109  # reuse
    kernel_cpp_20(c_void_p(buf125.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(arg221_1.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(arg221_1.data_ptr()), c_void_p(buf126.data_ptr()))
    del arg221_1
    del arg61_1
    del arg62_1
    del buf122
    buf128 = aten.convolution(buf127, arg63_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf128, (16, 128, 4, 4), (2048, 16, 4, 1))
    del arg63_1
    buf130 = buf119; del buf119  # reuse
    buf131 = buf130; del buf130  # reuse
    buf132 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf133 = buf128; del buf128  # reuse
    kernel_cpp_21(c_void_p(buf131.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(arg224_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()), c_void_p(arg224_1.data_ptr()), c_void_p(buf132.data_ptr()))
    del arg224_1
    del arg64_1
    del arg65_1
    buf134 = aten.convolution(buf133, arg66_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf134, (16, 128, 4, 4), (2048, 16, 4, 1))
    del arg66_1
    del buf133
    buf136 = buf131; del buf131  # reuse
    buf137 = buf136; del buf136  # reuse
    buf138 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf139 = buf134; del buf134  # reuse
    kernel_cpp_22(c_void_p(buf137.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(arg227_1.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(arg227_1.data_ptr()), c_void_p(buf138.data_ptr()))
    del arg227_1
    del arg67_1
    del arg68_1
    del buf137
    buf140 = aten.convolution(buf139, arg69_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf140, (16, 512, 4, 4), (8192, 16, 4, 1))
    del arg69_1
    del buf139
    buf142 = buf125; del buf125  # reuse
    buf143 = buf142; del buf142  # reuse
    buf144 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf145 = buf127; del buf127  # reuse
    kernel_cpp_23(c_void_p(buf143.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(arg230_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()), c_void_p(arg230_1.data_ptr()), c_void_p(buf144.data_ptr()))
    del arg230_1
    del arg70_1
    del arg71_1
    del buf140
    buf146 = aten.convolution(buf145, arg72_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf146, (16, 256, 4, 4), (4096, 16, 4, 1))
    del arg72_1
    buf148 = buf65; del buf65  # reuse
    buf149 = buf148; del buf148  # reuse
    buf150 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf151 = buf146; del buf146  # reuse
    kernel_cpp_24(c_void_p(buf149.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(arg233_1.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg233_1.data_ptr()), c_void_p(buf150.data_ptr()))
    del arg233_1
    del arg73_1
    del arg74_1
    buf152 = aten.convolution(buf151, arg75_1, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf152, (16, 256, 2, 2), (1024, 4, 2, 1))
    del arg75_1
    del buf151
    buf154 = buf149; del buf149  # reuse
    buf155 = buf154; del buf154  # reuse
    buf156 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf157 = buf152; del buf152  # reuse
    kernel_cpp_25(c_void_p(buf155.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(arg236_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(arg236_1.data_ptr()), c_void_p(buf156.data_ptr()))
    del arg236_1
    del arg76_1
    del arg77_1
    buf158 = aten.convolution(buf157, arg78_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf158, (16, 1024, 2, 2), (4096, 4, 2, 1))
    del arg78_1
    del buf157
    buf160 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf161 = buf160; del buf160  # reuse
    buf162 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    kernel_cpp_26(c_void_p(buf161.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(arg239_1.data_ptr()), c_void_p(arg239_1.data_ptr()), c_void_p(buf162.data_ptr()))
    del arg239_1
    buf163 = aten.convolution(buf145, arg81_1, None, (2, 2), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf163, (16, 1024, 2, 2), (4096, 4, 2, 1))
    del arg81_1
    del buf145
    buf165 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf166 = buf165; del buf165  # reuse
    buf167 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf168 = buf158; del buf158  # reuse
    buf169 = buf168; del buf168  # reuse
    kernel_cpp_27(c_void_p(buf166.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(arg242_1.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(arg242_1.data_ptr()), c_void_p(buf167.data_ptr()))
    del arg242_1
    del arg79_1
    del arg80_1
    del arg82_1
    del arg83_1
    del buf163
    buf170 = aten.convolution(buf169, arg84_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf170, (16, 256, 2, 2), (1024, 4, 2, 1))
    del arg84_1
    buf172 = buf155; del buf155  # reuse
    buf173 = buf172; del buf172  # reuse
    buf174 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf175 = buf170; del buf170  # reuse
    kernel_cpp_28(c_void_p(buf173.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(arg245_1.data_ptr()), c_void_p(arg85_1.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg245_1.data_ptr()), c_void_p(buf174.data_ptr()))
    del arg245_1
    del arg85_1
    del arg86_1
    buf176 = aten.convolution(buf175, arg87_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf176, (16, 256, 2, 2), (1024, 4, 2, 1))
    del arg87_1
    del buf175
    buf178 = buf173; del buf173  # reuse
    buf179 = buf178; del buf178  # reuse
    buf180 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf181 = buf176; del buf176  # reuse
    kernel_cpp_29(c_void_p(buf179.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(arg248_1.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()), c_void_p(arg248_1.data_ptr()), c_void_p(buf180.data_ptr()))
    del arg248_1
    del arg88_1
    del arg89_1
    buf182 = aten.convolution(buf181, arg90_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf182, (16, 1024, 2, 2), (4096, 4, 2, 1))
    del arg90_1
    del buf181
    buf184 = buf166; del buf166  # reuse
    buf185 = buf184; del buf184  # reuse
    buf186 = buf161; del buf161  # reuse
    buf187 = buf169; del buf169  # reuse
    kernel_cpp_30(c_void_p(buf185.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(arg251_1.data_ptr()), c_void_p(arg91_1.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg251_1.data_ptr()), c_void_p(buf186.data_ptr()))
    del arg251_1
    del arg91_1
    del arg92_1
    del buf182
    buf188 = aten.convolution(buf187, arg93_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf188, (16, 256, 2, 2), (1024, 4, 2, 1))
    del arg93_1
    buf190 = buf179; del buf179  # reuse
    buf191 = buf190; del buf190  # reuse
    buf192 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf193 = buf188; del buf188  # reuse
    kernel_cpp_31(c_void_p(buf191.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(arg254_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()), c_void_p(arg254_1.data_ptr()), c_void_p(buf192.data_ptr()))
    del arg254_1
    del arg94_1
    del arg95_1
    buf194 = aten.convolution(buf193, arg96_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf194, (16, 256, 2, 2), (1024, 4, 2, 1))
    del arg96_1
    del buf193
    buf196 = buf191; del buf191  # reuse
    buf197 = buf196; del buf196  # reuse
    buf198 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf199 = buf194; del buf194  # reuse
    kernel_cpp_32(c_void_p(buf197.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(arg97_1.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(buf198.data_ptr()))
    del arg257_1
    del arg97_1
    del arg98_1
    buf200 = aten.convolution(buf199, arg99_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf200, (16, 1024, 2, 2), (4096, 4, 2, 1))
    del arg99_1
    del buf199
    buf202 = buf185; del buf185  # reuse
    buf203 = buf202; del buf202  # reuse
    buf204 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf205 = buf187; del buf187  # reuse
    kernel_cpp_33(c_void_p(buf203.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(arg260_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(arg260_1.data_ptr()), c_void_p(buf204.data_ptr()))
    del arg100_1
    del arg101_1
    del arg260_1
    del buf200
    buf206 = aten.convolution(buf205, arg102_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf206, (16, 256, 2, 2), (1024, 4, 2, 1))
    del arg102_1
    buf208 = buf197; del buf197  # reuse
    buf209 = buf208; del buf208  # reuse
    buf210 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf211 = buf206; del buf206  # reuse
    kernel_cpp_34(c_void_p(buf209.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(arg263_1.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(arg263_1.data_ptr()), c_void_p(buf210.data_ptr()))
    del arg103_1
    del arg104_1
    del arg263_1
    buf212 = aten.convolution(buf211, arg105_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf212, (16, 256, 2, 2), (1024, 4, 2, 1))
    del arg105_1
    del buf211
    buf214 = buf209; del buf209  # reuse
    buf215 = buf214; del buf214  # reuse
    buf216 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf217 = buf212; del buf212  # reuse
    kernel_cpp_35(c_void_p(buf215.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(arg266_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()), c_void_p(arg266_1.data_ptr()), c_void_p(buf216.data_ptr()))
    del arg106_1
    del arg107_1
    del arg266_1
    buf218 = aten.convolution(buf217, arg108_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf218, (16, 1024, 2, 2), (4096, 4, 2, 1))
    del arg108_1
    del buf217
    buf220 = buf203; del buf203  # reuse
    buf221 = buf220; del buf220  # reuse
    buf222 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf223 = buf205; del buf205  # reuse
    kernel_cpp_36(c_void_p(buf221.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(buf222.data_ptr()))
    del arg109_1
    del arg110_1
    del arg269_1
    del buf218
    buf224 = aten.convolution(buf223, arg111_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf224, (16, 256, 2, 2), (1024, 4, 2, 1))
    del arg111_1
    buf226 = buf215; del buf215  # reuse
    buf227 = buf226; del buf226  # reuse
    buf228 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf229 = buf224; del buf224  # reuse
    kernel_cpp_37(c_void_p(buf227.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(arg272_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg113_1.data_ptr()), c_void_p(arg272_1.data_ptr()), c_void_p(buf228.data_ptr()))
    del arg112_1
    del arg113_1
    del arg272_1
    buf230 = aten.convolution(buf229, arg114_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf230, (16, 256, 2, 2), (1024, 4, 2, 1))
    del arg114_1
    del buf229
    buf232 = buf227; del buf227  # reuse
    buf233 = buf232; del buf232  # reuse
    buf234 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf235 = buf230; del buf230  # reuse
    kernel_cpp_38(c_void_p(buf233.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(buf234.data_ptr()))
    del arg115_1
    del arg116_1
    del arg275_1
    buf236 = aten.convolution(buf235, arg117_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf236, (16, 1024, 2, 2), (4096, 4, 2, 1))
    del arg117_1
    del buf235
    buf238 = buf221; del buf221  # reuse
    buf239 = buf238; del buf238  # reuse
    buf240 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf241 = buf223; del buf223  # reuse
    kernel_cpp_39(c_void_p(buf239.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(arg278_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(arg119_1.data_ptr()), c_void_p(arg278_1.data_ptr()), c_void_p(buf240.data_ptr()))
    del arg118_1
    del arg119_1
    del arg278_1
    del buf236
    buf242 = aten.convolution(buf241, arg120_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf242, (16, 256, 2, 2), (1024, 4, 2, 1))
    del arg120_1
    buf244 = buf233; del buf233  # reuse
    buf245 = buf244; del buf244  # reuse
    buf246 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf247 = buf242; del buf242  # reuse
    kernel_cpp_40(c_void_p(buf245.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(arg281_1.data_ptr()), c_void_p(arg121_1.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(arg281_1.data_ptr()), c_void_p(buf246.data_ptr()))
    del arg121_1
    del arg122_1
    del arg281_1
    buf248 = aten.convolution(buf247, arg123_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf248, (16, 256, 2, 2), (1024, 4, 2, 1))
    del arg123_1
    del buf247
    buf250 = buf245; del buf245  # reuse
    buf251 = buf250; del buf250  # reuse
    buf252 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf253 = buf248; del buf248  # reuse
    kernel_cpp_41(c_void_p(buf251.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(arg284_1.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(arg125_1.data_ptr()), c_void_p(arg284_1.data_ptr()), c_void_p(buf252.data_ptr()))
    del arg124_1
    del arg125_1
    del arg284_1
    del buf251
    buf254 = aten.convolution(buf253, arg126_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf254, (16, 1024, 2, 2), (4096, 4, 2, 1))
    del arg126_1
    del buf253
    buf256 = buf239; del buf239  # reuse
    buf257 = buf256; del buf256  # reuse
    buf258 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf259 = buf241; del buf241  # reuse
    kernel_cpp_42(c_void_p(buf257.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(arg127_1.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(buf258.data_ptr()))
    del arg127_1
    del arg128_1
    del arg287_1
    del buf254
    del buf257
    buf260 = aten.convolution(buf259, arg129_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf260, (16, 512, 2, 2), (2048, 4, 2, 1))
    del arg129_1
    buf262 = buf143; del buf143  # reuse
    buf263 = buf262; del buf262  # reuse
    buf264 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf265 = buf260; del buf260  # reuse
    kernel_cpp_43(c_void_p(buf263.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(arg290_1.data_ptr()), c_void_p(arg130_1.data_ptr()), c_void_p(arg131_1.data_ptr()), c_void_p(arg290_1.data_ptr()), c_void_p(buf264.data_ptr()))
    del arg130_1
    del arg131_1
    del arg290_1
    buf266 = aten.convolution(buf265, arg132_1, None, (2, 2), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf266, (16, 512, 1, 1), (512, 1, 1, 1))
    del arg132_1
    del buf265
    buf268 = buf263; del buf263  # reuse
    buf269 = buf268; del buf268  # reuse
    buf270 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf271 = buf266; del buf266  # reuse
    kernel_cpp_44(c_void_p(buf269.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(arg293_1.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(arg293_1.data_ptr()), c_void_p(buf270.data_ptr()))
    del arg133_1
    del arg134_1
    del arg293_1
    buf272 = aten.convolution(buf271, arg135_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf272, (16, 2048, 1, 1), (2048, 1, 1, 1))
    del arg135_1
    del buf271
    buf274 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf275 = buf274; del buf274  # reuse
    buf276 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    kernel_cpp_45(c_void_p(buf275.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(arg296_1.data_ptr()), c_void_p(arg296_1.data_ptr()), c_void_p(buf276.data_ptr()))
    del arg296_1
    buf277 = aten.convolution(buf259, arg138_1, None, (2, 2), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf277, (16, 2048, 1, 1), (2048, 1, 1, 1))
    del arg138_1
    del buf259
    buf279 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf280 = buf279; del buf279  # reuse
    buf281 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf282 = as_strided(buf272, (16, 2048, 1, 1), (2048, 1, 32768, 32768)); del buf272  # reuse
    buf283 = as_strided(buf282, (16, 2048, 1, 1), (2048, 1, 1, 1)); del buf282  # reuse
    kernel_cpp_46(c_void_p(buf280.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(arg299_1.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(arg136_1.data_ptr()), c_void_p(arg137_1.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(arg299_1.data_ptr()), c_void_p(buf281.data_ptr()))
    del arg136_1
    del arg137_1
    del arg139_1
    del arg140_1
    del arg299_1
    del buf277
    buf284 = aten.convolution(buf283, arg141_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf284, (16, 512, 1, 1), (512, 1, 1, 1))
    del arg141_1
    buf286 = buf269; del buf269  # reuse
    buf287 = buf286; del buf286  # reuse
    buf288 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf289 = buf284; del buf284  # reuse
    kernel_cpp_47(c_void_p(buf287.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(arg302_1.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(arg143_1.data_ptr()), c_void_p(arg302_1.data_ptr()), c_void_p(buf288.data_ptr()))
    del arg142_1
    del arg143_1
    del arg302_1
    buf290 = aten.convolution(buf289, arg144_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf290, (16, 512, 1, 1), (512, 1, 1, 1))
    del arg144_1
    del buf289
    buf292 = buf287; del buf287  # reuse
    buf293 = buf292; del buf292  # reuse
    buf294 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf295 = buf290; del buf290  # reuse
    kernel_cpp_48(c_void_p(buf293.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(arg305_1.data_ptr()), c_void_p(arg145_1.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(arg305_1.data_ptr()), c_void_p(buf294.data_ptr()))
    del arg145_1
    del arg146_1
    del arg305_1
    buf296 = aten.convolution(buf295, arg147_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf296, (16, 2048, 1, 1), (2048, 1, 1, 1))
    del arg147_1
    del buf295
    buf298 = buf280; del buf280  # reuse
    buf299 = buf298; del buf298  # reuse
    buf300 = buf275; del buf275  # reuse
    buf301 = buf283; del buf283  # reuse
    kernel_cpp_49(c_void_p(buf299.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(arg308_1.data_ptr()), c_void_p(arg148_1.data_ptr()), c_void_p(arg149_1.data_ptr()), c_void_p(arg308_1.data_ptr()), c_void_p(buf300.data_ptr()))
    del arg148_1
    del arg149_1
    del arg308_1
    del buf296
    buf302 = aten.convolution(buf301, arg150_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf302, (16, 512, 1, 1), (512, 1, 1, 1))
    del arg150_1
    buf304 = buf293; del buf293  # reuse
    buf305 = buf304; del buf304  # reuse
    buf306 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf307 = buf302; del buf302  # reuse
    kernel_cpp_50(c_void_p(buf305.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(arg311_1.data_ptr()), c_void_p(arg151_1.data_ptr()), c_void_p(arg152_1.data_ptr()), c_void_p(arg311_1.data_ptr()), c_void_p(buf306.data_ptr()))
    del arg151_1
    del arg152_1
    del arg311_1
    buf308 = aten.convolution(buf307, arg153_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf308, (16, 512, 1, 1), (512, 1, 1, 1))
    del arg153_1
    del buf307
    buf310 = buf305; del buf305  # reuse
    buf311 = buf310; del buf310  # reuse
    buf312 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf313 = buf308; del buf308  # reuse
    kernel_cpp_51(c_void_p(buf311.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(arg314_1.data_ptr()), c_void_p(arg154_1.data_ptr()), c_void_p(arg155_1.data_ptr()), c_void_p(arg314_1.data_ptr()), c_void_p(buf312.data_ptr()))
    del arg154_1
    del arg155_1
    del arg314_1
    del buf311
    buf314 = aten.convolution(buf313, arg156_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf314, (16, 2048, 1, 1), (2048, 1, 1, 1))
    del arg156_1
    del buf313
    buf316 = buf299; del buf299  # reuse
    buf317 = buf316; del buf316  # reuse
    buf318 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf319 = buf301; del buf301  # reuse
    kernel_cpp_52(c_void_p(buf317.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(arg317_1.data_ptr()), c_void_p(arg157_1.data_ptr()), c_void_p(arg158_1.data_ptr()), c_void_p(arg317_1.data_ptr()), c_void_p(buf318.data_ptr()))
    del arg157_1
    del arg158_1
    del arg317_1
    del buf314
    del buf317
    buf320 = empty_strided((16, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    extern_kernels.addmm(arg160_1, as_strided(buf319, (16, 2048), (2048, 1)), as_strided(arg159_1, (2048, 1000), (1, 2048)), alpha=1, beta=1, out=buf320)
    del arg159_1
    del arg160_1
    del buf319
    buf323 = as_strided(buf4, (64, ), (1, )); del buf4  # reuse
    buf329 = as_strided(buf12, (64, ), (1, )); del buf12  # reuse
    buf335 = as_strided(buf18, (64, ), (1, )); del buf18  # reuse
    buf341 = as_strided(buf24, (256, ), (1, )); del buf24  # reuse
    buf347 = as_strided(buf29, (256, ), (1, )); del buf29  # reuse
    buf353 = as_strided(buf36, (64, ), (1, )); del buf36  # reuse
    buf359 = as_strided(buf42, (64, ), (1, )); del buf42  # reuse
    buf365 = as_strided(buf48, (256, ), (1, )); del buf48  # reuse
    buf371 = as_strided(buf54, (64, ), (1, )); del buf54  # reuse
    buf377 = as_strided(buf60, (64, ), (1, )); del buf60  # reuse
    buf383 = as_strided(buf66, (256, ), (1, )); del buf66  # reuse
    buf389 = as_strided(buf72, (128, ), (1, )); del buf72  # reuse
    buf395 = as_strided(buf78, (128, ), (1, )); del buf78  # reuse
    buf401 = as_strided(buf84, (512, ), (1, )); del buf84  # reuse
    buf407 = as_strided(buf89, (512, ), (1, )); del buf89  # reuse
    buf413 = as_strided(buf96, (128, ), (1, )); del buf96  # reuse
    buf419 = as_strided(buf102, (128, ), (1, )); del buf102  # reuse
    buf425 = as_strided(buf108, (512, ), (1, )); del buf108  # reuse
    buf431 = as_strided(buf114, (128, ), (1, )); del buf114  # reuse
    buf437 = as_strided(buf120, (128, ), (1, )); del buf120  # reuse
    buf443 = as_strided(buf126, (512, ), (1, )); del buf126  # reuse
    buf449 = as_strided(buf132, (128, ), (1, )); del buf132  # reuse
    buf455 = as_strided(buf138, (128, ), (1, )); del buf138  # reuse
    buf461 = as_strided(buf144, (512, ), (1, )); del buf144  # reuse
    buf467 = as_strided(buf150, (256, ), (1, )); del buf150  # reuse
    buf473 = as_strided(buf156, (256, ), (1, )); del buf156  # reuse
    buf479 = as_strided(buf162, (1024, ), (1, )); del buf162  # reuse
    buf485 = as_strided(buf167, (1024, ), (1, )); del buf167  # reuse
    buf491 = as_strided(buf174, (256, ), (1, )); del buf174  # reuse
    buf497 = as_strided(buf180, (256, ), (1, )); del buf180  # reuse
    buf503 = as_strided(buf186, (1024, ), (1, )); del buf186  # reuse
    buf509 = as_strided(buf192, (256, ), (1, )); del buf192  # reuse
    buf515 = as_strided(buf198, (256, ), (1, )); del buf198  # reuse
    buf521 = as_strided(buf204, (1024, ), (1, )); del buf204  # reuse
    buf527 = as_strided(buf210, (256, ), (1, )); del buf210  # reuse
    buf533 = as_strided(buf216, (256, ), (1, )); del buf216  # reuse
    buf539 = as_strided(buf222, (1024, ), (1, )); del buf222  # reuse
    buf545 = as_strided(buf228, (256, ), (1, )); del buf228  # reuse
    buf551 = as_strided(buf234, (256, ), (1, )); del buf234  # reuse
    buf557 = as_strided(buf240, (1024, ), (1, )); del buf240  # reuse
    buf563 = as_strided(buf246, (256, ), (1, )); del buf246  # reuse
    buf569 = as_strided(buf252, (256, ), (1, )); del buf252  # reuse
    buf575 = as_strided(buf258, (1024, ), (1, )); del buf258  # reuse
    buf581 = as_strided(buf264, (512, ), (1, )); del buf264  # reuse
    buf587 = as_strided(buf270, (512, ), (1, )); del buf270  # reuse
    buf593 = as_strided(buf276, (2048, ), (1, )); del buf276  # reuse
    buf599 = as_strided(buf281, (2048, ), (1, )); del buf281  # reuse
    buf605 = as_strided(buf288, (512, ), (1, )); del buf288  # reuse
    buf611 = as_strided(buf294, (512, ), (1, )); del buf294  # reuse
    buf617 = as_strided(buf300, (2048, ), (1, )); del buf300  # reuse
    buf623 = as_strided(buf306, (512, ), (1, )); del buf306  # reuse
    buf629 = as_strided(buf312, (512, ), (1, )); del buf312  # reuse
    buf635 = as_strided(buf318, (2048, ), (1, )); del buf318  # reuse
    kernel_cpp_53(c_void_p(buf323.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(buf551.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf581.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(buf599.data_ptr()), c_void_p(buf605.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(buf617.data_ptr()), c_void_p(buf623.data_ptr()), c_void_p(buf629.data_ptr()), c_void_p(buf635.data_ptr()), c_void_p(arg162_1.data_ptr()), c_void_p(arg163_1.data_ptr()), c_void_p(arg165_1.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(arg168_1.data_ptr()), c_void_p(arg169_1.data_ptr()), c_void_p(arg171_1.data_ptr()), c_void_p(arg172_1.data_ptr()), c_void_p(arg174_1.data_ptr()), c_void_p(arg175_1.data_ptr()), c_void_p(arg177_1.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(arg180_1.data_ptr()), c_void_p(arg181_1.data_ptr()), c_void_p(arg183_1.data_ptr()), c_void_p(arg184_1.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(arg187_1.data_ptr()), c_void_p(arg189_1.data_ptr()), c_void_p(arg190_1.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(arg193_1.data_ptr()), c_void_p(arg195_1.data_ptr()), c_void_p(arg196_1.data_ptr()), c_void_p(arg198_1.data_ptr()), c_void_p(arg199_1.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(arg204_1.data_ptr()), c_void_p(arg205_1.data_ptr()), c_void_p(arg207_1.data_ptr()), c_void_p(arg208_1.data_ptr()), c_void_p(arg210_1.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(arg213_1.data_ptr()), c_void_p(arg214_1.data_ptr()), c_void_p(arg216_1.data_ptr()), c_void_p(arg217_1.data_ptr()), c_void_p(arg219_1.data_ptr()), c_void_p(arg220_1.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(arg223_1.data_ptr()), c_void_p(arg225_1.data_ptr()), c_void_p(arg226_1.data_ptr()), c_void_p(arg228_1.data_ptr()), c_void_p(arg229_1.data_ptr()), c_void_p(arg231_1.data_ptr()), c_void_p(arg232_1.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(arg235_1.data_ptr()), c_void_p(arg237_1.data_ptr()), c_void_p(arg238_1.data_ptr()), c_void_p(arg240_1.data_ptr()), c_void_p(arg241_1.data_ptr()), c_void_p(arg243_1.data_ptr()), c_void_p(arg244_1.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(arg247_1.data_ptr()), c_void_p(arg249_1.data_ptr()), c_void_p(arg250_1.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(arg253_1.data_ptr()), c_void_p(arg255_1.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(arg259_1.data_ptr()), c_void_p(arg261_1.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(arg264_1.data_ptr()), c_void_p(arg265_1.data_ptr()), c_void_p(arg267_1.data_ptr()), c_void_p(arg268_1.data_ptr()), c_void_p(arg270_1.data_ptr()), c_void_p(arg271_1.data_ptr()), c_void_p(arg273_1.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(arg276_1.data_ptr()), c_void_p(arg277_1.data_ptr()), c_void_p(arg279_1.data_ptr()), c_void_p(arg280_1.data_ptr()), c_void_p(arg282_1.data_ptr()), c_void_p(arg283_1.data_ptr()), c_void_p(arg285_1.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(arg288_1.data_ptr()), c_void_p(arg289_1.data_ptr()), c_void_p(arg291_1.data_ptr()), c_void_p(arg292_1.data_ptr()), c_void_p(arg294_1.data_ptr()), c_void_p(arg295_1.data_ptr()), c_void_p(arg297_1.data_ptr()), c_void_p(arg298_1.data_ptr()), c_void_p(arg300_1.data_ptr()), c_void_p(arg301_1.data_ptr()), c_void_p(arg303_1.data_ptr()), c_void_p(arg304_1.data_ptr()), c_void_p(arg306_1.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(arg309_1.data_ptr()), c_void_p(arg310_1.data_ptr()), c_void_p(arg312_1.data_ptr()), c_void_p(arg313_1.data_ptr()), c_void_p(arg315_1.data_ptr()), c_void_p(arg316_1.data_ptr()), c_void_p(arg318_1.data_ptr()), c_void_p(arg319_1.data_ptr()), c_void_p(arg162_1.data_ptr()), c_void_p(arg163_1.data_ptr()), c_void_p(arg165_1.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(arg168_1.data_ptr()), c_void_p(arg169_1.data_ptr()), c_void_p(arg171_1.data_ptr()), c_void_p(arg172_1.data_ptr()), c_void_p(arg174_1.data_ptr()), c_void_p(arg175_1.data_ptr()), c_void_p(arg177_1.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(arg180_1.data_ptr()), c_void_p(arg181_1.data_ptr()), c_void_p(arg183_1.data_ptr()), c_void_p(arg184_1.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(arg187_1.data_ptr()), c_void_p(arg189_1.data_ptr()), c_void_p(arg190_1.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(arg193_1.data_ptr()), c_void_p(arg195_1.data_ptr()), c_void_p(arg196_1.data_ptr()), c_void_p(arg198_1.data_ptr()), c_void_p(arg199_1.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(arg204_1.data_ptr()), c_void_p(arg205_1.data_ptr()), c_void_p(arg207_1.data_ptr()), c_void_p(arg208_1.data_ptr()), c_void_p(arg210_1.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(arg213_1.data_ptr()), c_void_p(arg214_1.data_ptr()), c_void_p(arg216_1.data_ptr()), c_void_p(arg217_1.data_ptr()), c_void_p(arg219_1.data_ptr()), c_void_p(arg220_1.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(arg223_1.data_ptr()), c_void_p(arg225_1.data_ptr()), c_void_p(arg226_1.data_ptr()), c_void_p(arg228_1.data_ptr()), c_void_p(arg229_1.data_ptr()), c_void_p(arg231_1.data_ptr()), c_void_p(arg232_1.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(arg235_1.data_ptr()), c_void_p(arg237_1.data_ptr()), c_void_p(arg238_1.data_ptr()), c_void_p(arg240_1.data_ptr()), c_void_p(arg241_1.data_ptr()), c_void_p(arg243_1.data_ptr()), c_void_p(arg244_1.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(arg247_1.data_ptr()), c_void_p(arg249_1.data_ptr()), c_void_p(arg250_1.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(arg253_1.data_ptr()), c_void_p(arg255_1.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(arg259_1.data_ptr()), c_void_p(arg261_1.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(arg264_1.data_ptr()), c_void_p(arg265_1.data_ptr()), c_void_p(arg267_1.data_ptr()), c_void_p(arg268_1.data_ptr()), c_void_p(arg270_1.data_ptr()), c_void_p(arg271_1.data_ptr()), c_void_p(arg273_1.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(arg276_1.data_ptr()), c_void_p(arg277_1.data_ptr()), c_void_p(arg279_1.data_ptr()), c_void_p(arg280_1.data_ptr()), c_void_p(arg282_1.data_ptr()), c_void_p(arg283_1.data_ptr()), c_void_p(arg285_1.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(arg288_1.data_ptr()), c_void_p(arg289_1.data_ptr()), c_void_p(arg291_1.data_ptr()), c_void_p(arg292_1.data_ptr()), c_void_p(arg294_1.data_ptr()), c_void_p(arg295_1.data_ptr()), c_void_p(arg297_1.data_ptr()), c_void_p(arg298_1.data_ptr()), c_void_p(arg300_1.data_ptr()), c_void_p(arg301_1.data_ptr()), c_void_p(arg303_1.data_ptr()), c_void_p(arg304_1.data_ptr()), c_void_p(arg306_1.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(arg309_1.data_ptr()), c_void_p(arg310_1.data_ptr()), c_void_p(arg312_1.data_ptr()), c_void_p(arg313_1.data_ptr()), c_void_p(arg315_1.data_ptr()), c_void_p(arg316_1.data_ptr()), c_void_p(arg318_1.data_ptr()), c_void_p(arg319_1.data_ptr()))
    del arg162_1
    del arg163_1
    del arg165_1
    del arg166_1
    del arg168_1
    del arg169_1
    del arg171_1
    del arg172_1
    del arg174_1
    del arg175_1
    del arg177_1
    del arg178_1
    del arg180_1
    del arg181_1
    del arg183_1
    del arg184_1
    del arg186_1
    del arg187_1
    del arg189_1
    del arg190_1
    del arg192_1
    del arg193_1
    del arg195_1
    del arg196_1
    del arg198_1
    del arg199_1
    del arg201_1
    del arg202_1
    del arg204_1
    del arg205_1
    del arg207_1
    del arg208_1
    del arg210_1
    del arg211_1
    del arg213_1
    del arg214_1
    del arg216_1
    del arg217_1
    del arg219_1
    del arg220_1
    del arg222_1
    del arg223_1
    del arg225_1
    del arg226_1
    del arg228_1
    del arg229_1
    del arg231_1
    del arg232_1
    del arg234_1
    del arg235_1
    del arg237_1
    del arg238_1
    del arg240_1
    del arg241_1
    del arg243_1
    del arg244_1
    del arg246_1
    del arg247_1
    del arg249_1
    del arg250_1
    del arg252_1
    del arg253_1
    del arg255_1
    del arg256_1
    del arg258_1
    del arg259_1
    del arg261_1
    del arg262_1
    del arg264_1
    del arg265_1
    del arg267_1
    del arg268_1
    del arg270_1
    del arg271_1
    del arg273_1
    del arg274_1
    del arg276_1
    del arg277_1
    del arg279_1
    del arg280_1
    del arg282_1
    del arg283_1
    del arg285_1
    del arg286_1
    del arg288_1
    del arg289_1
    del arg291_1
    del arg292_1
    del arg294_1
    del arg295_1
    del arg297_1
    del arg298_1
    del arg300_1
    del arg301_1
    del arg303_1
    del arg304_1
    del arg306_1
    del arg307_1
    del arg309_1
    del arg310_1
    del arg312_1
    del arg313_1
    del arg315_1
    del arg316_1
    del arg318_1
    del arg319_1
    return (buf320, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((1000, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg164_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg167_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg170_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg173_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg176_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg179_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg182_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg185_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg188_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg191_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg194_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg197_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg200_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg203_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg206_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg209_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg212_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg215_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg218_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg221_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg224_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg227_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg230_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg233_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg236_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg239_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg242_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg245_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg248_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg251_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg254_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg257_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg260_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg263_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg266_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg269_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg272_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg275_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg278_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg281_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg284_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg287_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg290_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg293_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg296_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg299_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg302_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg305_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg308_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg311_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg314_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg317_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg320_1 = rand_strided((16, 3, 32, 32), (3072, 1024, 32, 1), device='cpu', dtype=torch.float32)
    print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1]))
