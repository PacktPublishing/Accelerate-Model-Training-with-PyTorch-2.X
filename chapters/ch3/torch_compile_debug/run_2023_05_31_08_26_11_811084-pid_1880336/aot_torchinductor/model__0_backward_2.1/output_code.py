
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
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const bool* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4)
{
    {
        #pragma GCC ivdep
        for(long i0=0; i0<1000; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<64; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (1000*i1)];
                    tmp1 += tmp0;
                }
                out_ptr0[i0] = tmp1;
            }
        }
    }
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<2048; i0+=1)
            {
                {
                    float tmp6 = 0;
                    float tmp11 = 0;
                    for(long i1=0; i1<64; i1+=1)
                    {
                        auto tmp0 = in_ptr1[i0 + (2048*i1)];
                        auto tmp2 = in_ptr2[i0 + (2048*i1)];
                        auto tmp7 = in_ptr3[i0 + (2048*i1)];
                        auto tmp8 = in_ptr4[i0];
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp3 = static_cast<float>(1);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = tmp0 ? tmp1 : tmp4;
                        auto tmp9 = tmp7 - tmp8;
                        auto tmp10 = tmp5 * tmp9;
                        tmp6 += tmp5;
                        tmp11 += tmp10;
                    }
                    out_ptr1[i0] = tmp6;
                    out_ptr2[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<128; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=2048; i0<2048; i0+=1)
                {
                    auto tmp0 = out_ptr2[i0];
                    auto tmp1 = in_ptr5[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr3[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                for(long i1=0; i1<128; i1+=1)
                {
                    float g_tmp_buffer_in_ptr1[16] = {0};
                    flag_to_float(in_ptr1 + (16*i1) + (2048*i0), g_tmp_buffer_in_ptr1, 16);
                    auto tmp0 = at::vec::Vectorized<float>::loadu(g_tmp_buffer_in_ptr1);
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (2048*i0));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + (16*i1) + (2048*i0));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i1);
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i1);
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + 16*i1);
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i1);
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + 16*i1);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1));
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.015625));
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp8 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    tmp22.store(out_ptr4 + (16*i1) + (2048*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i1=2048; i1<2048; i1+=1)
                {
                    auto tmp0 = in_ptr1[i1 + (2048*i0)];
                    auto tmp2 = in_ptr2[i1 + (2048*i0)];
                    auto tmp6 = in_ptr3[i1 + (2048*i0)];
                    auto tmp7 = in_ptr4[i1];
                    auto tmp9 = out_ptr2[i1];
                    auto tmp12 = in_ptr5[i1];
                    auto tmp17 = out_ptr1[i1];
                    auto tmp20 = in_ptr6[i1];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp3 = static_cast<float>(1);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = tmp0 ? tmp1 : tmp4;
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.015625);
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp8 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    out_ptr4[i1 + (2048*i0)] = tmp22;
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
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp6 = 0;
                float tmp11 = 0;
                for(long i1=0; i1<64; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    auto tmp4 = in_ptr1[i0 + (512*i1)];
                    auto tmp7 = in_ptr2[i0 + (512*i1)];
                    auto tmp8 = in_ptr3[i0];
                    auto tmp1 = static_cast<float>(0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp3 = static_cast<float>(0.0);
                    auto tmp5 = tmp2 ? tmp3 : tmp4;
                    auto tmp9 = tmp7 - tmp8;
                    auto tmp10 = tmp5 * tmp9;
                    tmp6 += tmp5;
                    tmp11 += tmp10;
                }
                out_ptr0[i0] = tmp6;
                out_ptr1[i0] = tmp11;
            }
        }
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr1[i0];
            auto tmp1 = in_ptr4[i0];
            auto tmp2 = tmp0 * tmp1;
            out_ptr2[i0] = tmp2;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<64; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (512*i0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i1) + (512*i0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (512*i0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i1);
                auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i1);
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + 16*i1);
                auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.015625));
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp12 * tmp12;
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp8 * tmp14;
                auto tmp16 = tmp5 - tmp15;
                auto tmp18 = tmp17 * tmp10;
                auto tmp19 = tmp16 - tmp18;
                auto tmp21 = tmp12 * tmp20;
                auto tmp22 = tmp19 * tmp21;
                tmp22.store(in_out_ptr0 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = in_ptr0[i1 + (512*i0)];
                auto tmp4 = in_out_ptr0[i1 + (512*i0)];
                auto tmp6 = in_ptr2[i1 + (512*i0)];
                auto tmp7 = in_ptr3[i1];
                auto tmp9 = out_ptr1[i1];
                auto tmp12 = in_ptr4[i1];
                auto tmp17 = out_ptr0[i1];
                auto tmp20 = in_ptr5[i1];
                auto tmp1 = static_cast<float>(0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = static_cast<float>(0.0);
                auto tmp5 = tmp2 ? tmp3 : tmp4;
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.015625);
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp12 * tmp12;
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp8 * tmp14;
                auto tmp16 = tmp5 - tmp15;
                auto tmp18 = tmp17 * tmp10;
                auto tmp19 = tmp16 - tmp18;
                auto tmp21 = tmp12 * tmp20;
                auto tmp22 = tmp19 * tmp21;
                in_out_ptr0[i1 + (512*i0)] = tmp22;
            }
        }
    }
}
''')


kernel_cpp_2 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp6 = 0;
                float tmp11 = 0;
                for(long i1=0; i1<64; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    auto tmp4 = in_ptr1[i0 + (512*i1)];
                    auto tmp7 = in_ptr2[i0 + (512*i1)];
                    auto tmp8 = in_ptr3[i0];
                    auto tmp1 = static_cast<float>(0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp3 = static_cast<float>(0.0);
                    auto tmp5 = tmp2 ? tmp3 : tmp4;
                    auto tmp9 = tmp7 - tmp8;
                    auto tmp10 = tmp5 * tmp9;
                    tmp6 += tmp5;
                    tmp11 += tmp10;
                }
                out_ptr0[i0] = tmp6;
                out_ptr1[i0] = tmp11;
            }
        }
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr1[i0];
            auto tmp1 = in_ptr4[i0];
            auto tmp2 = tmp0 * tmp1;
            out_ptr2[i0] = tmp2;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<64; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (512*i0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i1) + (512*i0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (512*i0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i1);
                auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i1);
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + 16*i1);
                auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.015625));
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp12 * tmp12;
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp8 * tmp14;
                auto tmp16 = tmp5 - tmp15;
                auto tmp18 = tmp17 * tmp10;
                auto tmp19 = tmp16 - tmp18;
                auto tmp21 = tmp12 * tmp20;
                auto tmp22 = tmp19 * tmp21;
                tmp22.store(in_out_ptr0 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = in_ptr0[i1 + (512*i0)];
                auto tmp4 = in_out_ptr0[i1 + (512*i0)];
                auto tmp6 = in_ptr2[i1 + (512*i0)];
                auto tmp7 = in_ptr3[i1];
                auto tmp9 = out_ptr1[i1];
                auto tmp12 = in_ptr4[i1];
                auto tmp17 = out_ptr0[i1];
                auto tmp20 = in_ptr5[i1];
                auto tmp1 = static_cast<float>(0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = static_cast<float>(0.0);
                auto tmp5 = tmp2 ? tmp3 : tmp4;
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.015625);
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp12 * tmp12;
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp8 * tmp14;
                auto tmp16 = tmp5 - tmp15;
                auto tmp18 = tmp17 * tmp10;
                auto tmp19 = tmp16 - tmp18;
                auto tmp21 = tmp12 * tmp20;
                auto tmp22 = tmp19 * tmp21;
                in_out_ptr0[i1 + (512*i0)] = tmp22;
            }
        }
    }
}
''')


kernel_cpp_3 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       const float* __restrict__ in_ptr0,
                       const bool* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       const float* __restrict__ in_ptr7,
                       float* __restrict__ out_ptr0)
{
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<2048; i0+=1)
            {
                {
                    float tmp12 = 0;
                    float tmp17 = 0;
                    for(long i1=0; i1<64; i1+=1)
                    {
                        auto tmp0 = in_ptr0[i0 + (2048*i1)];
                        auto tmp4 = in_ptr1[i0 + (2048*i1)];
                        auto tmp5 = in_ptr2[i0 + (2048*i1)];
                        auto tmp9 = in_ptr3[i0 + (2048*i1)];
                        auto tmp13 = in_ptr4[i0 + (2048*i1)];
                        auto tmp14 = in_ptr5[i0];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp6 = static_cast<float>(1);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = tmp4 ? tmp3 : tmp7;
                        auto tmp10 = tmp8 + tmp9;
                        auto tmp11 = tmp2 ? tmp3 : tmp10;
                        auto tmp15 = tmp13 - tmp14;
                        auto tmp16 = tmp11 * tmp15;
                        tmp12 += tmp11;
                        tmp17 += tmp16;
                    }
                    out_ptr0[i0] = tmp12;
                    out_ptr1[i0] = tmp17;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                for(long i1=0; i1<128; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (2048*i0));
                    float g_tmp_buffer_in_ptr1[16] = {0};
                    flag_to_float(in_ptr1 + (16*i1) + (2048*i0), g_tmp_buffer_in_ptr1, 16);
                    auto tmp4 = at::vec::Vectorized<float>::loadu(g_tmp_buffer_in_ptr1);
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (2048*i0));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + (16*i1) + (2048*i0));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + (16*i1) + (2048*i0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + 16*i1);
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i1);
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + 16*i1);
                    auto tmp23 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i1);
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + 16*i1);
                    auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                    auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1));
                    auto tmp7 = tmp5 / tmp6;
                    auto tmp8 = decltype(tmp3)::blendv(tmp7, tmp3, tmp4);
                    auto tmp10 = tmp8 + tmp9;
                    auto tmp11 = decltype(tmp3)::blendv(tmp10, tmp3, tmp2);
                    auto tmp14 = tmp12 - tmp13;
                    auto tmp16 = at::vec::Vectorized<float>(static_cast<float>(0.015625));
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp18 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    auto tmp21 = tmp14 * tmp20;
                    auto tmp22 = tmp11 - tmp21;
                    auto tmp24 = tmp23 * tmp16;
                    auto tmp25 = tmp22 - tmp24;
                    auto tmp27 = tmp18 * tmp26;
                    auto tmp28 = tmp25 * tmp27;
                    tmp28.store(in_out_ptr0 + (16*i1) + (2048*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i1=2048; i1<2048; i1+=1)
                {
                    auto tmp0 = in_ptr0[i1 + (2048*i0)];
                    auto tmp4 = in_ptr1[i1 + (2048*i0)];
                    auto tmp5 = in_ptr2[i1 + (2048*i0)];
                    auto tmp9 = in_ptr3[i1 + (2048*i0)];
                    auto tmp12 = in_ptr4[i1 + (2048*i0)];
                    auto tmp13 = in_ptr5[i1];
                    auto tmp15 = out_ptr1[i1];
                    auto tmp18 = in_ptr6[i1];
                    auto tmp23 = out_ptr0[i1];
                    auto tmp26 = in_ptr7[i1];
                    auto tmp1 = static_cast<float>(0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp3 = static_cast<float>(0.0);
                    auto tmp6 = static_cast<float>(1);
                    auto tmp7 = tmp5 / tmp6;
                    auto tmp8 = tmp4 ? tmp3 : tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    auto tmp11 = tmp2 ? tmp3 : tmp10;
                    auto tmp14 = tmp12 - tmp13;
                    auto tmp16 = static_cast<float>(0.015625);
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp18 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    auto tmp21 = tmp14 * tmp20;
                    auto tmp22 = tmp11 - tmp21;
                    auto tmp24 = tmp23 * tmp16;
                    auto tmp25 = tmp22 - tmp24;
                    auto tmp27 = tmp18 * tmp26;
                    auto tmp28 = tmp25 * tmp27;
                    in_out_ptr0[i1 + (2048*i0)] = tmp28;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<128; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=2048; i0<2048; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr6[i0];
                    auto tmp2 = tmp0 * tmp1;
                    in_out_ptr1[i0] = tmp2;
                }
            }
        }
    }
}
''')


kernel_cpp_4 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp6 = 0;
                float tmp11 = 0;
                for(long i1=0; i1<64; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    auto tmp4 = in_ptr1[i0 + (512*i1)];
                    auto tmp7 = in_ptr2[i0 + (512*i1)];
                    auto tmp8 = in_ptr3[i0];
                    auto tmp1 = static_cast<float>(0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp3 = static_cast<float>(0.0);
                    auto tmp5 = tmp2 ? tmp3 : tmp4;
                    auto tmp9 = tmp7 - tmp8;
                    auto tmp10 = tmp5 * tmp9;
                    tmp6 += tmp5;
                    tmp11 += tmp10;
                }
                out_ptr0[i0] = tmp6;
                out_ptr1[i0] = tmp11;
            }
        }
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr1[i0];
            auto tmp1 = in_ptr4[i0];
            auto tmp2 = tmp0 * tmp1;
            out_ptr2[i0] = tmp2;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<64; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (512*i0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i1) + (512*i0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (512*i0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i1);
                auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i1);
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + 16*i1);
                auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.015625));
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp12 * tmp12;
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp8 * tmp14;
                auto tmp16 = tmp5 - tmp15;
                auto tmp18 = tmp17 * tmp10;
                auto tmp19 = tmp16 - tmp18;
                auto tmp21 = tmp12 * tmp20;
                auto tmp22 = tmp19 * tmp21;
                tmp22.store(in_out_ptr0 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = in_ptr0[i1 + (512*i0)];
                auto tmp4 = in_out_ptr0[i1 + (512*i0)];
                auto tmp6 = in_ptr2[i1 + (512*i0)];
                auto tmp7 = in_ptr3[i1];
                auto tmp9 = out_ptr1[i1];
                auto tmp12 = in_ptr4[i1];
                auto tmp17 = out_ptr0[i1];
                auto tmp20 = in_ptr5[i1];
                auto tmp1 = static_cast<float>(0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = static_cast<float>(0.0);
                auto tmp5 = tmp2 ? tmp3 : tmp4;
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.015625);
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp12 * tmp12;
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp8 * tmp14;
                auto tmp16 = tmp5 - tmp15;
                auto tmp18 = tmp17 * tmp10;
                auto tmp19 = tmp16 - tmp18;
                auto tmp21 = tmp12 * tmp20;
                auto tmp22 = tmp19 * tmp21;
                in_out_ptr0[i1 + (512*i0)] = tmp22;
            }
        }
    }
}
''')


kernel_cpp_5 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp6 = 0;
                float tmp11 = 0;
                for(long i1=0; i1<64; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    auto tmp4 = in_ptr1[i0 + (512*i1)];
                    auto tmp7 = in_ptr2[i0 + (512*i1)];
                    auto tmp8 = in_ptr3[i0];
                    auto tmp1 = static_cast<float>(0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp3 = static_cast<float>(0.0);
                    auto tmp5 = tmp2 ? tmp3 : tmp4;
                    auto tmp9 = tmp7 - tmp8;
                    auto tmp10 = tmp5 * tmp9;
                    tmp6 += tmp5;
                    tmp11 += tmp10;
                }
                out_ptr0[i0] = tmp6;
                out_ptr1[i0] = tmp11;
            }
        }
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr1[i0];
            auto tmp1 = in_ptr4[i0];
            auto tmp2 = tmp0 * tmp1;
            out_ptr2[i0] = tmp2;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<64; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (512*i0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i1) + (512*i0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (512*i0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i1);
                auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i1);
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + 16*i1);
                auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.015625));
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp12 * tmp12;
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp8 * tmp14;
                auto tmp16 = tmp5 - tmp15;
                auto tmp18 = tmp17 * tmp10;
                auto tmp19 = tmp16 - tmp18;
                auto tmp21 = tmp12 * tmp20;
                auto tmp22 = tmp19 * tmp21;
                tmp22.store(in_out_ptr0 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = in_ptr0[i1 + (512*i0)];
                auto tmp4 = in_out_ptr0[i1 + (512*i0)];
                auto tmp6 = in_ptr2[i1 + (512*i0)];
                auto tmp7 = in_ptr3[i1];
                auto tmp9 = out_ptr1[i1];
                auto tmp12 = in_ptr4[i1];
                auto tmp17 = out_ptr0[i1];
                auto tmp20 = in_ptr5[i1];
                auto tmp1 = static_cast<float>(0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = static_cast<float>(0.0);
                auto tmp5 = tmp2 ? tmp3 : tmp4;
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.015625);
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp12 * tmp12;
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp8 * tmp14;
                auto tmp16 = tmp5 - tmp15;
                auto tmp18 = tmp17 * tmp10;
                auto tmp19 = tmp16 - tmp18;
                auto tmp21 = tmp12 * tmp20;
                auto tmp22 = tmp19 * tmp21;
                in_out_ptr0[i1 + (512*i0)] = tmp22;
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
                       const bool* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       const float* __restrict__ in_ptr7,
                       const float* __restrict__ in_ptr8,
                       const float* __restrict__ in_ptr9,
                       const float* __restrict__ in_ptr10,
                       const float* __restrict__ in_ptr11,
                       const float* __restrict__ in_ptr12,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<8192; i0+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
                float g_tmp_buffer_in_ptr2[16] = {0};
                flag_to_float(in_ptr2 + 16*i0, g_tmp_buffer_in_ptr2, 16);
                auto tmp6 = at::vec::Vectorized<float>::loadu(g_tmp_buffer_in_ptr2);
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i0);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                auto tmp5 = tmp4 <= tmp1;
                auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(1));
                auto tmp9 = tmp7 / tmp8;
                auto tmp10 = decltype(tmp3)::blendv(tmp9, tmp3, tmp6);
                auto tmp12 = tmp10 + tmp11;
                auto tmp13 = decltype(tmp3)::blendv(tmp12, tmp3, tmp5);
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = decltype(tmp3)::blendv(tmp15, tmp3, tmp2);
                tmp16.store(in_out_ptr0 + 16*i0);
            }
            #pragma omp for simd simdlen(8) 
            for(long i0=131072; i0<131072; i0+=1)
            {
                auto tmp0 = in_ptr0[i0];
                auto tmp4 = in_ptr1[i0];
                auto tmp6 = in_ptr2[i0];
                auto tmp7 = in_out_ptr0[i0];
                auto tmp11 = in_ptr3[i0];
                auto tmp14 = in_ptr4[i0];
                auto tmp1 = static_cast<float>(0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = static_cast<float>(0.0);
                auto tmp5 = tmp4 <= tmp1;
                auto tmp8 = static_cast<float>(1);
                auto tmp9 = tmp7 / tmp8;
                auto tmp10 = tmp6 ? tmp3 : tmp9;
                auto tmp12 = tmp10 + tmp11;
                auto tmp13 = tmp5 ? tmp3 : tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = tmp2 ? tmp3 : tmp15;
                in_out_ptr0[i0] = tmp16;
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<2048; i0+=1)
            {
                {
                    float tmp1 = 0;
                    float tmp6 = 0;
                    float tmp11 = 0;
                    for(long i1=0; i1<64; i1+=1)
                    {
                        auto tmp0 = in_out_ptr0[i0 + (2048*i1)];
                        auto tmp2 = in_ptr5[i0 + (2048*i1)];
                        auto tmp3 = in_ptr6[i0];
                        auto tmp7 = in_ptr7[i0 + (2048*i1)];
                        auto tmp8 = in_ptr8[i0];
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        auto tmp9 = tmp7 - tmp8;
                        auto tmp10 = tmp0 * tmp9;
                        tmp1 += tmp0;
                        tmp6 += tmp5;
                        tmp11 += tmp10;
                    }
                    out_ptr0[i0] = tmp1;
                    out_ptr1[i0] = tmp6;
                    out_ptr2[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<128; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=2048; i0<2048; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr9[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr3[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                for(long i1=0; i1<128; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i1) + (2048*i0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + (16*i1) + (2048*i0));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + 16*i1);
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i1);
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr9 + 16*i1);
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i1);
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr10 + 16*i1);
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr7 + (16*i1) + (2048*i0));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr8 + 16*i1);
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i1);
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr11 + 16*i1);
                    auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr12 + 16*i1);
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.015625));
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp8 = tmp7 * tmp7;
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp10 = tmp3 * tmp9;
                    auto tmp11 = tmp0 - tmp10;
                    auto tmp13 = tmp12 * tmp5;
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp16 = tmp7 * tmp15;
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp20 = tmp18 - tmp19;
                    auto tmp22 = tmp21 * tmp5;
                    auto tmp24 = tmp23 * tmp23;
                    auto tmp25 = tmp22 * tmp24;
                    auto tmp26 = tmp20 * tmp25;
                    auto tmp27 = tmp0 - tmp26;
                    auto tmp28 = tmp27 - tmp13;
                    auto tmp30 = tmp23 * tmp29;
                    auto tmp31 = tmp28 * tmp30;
                    tmp17.store(out_ptr4 + (16*i1) + (2048*i0));
                    tmp31.store(out_ptr5 + (16*i1) + (2048*i0));
                }
                #pragma omp simd simdlen(8) 
                for(long i1=2048; i1<2048; i1+=1)
                {
                    auto tmp0 = in_out_ptr0[i1 + (2048*i0)];
                    auto tmp1 = in_ptr5[i1 + (2048*i0)];
                    auto tmp2 = in_ptr6[i1];
                    auto tmp4 = out_ptr1[i1];
                    auto tmp7 = in_ptr9[i1];
                    auto tmp12 = out_ptr0[i1];
                    auto tmp15 = in_ptr10[i1];
                    auto tmp18 = in_ptr7[i1 + (2048*i0)];
                    auto tmp19 = in_ptr8[i1];
                    auto tmp21 = out_ptr2[i1];
                    auto tmp23 = in_ptr11[i1];
                    auto tmp29 = in_ptr12[i1];
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.015625);
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp8 = tmp7 * tmp7;
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp10 = tmp3 * tmp9;
                    auto tmp11 = tmp0 - tmp10;
                    auto tmp13 = tmp12 * tmp5;
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp16 = tmp7 * tmp15;
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp20 = tmp18 - tmp19;
                    auto tmp22 = tmp21 * tmp5;
                    auto tmp24 = tmp23 * tmp23;
                    auto tmp25 = tmp22 * tmp24;
                    auto tmp26 = tmp20 * tmp25;
                    auto tmp27 = tmp0 - tmp26;
                    auto tmp28 = tmp27 - tmp13;
                    auto tmp30 = tmp23 * tmp29;
                    auto tmp31 = tmp28 * tmp30;
                    out_ptr4[i1 + (2048*i0)] = tmp17;
                    out_ptr5[i1 + (2048*i0)] = tmp31;
                }
            }
        }
    }
}
''')


kernel_cpp_7 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0)
{
    {
        for(long i0=0; i0<128; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = in_out_ptr0[i0];
            auto tmp1 = in_ptr0[i0];
            auto tmp2 = tmp0 * tmp1;
            in_out_ptr0[i0] = tmp2;
        }
    }
}
''')


kernel_cpp_8 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp6 = 0;
                float tmp11 = 0;
                for(long i1=0; i1<64; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    auto tmp4 = in_ptr1[i0 + (512*i1)];
                    auto tmp7 = in_ptr2[i0 + (512*i1)];
                    auto tmp8 = in_ptr3[i0];
                    auto tmp1 = static_cast<float>(0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp3 = static_cast<float>(0.0);
                    auto tmp5 = tmp2 ? tmp3 : tmp4;
                    auto tmp9 = tmp7 - tmp8;
                    auto tmp10 = tmp5 * tmp9;
                    tmp6 += tmp5;
                    tmp11 += tmp10;
                }
                out_ptr0[i0] = tmp6;
                out_ptr1[i0] = tmp11;
            }
        }
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr1[i0];
            auto tmp1 = in_ptr4[i0];
            auto tmp2 = tmp0 * tmp1;
            out_ptr2[i0] = tmp2;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<64; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (512*i0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i1) + (512*i0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (512*i0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i1);
                auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i1);
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + 16*i1);
                auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.015625));
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp12 * tmp12;
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp8 * tmp14;
                auto tmp16 = tmp5 - tmp15;
                auto tmp18 = tmp17 * tmp10;
                auto tmp19 = tmp16 - tmp18;
                auto tmp21 = tmp12 * tmp20;
                auto tmp22 = tmp19 * tmp21;
                tmp22.store(in_out_ptr0 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = in_ptr0[i1 + (512*i0)];
                auto tmp4 = in_out_ptr0[i1 + (512*i0)];
                auto tmp6 = in_ptr2[i1 + (512*i0)];
                auto tmp7 = in_ptr3[i1];
                auto tmp9 = out_ptr1[i1];
                auto tmp12 = in_ptr4[i1];
                auto tmp17 = out_ptr0[i1];
                auto tmp20 = in_ptr5[i1];
                auto tmp1 = static_cast<float>(0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = static_cast<float>(0.0);
                auto tmp5 = tmp2 ? tmp3 : tmp4;
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.015625);
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = tmp12 * tmp12;
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp8 * tmp14;
                auto tmp16 = tmp5 - tmp15;
                auto tmp18 = tmp17 * tmp10;
                auto tmp19 = tmp16 - tmp18;
                auto tmp21 = tmp12 * tmp20;
                auto tmp22 = tmp19 * tmp21;
                in_out_ptr0[i1 + (512*i0)] = tmp22;
            }
        }
    }
}
''')


kernel_cpp_9 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<512; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<0; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (2048*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (4*i0) + (16*i2) + (2048*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i0) + (16*i2) + (2048*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (4*i0) + (2048*i1)];
                            auto tmp4 = in_ptr1[i2 + (4*i0) + (2048*i1)];
                            auto tmp7 = in_ptr2[i2 + (4*i0) + (2048*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<32; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=512; i0<512; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<512; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (2048*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (4*i1) + (16*i2) + (2048*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i1) + (16*i2) + (2048*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.00390625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (4*i1) + (16*i2) + (2048*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i1) + (2048*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (4*i1) + (2048*i0)];
                        auto tmp6 = in_ptr2[i2 + (4*i1) + (2048*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.00390625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (4*i1) + (2048*i0)] = tmp22;
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
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<1024; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp8 = 0;
                    auto tmp8_vec = at::vec::Vectorized<float>(tmp8);
                    float tmp13 = 0;
                    auto tmp13_vec = at::vec::Vectorized<float>(tmp13);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<0; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (4096*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (4*i0) + (16*i2) + (4096*i1));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i0) + (16*i2) + (4096*i1));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + (4*i0) + (16*i2) + (4096*i1));
                            auto tmp10 = at::vec::Vectorized<float>(in_ptr4[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp2);
                            auto tmp11 = tmp9 - tmp10;
                            auto tmp12 = tmp7 * tmp11;
                            tmp8_vec += tmp7;
                            tmp13_vec += tmp12;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp8) reduction(+:tmp13)
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (4*i0) + (4096*i1)];
                            auto tmp4 = in_ptr1[i2 + (4*i0) + (4096*i1)];
                            auto tmp5 = in_ptr2[i2 + (4*i0) + (4096*i1)];
                            auto tmp9 = in_ptr3[i2 + (4*i0) + (4096*i1)];
                            auto tmp10 = in_ptr4[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp7 = tmp2 ? tmp3 : tmp6;
                            auto tmp11 = tmp9 - tmp10;
                            auto tmp12 = tmp7 * tmp11;
                            tmp8 += tmp7;
                            tmp13 += tmp12;
                        }
                    }
                    tmp8 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp8_vec);
                    tmp13 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp13_vec);
                    out_ptr0[i0] = tmp8;
                    out_ptr1[i0] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<1024; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp9 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp11 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp14 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp19 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp22 = at::vec::Vectorized<float>(in_ptr6[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp2);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp12 = at::vec::Vectorized<float>(static_cast<float>(0.00390625));
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp14 * tmp14;
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp10 * tmp16;
                        auto tmp18 = tmp7 - tmp17;
                        auto tmp20 = tmp19 * tmp12;
                        auto tmp21 = tmp18 - tmp20;
                        auto tmp23 = tmp14 * tmp22;
                        auto tmp24 = tmp21 * tmp23;
                        tmp24.store(out_ptr2 + (4*i1) + (16*i2) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i1) + (4096*i0)];
                        auto tmp4 = in_ptr1[i2 + (4*i1) + (4096*i0)];
                        auto tmp5 = in_ptr2[i2 + (4*i1) + (4096*i0)];
                        auto tmp8 = in_ptr3[i2 + (4*i1) + (4096*i0)];
                        auto tmp9 = in_ptr4[i1];
                        auto tmp11 = out_ptr1[i1];
                        auto tmp14 = in_ptr5[i1];
                        auto tmp19 = out_ptr0[i1];
                        auto tmp22 = in_ptr6[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = tmp2 ? tmp3 : tmp6;
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp12 = static_cast<float>(0.00390625);
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp14 * tmp14;
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp10 * tmp16;
                        auto tmp18 = tmp7 - tmp17;
                        auto tmp20 = tmp19 * tmp12;
                        auto tmp21 = tmp18 - tmp20;
                        auto tmp23 = tmp14 * tmp22;
                        auto tmp24 = tmp21 * tmp23;
                        out_ptr2[i2 + (4*i1) + (4096*i0)] = tmp24;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<64; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=1024; i0<1024; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr5[i0];
                    auto tmp2 = tmp0 * tmp1;
                    in_out_ptr0[i0] = tmp2;
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
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<256; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<0; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                            auto tmp4 = in_ptr1[i2 + (4*i0) + (1024*i1)];
                            auto tmp7 = in_ptr2[i2 + (4*i0) + (1024*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<256; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.00390625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i1) + (1024*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (4*i1) + (1024*i0)];
                        auto tmp6 = in_ptr2[i2 + (4*i1) + (1024*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.00390625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (4*i1) + (1024*i0)] = tmp22;
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
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<256; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<0; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                            auto tmp4 = in_ptr1[i2 + (4*i0) + (1024*i1)];
                            auto tmp7 = in_ptr2[i2 + (4*i0) + (1024*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<256; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.00390625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i1) + (1024*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (4*i1) + (1024*i0)];
                        auto tmp6 = in_ptr2[i2 + (4*i1) + (1024*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.00390625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (4*i1) + (1024*i0)] = tmp22;
                    }
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
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       const float* __restrict__ in_ptr7,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<16384; i0+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i0);
                auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                auto tmp5 = tmp4 <= tmp1;
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = decltype(tmp3)::blendv(tmp8, tmp3, tmp5);
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = decltype(tmp3)::blendv(tmp11, tmp3, tmp2);
                tmp12.store(in_out_ptr0 + 16*i0);
            }
            #pragma omp for simd simdlen(8) 
            for(long i0=262144; i0<262144; i0+=1)
            {
                auto tmp0 = in_ptr0[i0];
                auto tmp4 = in_ptr1[i0];
                auto tmp6 = in_out_ptr0[i0];
                auto tmp7 = in_ptr2[i0];
                auto tmp10 = in_ptr3[i0];
                auto tmp1 = static_cast<float>(0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = static_cast<float>(0.0);
                auto tmp5 = tmp4 <= tmp1;
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp5 ? tmp3 : tmp8;
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = tmp2 ? tmp3 : tmp11;
                in_out_ptr0[i0] = tmp12;
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<1024; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<0; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (4*i0) + (16*i2) + (4096*i1));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr4 + (4*i0) + (16*i2) + (4096*i1));
                            auto tmp3 = at::vec::Vectorized<float>(in_ptr5[i0]);
                            auto tmp4 = tmp2 - tmp3;
                            auto tmp5 = tmp0 * tmp4;
                            tmp1_vec += tmp0;
                            tmp6_vec += tmp5;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp1) reduction(+:tmp6)
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = in_out_ptr0[i2 + (4*i0) + (4096*i1)];
                            auto tmp2 = in_ptr4[i2 + (4*i0) + (4096*i1)];
                            auto tmp3 = in_ptr5[i0];
                            auto tmp4 = tmp2 - tmp3;
                            auto tmp5 = tmp0 * tmp4;
                            tmp1 += tmp0;
                            tmp6 += tmp5;
                        }
                    }
                    tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    out_ptr0[i0] = tmp1;
                    out_ptr1[i0] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<64; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=1024; i0<1024; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr6[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<1024; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp2 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp4 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr6[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp15 = at::vec::Vectorized<float>(in_ptr7[i1]);
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.00390625));
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp7 * tmp7;
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = tmp3 * tmp9;
                        auto tmp11 = tmp0 - tmp10;
                        auto tmp13 = tmp12 * tmp5;
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = tmp7 * tmp15;
                        auto tmp17 = tmp14 * tmp16;
                        tmp17.store(out_ptr3 + (4*i1) + (16*i2) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_out_ptr0[i2 + (4*i1) + (4096*i0)];
                        auto tmp1 = in_ptr4[i2 + (4*i1) + (4096*i0)];
                        auto tmp2 = in_ptr5[i1];
                        auto tmp4 = out_ptr1[i1];
                        auto tmp7 = in_ptr6[i1];
                        auto tmp12 = out_ptr0[i1];
                        auto tmp15 = in_ptr7[i1];
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.00390625);
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp7 * tmp7;
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = tmp3 * tmp9;
                        auto tmp11 = tmp0 - tmp10;
                        auto tmp13 = tmp12 * tmp5;
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = tmp7 * tmp15;
                        auto tmp17 = tmp14 * tmp16;
                        out_ptr3[i2 + (4*i1) + (4096*i0)] = tmp17;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_14 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<256; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<0; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                            auto tmp4 = in_ptr1[i2 + (4*i0) + (1024*i1)];
                            auto tmp7 = in_ptr2[i2 + (4*i0) + (1024*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<256; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.00390625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i1) + (1024*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (4*i1) + (1024*i0)];
                        auto tmp6 = in_ptr2[i2 + (4*i1) + (1024*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.00390625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (4*i1) + (1024*i0)] = tmp22;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_15 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<256; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<0; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                            auto tmp4 = in_ptr1[i2 + (4*i0) + (1024*i1)];
                            auto tmp7 = in_ptr2[i2 + (4*i0) + (1024*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<256; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.00390625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i1) + (1024*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (4*i1) + (1024*i0)];
                        auto tmp6 = in_ptr2[i2 + (4*i1) + (1024*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.00390625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (4*i1) + (1024*i0)] = tmp22;
                    }
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
                       const float* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<1024; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp8 = 0;
                    auto tmp8_vec = at::vec::Vectorized<float>(tmp8);
                    float tmp13 = 0;
                    auto tmp13_vec = at::vec::Vectorized<float>(tmp13);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<0; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (4096*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (4*i0) + (16*i2) + (4096*i1));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i0) + (16*i2) + (4096*i1));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + (4*i0) + (16*i2) + (4096*i1));
                            auto tmp10 = at::vec::Vectorized<float>(in_ptr4[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp2);
                            auto tmp11 = tmp9 - tmp10;
                            auto tmp12 = tmp7 * tmp11;
                            tmp8_vec += tmp7;
                            tmp13_vec += tmp12;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp8) reduction(+:tmp13)
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (4*i0) + (4096*i1)];
                            auto tmp4 = in_ptr1[i2 + (4*i0) + (4096*i1)];
                            auto tmp5 = in_ptr2[i2 + (4*i0) + (4096*i1)];
                            auto tmp9 = in_ptr3[i2 + (4*i0) + (4096*i1)];
                            auto tmp10 = in_ptr4[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp7 = tmp2 ? tmp3 : tmp6;
                            auto tmp11 = tmp9 - tmp10;
                            auto tmp12 = tmp7 * tmp11;
                            tmp8 += tmp7;
                            tmp13 += tmp12;
                        }
                    }
                    tmp8 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp8_vec);
                    tmp13 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp13_vec);
                    out_ptr0[i0] = tmp8;
                    out_ptr1[i0] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<1024; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp9 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp11 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp14 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp19 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp22 = at::vec::Vectorized<float>(in_ptr6[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp2);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp12 = at::vec::Vectorized<float>(static_cast<float>(0.00390625));
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp14 * tmp14;
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp10 * tmp16;
                        auto tmp18 = tmp7 - tmp17;
                        auto tmp20 = tmp19 * tmp12;
                        auto tmp21 = tmp18 - tmp20;
                        auto tmp23 = tmp14 * tmp22;
                        auto tmp24 = tmp21 * tmp23;
                        tmp24.store(out_ptr2 + (4*i1) + (16*i2) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i1) + (4096*i0)];
                        auto tmp4 = in_ptr1[i2 + (4*i1) + (4096*i0)];
                        auto tmp5 = in_ptr2[i2 + (4*i1) + (4096*i0)];
                        auto tmp8 = in_ptr3[i2 + (4*i1) + (4096*i0)];
                        auto tmp9 = in_ptr4[i1];
                        auto tmp11 = out_ptr1[i1];
                        auto tmp14 = in_ptr5[i1];
                        auto tmp19 = out_ptr0[i1];
                        auto tmp22 = in_ptr6[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = tmp2 ? tmp3 : tmp6;
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp12 = static_cast<float>(0.00390625);
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp14 * tmp14;
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp10 * tmp16;
                        auto tmp18 = tmp7 - tmp17;
                        auto tmp20 = tmp19 * tmp12;
                        auto tmp21 = tmp18 - tmp20;
                        auto tmp23 = tmp14 * tmp22;
                        auto tmp24 = tmp21 * tmp23;
                        out_ptr2[i2 + (4*i1) + (4096*i0)] = tmp24;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<64; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=1024; i0<1024; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr5[i0];
                    auto tmp2 = tmp0 * tmp1;
                    in_out_ptr0[i0] = tmp2;
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
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<256; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<0; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                            auto tmp4 = in_ptr1[i2 + (4*i0) + (1024*i1)];
                            auto tmp7 = in_ptr2[i2 + (4*i0) + (1024*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<256; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.00390625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i1) + (1024*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (4*i1) + (1024*i0)];
                        auto tmp6 = in_ptr2[i2 + (4*i1) + (1024*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.00390625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (4*i1) + (1024*i0)] = tmp22;
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
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<256; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<0; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                            auto tmp4 = in_ptr1[i2 + (4*i0) + (1024*i1)];
                            auto tmp7 = in_ptr2[i2 + (4*i0) + (1024*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<256; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.00390625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i1) + (1024*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (4*i1) + (1024*i0)];
                        auto tmp6 = in_ptr2[i2 + (4*i1) + (1024*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.00390625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (4*i1) + (1024*i0)] = tmp22;
                    }
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
                       const float* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       const float* __restrict__ in_ptr7,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<16384; i0+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i0);
                auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                auto tmp5 = tmp4 <= tmp1;
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = decltype(tmp3)::blendv(tmp8, tmp3, tmp5);
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = decltype(tmp3)::blendv(tmp11, tmp3, tmp2);
                tmp12.store(in_out_ptr0 + 16*i0);
            }
            #pragma omp for simd simdlen(8) 
            for(long i0=262144; i0<262144; i0+=1)
            {
                auto tmp0 = in_ptr0[i0];
                auto tmp4 = in_ptr1[i0];
                auto tmp6 = in_ptr2[i0];
                auto tmp7 = in_out_ptr0[i0];
                auto tmp10 = in_ptr3[i0];
                auto tmp1 = static_cast<float>(0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = static_cast<float>(0.0);
                auto tmp5 = tmp4 <= tmp1;
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp5 ? tmp3 : tmp8;
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = tmp2 ? tmp3 : tmp11;
                in_out_ptr0[i0] = tmp12;
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<1024; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<0; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (4*i0) + (16*i2) + (4096*i1));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr4 + (4*i0) + (16*i2) + (4096*i1));
                            auto tmp3 = at::vec::Vectorized<float>(in_ptr5[i0]);
                            auto tmp4 = tmp2 - tmp3;
                            auto tmp5 = tmp0 * tmp4;
                            tmp1_vec += tmp0;
                            tmp6_vec += tmp5;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp1) reduction(+:tmp6)
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = in_out_ptr0[i2 + (4*i0) + (4096*i1)];
                            auto tmp2 = in_ptr4[i2 + (4*i0) + (4096*i1)];
                            auto tmp3 = in_ptr5[i0];
                            auto tmp4 = tmp2 - tmp3;
                            auto tmp5 = tmp0 * tmp4;
                            tmp1 += tmp0;
                            tmp6 += tmp5;
                        }
                    }
                    tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    out_ptr0[i0] = tmp1;
                    out_ptr1[i0] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<64; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=1024; i0<1024; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr6[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<1024; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp2 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp4 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr6[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp15 = at::vec::Vectorized<float>(in_ptr7[i1]);
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.00390625));
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp7 * tmp7;
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = tmp3 * tmp9;
                        auto tmp11 = tmp0 - tmp10;
                        auto tmp13 = tmp12 * tmp5;
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = tmp7 * tmp15;
                        auto tmp17 = tmp14 * tmp16;
                        tmp17.store(out_ptr3 + (4*i1) + (16*i2) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_out_ptr0[i2 + (4*i1) + (4096*i0)];
                        auto tmp1 = in_ptr4[i2 + (4*i1) + (4096*i0)];
                        auto tmp2 = in_ptr5[i1];
                        auto tmp4 = out_ptr1[i1];
                        auto tmp7 = in_ptr6[i1];
                        auto tmp12 = out_ptr0[i1];
                        auto tmp15 = in_ptr7[i1];
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.00390625);
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp7 * tmp7;
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = tmp3 * tmp9;
                        auto tmp11 = tmp0 - tmp10;
                        auto tmp13 = tmp12 * tmp5;
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = tmp7 * tmp15;
                        auto tmp17 = tmp14 * tmp16;
                        out_ptr3[i2 + (4*i1) + (4096*i0)] = tmp17;
                    }
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
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<256; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<0; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                            auto tmp4 = in_ptr1[i2 + (4*i0) + (1024*i1)];
                            auto tmp7 = in_ptr2[i2 + (4*i0) + (1024*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<256; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.00390625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i1) + (1024*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (4*i1) + (1024*i0)];
                        auto tmp6 = in_ptr2[i2 + (4*i1) + (1024*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.00390625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (4*i1) + (1024*i0)] = tmp22;
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
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<256; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<0; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                            auto tmp4 = in_ptr1[i2 + (4*i0) + (1024*i1)];
                            auto tmp7 = in_ptr2[i2 + (4*i0) + (1024*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<256; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.00390625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i1) + (1024*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (4*i1) + (1024*i0)];
                        auto tmp6 = in_ptr2[i2 + (4*i1) + (1024*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.00390625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (4*i1) + (1024*i0)] = tmp22;
                    }
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
                       const float* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<1024; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp8 = 0;
                    auto tmp8_vec = at::vec::Vectorized<float>(tmp8);
                    float tmp13 = 0;
                    auto tmp13_vec = at::vec::Vectorized<float>(tmp13);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<0; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (4096*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (4*i0) + (16*i2) + (4096*i1));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i0) + (16*i2) + (4096*i1));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + (4*i0) + (16*i2) + (4096*i1));
                            auto tmp10 = at::vec::Vectorized<float>(in_ptr4[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp2);
                            auto tmp11 = tmp9 - tmp10;
                            auto tmp12 = tmp7 * tmp11;
                            tmp8_vec += tmp7;
                            tmp13_vec += tmp12;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp8) reduction(+:tmp13)
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (4*i0) + (4096*i1)];
                            auto tmp4 = in_ptr1[i2 + (4*i0) + (4096*i1)];
                            auto tmp5 = in_ptr2[i2 + (4*i0) + (4096*i1)];
                            auto tmp9 = in_ptr3[i2 + (4*i0) + (4096*i1)];
                            auto tmp10 = in_ptr4[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp7 = tmp2 ? tmp3 : tmp6;
                            auto tmp11 = tmp9 - tmp10;
                            auto tmp12 = tmp7 * tmp11;
                            tmp8 += tmp7;
                            tmp13 += tmp12;
                        }
                    }
                    tmp8 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp8_vec);
                    tmp13 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp13_vec);
                    out_ptr0[i0] = tmp8;
                    out_ptr1[i0] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<1024; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp9 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp11 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp14 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp19 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp22 = at::vec::Vectorized<float>(in_ptr6[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp2);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp12 = at::vec::Vectorized<float>(static_cast<float>(0.00390625));
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp14 * tmp14;
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp10 * tmp16;
                        auto tmp18 = tmp7 - tmp17;
                        auto tmp20 = tmp19 * tmp12;
                        auto tmp21 = tmp18 - tmp20;
                        auto tmp23 = tmp14 * tmp22;
                        auto tmp24 = tmp21 * tmp23;
                        tmp24.store(out_ptr2 + (4*i1) + (16*i2) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i1) + (4096*i0)];
                        auto tmp4 = in_ptr1[i2 + (4*i1) + (4096*i0)];
                        auto tmp5 = in_ptr2[i2 + (4*i1) + (4096*i0)];
                        auto tmp8 = in_ptr3[i2 + (4*i1) + (4096*i0)];
                        auto tmp9 = in_ptr4[i1];
                        auto tmp11 = out_ptr1[i1];
                        auto tmp14 = in_ptr5[i1];
                        auto tmp19 = out_ptr0[i1];
                        auto tmp22 = in_ptr6[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = tmp2 ? tmp3 : tmp6;
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp12 = static_cast<float>(0.00390625);
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp14 * tmp14;
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp10 * tmp16;
                        auto tmp18 = tmp7 - tmp17;
                        auto tmp20 = tmp19 * tmp12;
                        auto tmp21 = tmp18 - tmp20;
                        auto tmp23 = tmp14 * tmp22;
                        auto tmp24 = tmp21 * tmp23;
                        out_ptr2[i2 + (4*i1) + (4096*i0)] = tmp24;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<64; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=1024; i0<1024; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr5[i0];
                    auto tmp2 = tmp0 * tmp1;
                    in_out_ptr0[i0] = tmp2;
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
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<256; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<0; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                            auto tmp4 = in_ptr1[i2 + (4*i0) + (1024*i1)];
                            auto tmp7 = in_ptr2[i2 + (4*i0) + (1024*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<256; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.00390625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i1) + (1024*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (4*i1) + (1024*i0)];
                        auto tmp6 = in_ptr2[i2 + (4*i1) + (1024*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.00390625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (4*i1) + (1024*i0)] = tmp22;
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
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<256; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<0; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                            auto tmp4 = in_ptr1[i2 + (4*i0) + (1024*i1)];
                            auto tmp7 = in_ptr2[i2 + (4*i0) + (1024*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<256; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.00390625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i1) + (1024*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (4*i1) + (1024*i0)];
                        auto tmp6 = in_ptr2[i2 + (4*i1) + (1024*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.00390625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (4*i1) + (1024*i0)] = tmp22;
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
                       const float* __restrict__ in_ptr10,
                       const float* __restrict__ in_ptr11,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<16384; i0+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i0);
                auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                auto tmp5 = tmp4 <= tmp1;
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = decltype(tmp3)::blendv(tmp8, tmp3, tmp5);
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = decltype(tmp3)::blendv(tmp11, tmp3, tmp2);
                tmp12.store(in_out_ptr0 + 16*i0);
            }
            #pragma omp for simd simdlen(8) 
            for(long i0=262144; i0<262144; i0+=1)
            {
                auto tmp0 = in_ptr0[i0];
                auto tmp4 = in_ptr1[i0];
                auto tmp6 = in_out_ptr0[i0];
                auto tmp7 = in_ptr2[i0];
                auto tmp10 = in_ptr3[i0];
                auto tmp1 = static_cast<float>(0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = static_cast<float>(0.0);
                auto tmp5 = tmp4 <= tmp1;
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp5 ? tmp3 : tmp8;
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = tmp2 ? tmp3 : tmp11;
                in_out_ptr0[i0] = tmp12;
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<1024; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<0; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (4*i0) + (16*i2) + (4096*i1));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr4 + (4*i0) + (16*i2) + (4096*i1));
                            auto tmp3 = at::vec::Vectorized<float>(in_ptr5[i0]);
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + (4*i0) + (16*i2) + (4096*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr7[i0]);
                            auto tmp4 = tmp2 - tmp3;
                            auto tmp5 = tmp0 * tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp0 * tmp9;
                            tmp1_vec += tmp0;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp1) reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = in_out_ptr0[i2 + (4*i0) + (4096*i1)];
                            auto tmp2 = in_ptr4[i2 + (4*i0) + (4096*i1)];
                            auto tmp3 = in_ptr5[i0];
                            auto tmp7 = in_ptr6[i2 + (4*i0) + (4096*i1)];
                            auto tmp8 = in_ptr7[i0];
                            auto tmp4 = tmp2 - tmp3;
                            auto tmp5 = tmp0 * tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp0 * tmp9;
                            tmp1 += tmp0;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp1;
                    out_ptr1[i0] = tmp6;
                    out_ptr2[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<64; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=1024; i0<1024; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr8[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr3[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<1024; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp2 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp4 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr8[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp15 = at::vec::Vectorized<float>(in_ptr9[i1]);
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + (4*i1) + (16*i2) + (4096*i0));
                        auto tmp19 = at::vec::Vectorized<float>(in_ptr7[i1]);
                        auto tmp21 = at::vec::Vectorized<float>(out_ptr2[i1]);
                        auto tmp23 = at::vec::Vectorized<float>(in_ptr10[i1]);
                        auto tmp29 = at::vec::Vectorized<float>(in_ptr11[i1]);
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.00390625));
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp7 * tmp7;
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = tmp3 * tmp9;
                        auto tmp11 = tmp0 - tmp10;
                        auto tmp13 = tmp12 * tmp5;
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = tmp7 * tmp15;
                        auto tmp17 = tmp14 * tmp16;
                        auto tmp20 = tmp18 - tmp19;
                        auto tmp22 = tmp21 * tmp5;
                        auto tmp24 = tmp23 * tmp23;
                        auto tmp25 = tmp22 * tmp24;
                        auto tmp26 = tmp20 * tmp25;
                        auto tmp27 = tmp0 - tmp26;
                        auto tmp28 = tmp27 - tmp13;
                        auto tmp30 = tmp23 * tmp29;
                        auto tmp31 = tmp28 * tmp30;
                        tmp17.store(out_ptr4 + (4*i1) + (16*i2) + (4096*i0));
                        tmp31.store(out_ptr5 + (4*i1) + (16*i2) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_out_ptr0[i2 + (4*i1) + (4096*i0)];
                        auto tmp1 = in_ptr4[i2 + (4*i1) + (4096*i0)];
                        auto tmp2 = in_ptr5[i1];
                        auto tmp4 = out_ptr1[i1];
                        auto tmp7 = in_ptr8[i1];
                        auto tmp12 = out_ptr0[i1];
                        auto tmp15 = in_ptr9[i1];
                        auto tmp18 = in_ptr6[i2 + (4*i1) + (4096*i0)];
                        auto tmp19 = in_ptr7[i1];
                        auto tmp21 = out_ptr2[i1];
                        auto tmp23 = in_ptr10[i1];
                        auto tmp29 = in_ptr11[i1];
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.00390625);
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp7 * tmp7;
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = tmp3 * tmp9;
                        auto tmp11 = tmp0 - tmp10;
                        auto tmp13 = tmp12 * tmp5;
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = tmp7 * tmp15;
                        auto tmp17 = tmp14 * tmp16;
                        auto tmp20 = tmp18 - tmp19;
                        auto tmp22 = tmp21 * tmp5;
                        auto tmp24 = tmp23 * tmp23;
                        auto tmp25 = tmp22 * tmp24;
                        auto tmp26 = tmp20 * tmp25;
                        auto tmp27 = tmp0 - tmp26;
                        auto tmp28 = tmp27 - tmp13;
                        auto tmp30 = tmp23 * tmp29;
                        auto tmp31 = tmp28 * tmp30;
                        out_ptr4[i2 + (4*i1) + (4096*i0)] = tmp17;
                        out_ptr5[i2 + (4*i1) + (4096*i0)] = tmp31;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_26 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0)
{
    {
        for(long i0=0; i0<64; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=1024; i0<1024; i0+=1)
        {
            auto tmp0 = in_out_ptr0[i0];
            auto tmp1 = in_ptr0[i0];
            auto tmp2 = tmp0 * tmp1;
            in_out_ptr0[i0] = tmp2;
        }
    }
}
''')


kernel_cpp_27 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<256; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<0; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i0) + (16*i2) + (1024*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (4*i0) + (1024*i1)];
                            auto tmp4 = in_ptr1[i2 + (4*i0) + (1024*i1)];
                            auto tmp7 = in_ptr2[i2 + (4*i0) + (1024*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<256; i1+=1)
                {
                    for(long i2=0; i2<0; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (4*i1) + (16*i2) + (1024*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.00390625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (4*i1) + (16*i2) + (1024*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (4*i1) + (1024*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (4*i1) + (1024*i0)];
                        auto tmp6 = in_ptr2[i2 + (4*i1) + (1024*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.00390625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (4*i1) + (1024*i0)] = tmp22;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_28 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<256; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<1; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (4096*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i0) + (16*i2) + (4096*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i0) + (16*i2) + (4096*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=16; i2<16; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (16*i0) + (4096*i1)];
                            auto tmp4 = in_ptr1[i2 + (16*i0) + (4096*i1)];
                            auto tmp7 = in_ptr2[i2 + (16*i0) + (4096*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<256; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (4096*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i1) + (16*i2) + (4096*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (16*i2) + (4096*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.0009765625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (16*i1) + (16*i2) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i1) + (4096*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (16*i1) + (4096*i0)];
                        auto tmp6 = in_ptr2[i2 + (16*i1) + (4096*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.0009765625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (16*i1) + (4096*i0)] = tmp22;
                    }
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
                       const float* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<512; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp8 = 0;
                    auto tmp8_vec = at::vec::Vectorized<float>(tmp8);
                    float tmp13 = 0;
                    auto tmp13_vec = at::vec::Vectorized<float>(tmp13);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<1; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (8192*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i0) + (16*i2) + (8192*i1));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i0) + (16*i2) + (8192*i1));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + (16*i0) + (16*i2) + (8192*i1));
                            auto tmp10 = at::vec::Vectorized<float>(in_ptr4[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp2);
                            auto tmp11 = tmp9 - tmp10;
                            auto tmp12 = tmp7 * tmp11;
                            tmp8_vec += tmp7;
                            tmp13_vec += tmp12;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp8) reduction(+:tmp13)
                        for(long i2=16; i2<16; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (16*i0) + (8192*i1)];
                            auto tmp4 = in_ptr1[i2 + (16*i0) + (8192*i1)];
                            auto tmp5 = in_ptr2[i2 + (16*i0) + (8192*i1)];
                            auto tmp9 = in_ptr3[i2 + (16*i0) + (8192*i1)];
                            auto tmp10 = in_ptr4[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp7 = tmp2 ? tmp3 : tmp6;
                            auto tmp11 = tmp9 - tmp10;
                            auto tmp12 = tmp7 * tmp11;
                            tmp8 += tmp7;
                            tmp13 += tmp12;
                        }
                    }
                    tmp8 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp8_vec);
                    tmp13 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp13_vec);
                    out_ptr0[i0] = tmp8;
                    out_ptr1[i0] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<512; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp9 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp11 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp14 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp19 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp22 = at::vec::Vectorized<float>(in_ptr6[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp2);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp12 = at::vec::Vectorized<float>(static_cast<float>(0.0009765625));
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp14 * tmp14;
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp10 * tmp16;
                        auto tmp18 = tmp7 - tmp17;
                        auto tmp20 = tmp19 * tmp12;
                        auto tmp21 = tmp18 - tmp20;
                        auto tmp23 = tmp14 * tmp22;
                        auto tmp24 = tmp21 * tmp23;
                        tmp24.store(out_ptr2 + (16*i1) + (16*i2) + (8192*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i1) + (8192*i0)];
                        auto tmp4 = in_ptr1[i2 + (16*i1) + (8192*i0)];
                        auto tmp5 = in_ptr2[i2 + (16*i1) + (8192*i0)];
                        auto tmp8 = in_ptr3[i2 + (16*i1) + (8192*i0)];
                        auto tmp9 = in_ptr4[i1];
                        auto tmp11 = out_ptr1[i1];
                        auto tmp14 = in_ptr5[i1];
                        auto tmp19 = out_ptr0[i1];
                        auto tmp22 = in_ptr6[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = tmp2 ? tmp3 : tmp6;
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp12 = static_cast<float>(0.0009765625);
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp14 * tmp14;
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp10 * tmp16;
                        auto tmp18 = tmp7 - tmp17;
                        auto tmp20 = tmp19 * tmp12;
                        auto tmp21 = tmp18 - tmp20;
                        auto tmp23 = tmp14 * tmp22;
                        auto tmp24 = tmp21 * tmp23;
                        out_ptr2[i2 + (16*i1) + (8192*i0)] = tmp24;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<32; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=512; i0<512; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr5[i0];
                    auto tmp2 = tmp0 * tmp1;
                    in_out_ptr0[i0] = tmp2;
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
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<128; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<1; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (2048*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i0) + (16*i2) + (2048*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i0) + (16*i2) + (2048*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=16; i2<16; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (16*i0) + (2048*i1)];
                            auto tmp4 = in_ptr1[i2 + (16*i0) + (2048*i1)];
                            auto tmp7 = in_ptr2[i2 + (16*i0) + (2048*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<8; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=128; i0<128; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<128; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (16*i2) + (2048*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.0009765625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i1) + (2048*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (16*i1) + (2048*i0)];
                        auto tmp6 = in_ptr2[i2 + (16*i1) + (2048*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.0009765625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (16*i1) + (2048*i0)] = tmp22;
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
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<128; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<1; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (2048*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i0) + (16*i2) + (2048*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i0) + (16*i2) + (2048*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=16; i2<16; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (16*i0) + (2048*i1)];
                            auto tmp4 = in_ptr1[i2 + (16*i0) + (2048*i1)];
                            auto tmp7 = in_ptr2[i2 + (16*i0) + (2048*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<8; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=128; i0<128; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<128; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (16*i2) + (2048*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.0009765625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i1) + (2048*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (16*i1) + (2048*i0)];
                        auto tmp6 = in_ptr2[i2 + (16*i1) + (2048*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.0009765625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (16*i1) + (2048*i0)] = tmp22;
                    }
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
                       const float* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       const float* __restrict__ in_ptr7,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<32768; i0+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i0);
                auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                auto tmp5 = tmp4 <= tmp1;
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = decltype(tmp3)::blendv(tmp8, tmp3, tmp5);
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = decltype(tmp3)::blendv(tmp11, tmp3, tmp2);
                tmp12.store(in_out_ptr0 + 16*i0);
            }
            #pragma omp for simd simdlen(8) 
            for(long i0=524288; i0<524288; i0+=1)
            {
                auto tmp0 = in_ptr0[i0];
                auto tmp4 = in_ptr1[i0];
                auto tmp6 = in_out_ptr0[i0];
                auto tmp7 = in_ptr2[i0];
                auto tmp10 = in_ptr3[i0];
                auto tmp1 = static_cast<float>(0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = static_cast<float>(0.0);
                auto tmp5 = tmp4 <= tmp1;
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp5 ? tmp3 : tmp8;
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = tmp2 ? tmp3 : tmp11;
                in_out_ptr0[i0] = tmp12;
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<512; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<1; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i0) + (16*i2) + (8192*i1));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr4 + (16*i0) + (16*i2) + (8192*i1));
                            auto tmp3 = at::vec::Vectorized<float>(in_ptr5[i0]);
                            auto tmp4 = tmp2 - tmp3;
                            auto tmp5 = tmp0 * tmp4;
                            tmp1_vec += tmp0;
                            tmp6_vec += tmp5;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp1) reduction(+:tmp6)
                        for(long i2=16; i2<16; i2+=1)
                        {
                            auto tmp0 = in_out_ptr0[i2 + (16*i0) + (8192*i1)];
                            auto tmp2 = in_ptr4[i2 + (16*i0) + (8192*i1)];
                            auto tmp3 = in_ptr5[i0];
                            auto tmp4 = tmp2 - tmp3;
                            auto tmp5 = tmp0 * tmp4;
                            tmp1 += tmp0;
                            tmp6 += tmp5;
                        }
                    }
                    tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    out_ptr0[i0] = tmp1;
                    out_ptr1[i0] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<32; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=512; i0<512; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr6[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<512; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp2 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp4 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr6[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp15 = at::vec::Vectorized<float>(in_ptr7[i1]);
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.0009765625));
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp7 * tmp7;
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = tmp3 * tmp9;
                        auto tmp11 = tmp0 - tmp10;
                        auto tmp13 = tmp12 * tmp5;
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = tmp7 * tmp15;
                        auto tmp17 = tmp14 * tmp16;
                        tmp17.store(out_ptr3 + (16*i1) + (16*i2) + (8192*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_out_ptr0[i2 + (16*i1) + (8192*i0)];
                        auto tmp1 = in_ptr4[i2 + (16*i1) + (8192*i0)];
                        auto tmp2 = in_ptr5[i1];
                        auto tmp4 = out_ptr1[i1];
                        auto tmp7 = in_ptr6[i1];
                        auto tmp12 = out_ptr0[i1];
                        auto tmp15 = in_ptr7[i1];
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0009765625);
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp7 * tmp7;
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = tmp3 * tmp9;
                        auto tmp11 = tmp0 - tmp10;
                        auto tmp13 = tmp12 * tmp5;
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = tmp7 * tmp15;
                        auto tmp17 = tmp14 * tmp16;
                        out_ptr3[i2 + (16*i1) + (8192*i0)] = tmp17;
                    }
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
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<128; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<1; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (2048*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i0) + (16*i2) + (2048*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i0) + (16*i2) + (2048*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=16; i2<16; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (16*i0) + (2048*i1)];
                            auto tmp4 = in_ptr1[i2 + (16*i0) + (2048*i1)];
                            auto tmp7 = in_ptr2[i2 + (16*i0) + (2048*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<8; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=128; i0<128; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<128; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (16*i2) + (2048*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.0009765625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i1) + (2048*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (16*i1) + (2048*i0)];
                        auto tmp6 = in_ptr2[i2 + (16*i1) + (2048*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.0009765625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (16*i1) + (2048*i0)] = tmp22;
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
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<128; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<1; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (2048*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i0) + (16*i2) + (2048*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i0) + (16*i2) + (2048*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=16; i2<16; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (16*i0) + (2048*i1)];
                            auto tmp4 = in_ptr1[i2 + (16*i0) + (2048*i1)];
                            auto tmp7 = in_ptr2[i2 + (16*i0) + (2048*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<8; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=128; i0<128; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<128; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (16*i2) + (2048*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.0009765625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i1) + (2048*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (16*i1) + (2048*i0)];
                        auto tmp6 = in_ptr2[i2 + (16*i1) + (2048*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.0009765625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (16*i1) + (2048*i0)] = tmp22;
                    }
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
                       const float* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<512; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp8 = 0;
                    auto tmp8_vec = at::vec::Vectorized<float>(tmp8);
                    float tmp13 = 0;
                    auto tmp13_vec = at::vec::Vectorized<float>(tmp13);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<1; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (8192*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i0) + (16*i2) + (8192*i1));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i0) + (16*i2) + (8192*i1));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + (16*i0) + (16*i2) + (8192*i1));
                            auto tmp10 = at::vec::Vectorized<float>(in_ptr4[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp2);
                            auto tmp11 = tmp9 - tmp10;
                            auto tmp12 = tmp7 * tmp11;
                            tmp8_vec += tmp7;
                            tmp13_vec += tmp12;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp8) reduction(+:tmp13)
                        for(long i2=16; i2<16; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (16*i0) + (8192*i1)];
                            auto tmp4 = in_ptr1[i2 + (16*i0) + (8192*i1)];
                            auto tmp5 = in_ptr2[i2 + (16*i0) + (8192*i1)];
                            auto tmp9 = in_ptr3[i2 + (16*i0) + (8192*i1)];
                            auto tmp10 = in_ptr4[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp7 = tmp2 ? tmp3 : tmp6;
                            auto tmp11 = tmp9 - tmp10;
                            auto tmp12 = tmp7 * tmp11;
                            tmp8 += tmp7;
                            tmp13 += tmp12;
                        }
                    }
                    tmp8 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp8_vec);
                    tmp13 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp13_vec);
                    out_ptr0[i0] = tmp8;
                    out_ptr1[i0] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<512; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp9 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp11 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp14 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp19 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp22 = at::vec::Vectorized<float>(in_ptr6[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp2);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp12 = at::vec::Vectorized<float>(static_cast<float>(0.0009765625));
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp14 * tmp14;
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp10 * tmp16;
                        auto tmp18 = tmp7 - tmp17;
                        auto tmp20 = tmp19 * tmp12;
                        auto tmp21 = tmp18 - tmp20;
                        auto tmp23 = tmp14 * tmp22;
                        auto tmp24 = tmp21 * tmp23;
                        tmp24.store(out_ptr2 + (16*i1) + (16*i2) + (8192*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i1) + (8192*i0)];
                        auto tmp4 = in_ptr1[i2 + (16*i1) + (8192*i0)];
                        auto tmp5 = in_ptr2[i2 + (16*i1) + (8192*i0)];
                        auto tmp8 = in_ptr3[i2 + (16*i1) + (8192*i0)];
                        auto tmp9 = in_ptr4[i1];
                        auto tmp11 = out_ptr1[i1];
                        auto tmp14 = in_ptr5[i1];
                        auto tmp19 = out_ptr0[i1];
                        auto tmp22 = in_ptr6[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = tmp2 ? tmp3 : tmp6;
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp12 = static_cast<float>(0.0009765625);
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp14 * tmp14;
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp10 * tmp16;
                        auto tmp18 = tmp7 - tmp17;
                        auto tmp20 = tmp19 * tmp12;
                        auto tmp21 = tmp18 - tmp20;
                        auto tmp23 = tmp14 * tmp22;
                        auto tmp24 = tmp21 * tmp23;
                        out_ptr2[i2 + (16*i1) + (8192*i0)] = tmp24;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<32; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=512; i0<512; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr5[i0];
                    auto tmp2 = tmp0 * tmp1;
                    in_out_ptr0[i0] = tmp2;
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
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<128; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<1; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (2048*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i0) + (16*i2) + (2048*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i0) + (16*i2) + (2048*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=16; i2<16; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (16*i0) + (2048*i1)];
                            auto tmp4 = in_ptr1[i2 + (16*i0) + (2048*i1)];
                            auto tmp7 = in_ptr2[i2 + (16*i0) + (2048*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<8; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=128; i0<128; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<128; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (16*i2) + (2048*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.0009765625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i1) + (2048*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (16*i1) + (2048*i0)];
                        auto tmp6 = in_ptr2[i2 + (16*i1) + (2048*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.0009765625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (16*i1) + (2048*i0)] = tmp22;
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
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<128; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<1; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (2048*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i0) + (16*i2) + (2048*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i0) + (16*i2) + (2048*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=16; i2<16; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (16*i0) + (2048*i1)];
                            auto tmp4 = in_ptr1[i2 + (16*i0) + (2048*i1)];
                            auto tmp7 = in_ptr2[i2 + (16*i0) + (2048*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<8; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=128; i0<128; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<128; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (16*i2) + (2048*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.0009765625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i1) + (2048*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (16*i1) + (2048*i0)];
                        auto tmp6 = in_ptr2[i2 + (16*i1) + (2048*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.0009765625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (16*i1) + (2048*i0)] = tmp22;
                    }
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
                       const float* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       const float* __restrict__ in_ptr7,
                       const float* __restrict__ in_ptr8,
                       const float* __restrict__ in_ptr9,
                       const float* __restrict__ in_ptr10,
                       const float* __restrict__ in_ptr11,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<32768; i0+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i0);
                auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                auto tmp5 = tmp4 <= tmp1;
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = decltype(tmp3)::blendv(tmp8, tmp3, tmp5);
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = decltype(tmp3)::blendv(tmp11, tmp3, tmp2);
                tmp12.store(in_out_ptr0 + 16*i0);
            }
            #pragma omp for simd simdlen(8) 
            for(long i0=524288; i0<524288; i0+=1)
            {
                auto tmp0 = in_ptr0[i0];
                auto tmp4 = in_ptr1[i0];
                auto tmp6 = in_out_ptr0[i0];
                auto tmp7 = in_ptr2[i0];
                auto tmp10 = in_ptr3[i0];
                auto tmp1 = static_cast<float>(0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = static_cast<float>(0.0);
                auto tmp5 = tmp4 <= tmp1;
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp5 ? tmp3 : tmp8;
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = tmp2 ? tmp3 : tmp11;
                in_out_ptr0[i0] = tmp12;
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<512; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<1; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i0) + (16*i2) + (8192*i1));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr4 + (16*i0) + (16*i2) + (8192*i1));
                            auto tmp3 = at::vec::Vectorized<float>(in_ptr5[i0]);
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + (16*i0) + (16*i2) + (8192*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr7[i0]);
                            auto tmp4 = tmp2 - tmp3;
                            auto tmp5 = tmp0 * tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp0 * tmp9;
                            tmp1_vec += tmp0;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp1) reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=16; i2<16; i2+=1)
                        {
                            auto tmp0 = in_out_ptr0[i2 + (16*i0) + (8192*i1)];
                            auto tmp2 = in_ptr4[i2 + (16*i0) + (8192*i1)];
                            auto tmp3 = in_ptr5[i0];
                            auto tmp7 = in_ptr6[i2 + (16*i0) + (8192*i1)];
                            auto tmp8 = in_ptr7[i0];
                            auto tmp4 = tmp2 - tmp3;
                            auto tmp5 = tmp0 * tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp0 * tmp9;
                            tmp1 += tmp0;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp1;
                    out_ptr1[i0] = tmp6;
                    out_ptr2[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<32; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=512; i0<512; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr8[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr3[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<512; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp2 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp4 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr8[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp15 = at::vec::Vectorized<float>(in_ptr9[i1]);
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + (16*i1) + (16*i2) + (8192*i0));
                        auto tmp19 = at::vec::Vectorized<float>(in_ptr7[i1]);
                        auto tmp21 = at::vec::Vectorized<float>(out_ptr2[i1]);
                        auto tmp23 = at::vec::Vectorized<float>(in_ptr10[i1]);
                        auto tmp29 = at::vec::Vectorized<float>(in_ptr11[i1]);
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.0009765625));
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp7 * tmp7;
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = tmp3 * tmp9;
                        auto tmp11 = tmp0 - tmp10;
                        auto tmp13 = tmp12 * tmp5;
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = tmp7 * tmp15;
                        auto tmp17 = tmp14 * tmp16;
                        auto tmp20 = tmp18 - tmp19;
                        auto tmp22 = tmp21 * tmp5;
                        auto tmp24 = tmp23 * tmp23;
                        auto tmp25 = tmp22 * tmp24;
                        auto tmp26 = tmp20 * tmp25;
                        auto tmp27 = tmp0 - tmp26;
                        auto tmp28 = tmp27 - tmp13;
                        auto tmp30 = tmp23 * tmp29;
                        auto tmp31 = tmp28 * tmp30;
                        tmp17.store(out_ptr4 + (16*i1) + (16*i2) + (8192*i0));
                        tmp31.store(out_ptr5 + (16*i1) + (16*i2) + (8192*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_out_ptr0[i2 + (16*i1) + (8192*i0)];
                        auto tmp1 = in_ptr4[i2 + (16*i1) + (8192*i0)];
                        auto tmp2 = in_ptr5[i1];
                        auto tmp4 = out_ptr1[i1];
                        auto tmp7 = in_ptr8[i1];
                        auto tmp12 = out_ptr0[i1];
                        auto tmp15 = in_ptr9[i1];
                        auto tmp18 = in_ptr6[i2 + (16*i1) + (8192*i0)];
                        auto tmp19 = in_ptr7[i1];
                        auto tmp21 = out_ptr2[i1];
                        auto tmp23 = in_ptr10[i1];
                        auto tmp29 = in_ptr11[i1];
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0009765625);
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp7 * tmp7;
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = tmp3 * tmp9;
                        auto tmp11 = tmp0 - tmp10;
                        auto tmp13 = tmp12 * tmp5;
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = tmp7 * tmp15;
                        auto tmp17 = tmp14 * tmp16;
                        auto tmp20 = tmp18 - tmp19;
                        auto tmp22 = tmp21 * tmp5;
                        auto tmp24 = tmp23 * tmp23;
                        auto tmp25 = tmp22 * tmp24;
                        auto tmp26 = tmp20 * tmp25;
                        auto tmp27 = tmp0 - tmp26;
                        auto tmp28 = tmp27 - tmp13;
                        auto tmp30 = tmp23 * tmp29;
                        auto tmp31 = tmp28 * tmp30;
                        out_ptr4[i2 + (16*i1) + (8192*i0)] = tmp17;
                        out_ptr5[i2 + (16*i1) + (8192*i0)] = tmp31;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_39 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0)
{
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = in_out_ptr0[i0];
            auto tmp1 = in_ptr0[i0];
            auto tmp2 = tmp0 * tmp1;
            in_out_ptr0[i0] = tmp2;
        }
    }
}
''')


kernel_cpp_40 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<128; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<1; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i0) + (16*i2) + (2048*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i0) + (16*i2) + (2048*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i0) + (16*i2) + (2048*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=16; i2<16; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (16*i0) + (2048*i1)];
                            auto tmp4 = in_ptr1[i2 + (16*i0) + (2048*i1)];
                            auto tmp7 = in_ptr2[i2 + (16*i0) + (2048*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<8; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=128; i0<128; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<128; i1+=1)
                {
                    for(long i2=0; i2<1; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (16*i2) + (2048*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.0009765625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (16*i1) + (16*i2) + (2048*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=16; i2<16; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (16*i1) + (2048*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (16*i1) + (2048*i0)];
                        auto tmp6 = in_ptr2[i2 + (16*i1) + (2048*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.0009765625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (16*i1) + (2048*i0)] = tmp22;
                    }
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
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<128; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (8192*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i2) + (64*i0) + (8192*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i2) + (64*i0) + (8192*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=64; i2<64; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (64*i0) + (8192*i1)];
                            auto tmp4 = in_ptr1[i2 + (64*i0) + (8192*i1)];
                            auto tmp7 = in_ptr2[i2 + (64*i0) + (8192*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<8; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=128; i0<128; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<128; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i1) + (8192*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i2) + (64*i1) + (8192*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i2) + (64*i1) + (8192*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.000244140625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (16*i2) + (64*i1) + (8192*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i1) + (8192*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (64*i1) + (8192*i0)];
                        auto tmp6 = in_ptr2[i2 + (64*i1) + (8192*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.000244140625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (64*i1) + (8192*i0)] = tmp22;
                    }
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
                       const float* __restrict__ in_ptr6,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<256; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp8 = 0;
                    auto tmp8_vec = at::vec::Vectorized<float>(tmp8);
                    float tmp13 = 0;
                    auto tmp13_vec = at::vec::Vectorized<float>(tmp13);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (16384*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i2) + (64*i0) + (16384*i1));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i2) + (64*i0) + (16384*i1));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + (16*i2) + (64*i0) + (16384*i1));
                            auto tmp10 = at::vec::Vectorized<float>(in_ptr4[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp2);
                            auto tmp11 = tmp9 - tmp10;
                            auto tmp12 = tmp7 * tmp11;
                            tmp8_vec += tmp7;
                            tmp13_vec += tmp12;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp8) reduction(+:tmp13)
                        for(long i2=64; i2<64; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (64*i0) + (16384*i1)];
                            auto tmp4 = in_ptr1[i2 + (64*i0) + (16384*i1)];
                            auto tmp5 = in_ptr2[i2 + (64*i0) + (16384*i1)];
                            auto tmp9 = in_ptr3[i2 + (64*i0) + (16384*i1)];
                            auto tmp10 = in_ptr4[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp7 = tmp2 ? tmp3 : tmp6;
                            auto tmp11 = tmp9 - tmp10;
                            auto tmp12 = tmp7 * tmp11;
                            tmp8 += tmp7;
                            tmp13 += tmp12;
                        }
                    }
                    tmp8 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp8_vec);
                    tmp13 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp13_vec);
                    out_ptr0[i0] = tmp8;
                    out_ptr1[i0] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<256; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i1) + (16384*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i2) + (64*i1) + (16384*i0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i2) + (64*i1) + (16384*i0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + (16*i2) + (64*i1) + (16384*i0));
                        auto tmp9 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp11 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp14 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp19 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp22 = at::vec::Vectorized<float>(in_ptr6[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp2);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp12 = at::vec::Vectorized<float>(static_cast<float>(0.000244140625));
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp14 * tmp14;
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp10 * tmp16;
                        auto tmp18 = tmp7 - tmp17;
                        auto tmp20 = tmp19 * tmp12;
                        auto tmp21 = tmp18 - tmp20;
                        auto tmp23 = tmp14 * tmp22;
                        auto tmp24 = tmp21 * tmp23;
                        tmp24.store(out_ptr2 + (16*i2) + (64*i1) + (16384*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i1) + (16384*i0)];
                        auto tmp4 = in_ptr1[i2 + (64*i1) + (16384*i0)];
                        auto tmp5 = in_ptr2[i2 + (64*i1) + (16384*i0)];
                        auto tmp8 = in_ptr3[i2 + (64*i1) + (16384*i0)];
                        auto tmp9 = in_ptr4[i1];
                        auto tmp11 = out_ptr1[i1];
                        auto tmp14 = in_ptr5[i1];
                        auto tmp19 = out_ptr0[i1];
                        auto tmp22 = in_ptr6[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = tmp2 ? tmp3 : tmp6;
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp12 = static_cast<float>(0.000244140625);
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp14 * tmp14;
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp10 * tmp16;
                        auto tmp18 = tmp7 - tmp17;
                        auto tmp20 = tmp19 * tmp12;
                        auto tmp21 = tmp18 - tmp20;
                        auto tmp23 = tmp14 * tmp22;
                        auto tmp24 = tmp21 * tmp23;
                        out_ptr2[i2 + (64*i1) + (16384*i0)] = tmp24;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr5[i0];
                    auto tmp2 = tmp0 * tmp1;
                    in_out_ptr0[i0] = tmp2;
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
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (4096*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i2) + (64*i0) + (4096*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i2) + (64*i0) + (4096*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=64; i2<64; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (64*i0) + (4096*i1)];
                            auto tmp4 = in_ptr1[i2 + (64*i0) + (4096*i1)];
                            auto tmp7 = in_ptr2[i2 + (64*i0) + (4096*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<4; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=64; i0<64; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<64; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.000244140625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i1) + (4096*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (64*i1) + (4096*i0)];
                        auto tmp6 = in_ptr2[i2 + (64*i1) + (4096*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.000244140625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (64*i1) + (4096*i0)] = tmp22;
                    }
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
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (4096*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i2) + (64*i0) + (4096*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i2) + (64*i0) + (4096*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=64; i2<64; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (64*i0) + (4096*i1)];
                            auto tmp4 = in_ptr1[i2 + (64*i0) + (4096*i1)];
                            auto tmp7 = in_ptr2[i2 + (64*i0) + (4096*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<4; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=64; i0<64; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<64; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.000244140625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i1) + (4096*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (64*i1) + (4096*i0)];
                        auto tmp6 = in_ptr2[i2 + (64*i1) + (4096*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.000244140625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (64*i1) + (4096*i0)] = tmp22;
                    }
                }
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
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       const float* __restrict__ in_ptr7,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<65536; i0+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i0);
                auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                auto tmp5 = tmp4 <= tmp1;
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = decltype(tmp3)::blendv(tmp8, tmp3, tmp5);
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = decltype(tmp3)::blendv(tmp11, tmp3, tmp2);
                tmp12.store(in_out_ptr0 + 16*i0);
            }
            #pragma omp for simd simdlen(8) 
            for(long i0=1048576; i0<1048576; i0+=1)
            {
                auto tmp0 = in_ptr0[i0];
                auto tmp4 = in_ptr1[i0];
                auto tmp6 = in_out_ptr0[i0];
                auto tmp7 = in_ptr2[i0];
                auto tmp10 = in_ptr3[i0];
                auto tmp1 = static_cast<float>(0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = static_cast<float>(0.0);
                auto tmp5 = tmp4 <= tmp1;
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp5 ? tmp3 : tmp8;
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = tmp2 ? tmp3 : tmp11;
                in_out_ptr0[i0] = tmp12;
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<256; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp1 = 0;
                    auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i2) + (64*i0) + (16384*i1));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr4 + (16*i2) + (64*i0) + (16384*i1));
                            auto tmp3 = at::vec::Vectorized<float>(in_ptr5[i0]);
                            auto tmp4 = tmp2 - tmp3;
                            auto tmp5 = tmp0 * tmp4;
                            tmp1_vec += tmp0;
                            tmp6_vec += tmp5;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp1) reduction(+:tmp6)
                        for(long i2=64; i2<64; i2+=1)
                        {
                            auto tmp0 = in_out_ptr0[i2 + (64*i0) + (16384*i1)];
                            auto tmp2 = in_ptr4[i2 + (64*i0) + (16384*i1)];
                            auto tmp3 = in_ptr5[i0];
                            auto tmp4 = tmp2 - tmp3;
                            auto tmp5 = tmp0 * tmp4;
                            tmp1 += tmp0;
                            tmp6 += tmp5;
                        }
                    }
                    tmp1 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    out_ptr0[i0] = tmp1;
                    out_ptr1[i0] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr6[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<256; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i2) + (64*i1) + (16384*i0));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + (16*i2) + (64*i1) + (16384*i0));
                        auto tmp2 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp4 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr6[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp15 = at::vec::Vectorized<float>(in_ptr7[i1]);
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.000244140625));
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp7 * tmp7;
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = tmp3 * tmp9;
                        auto tmp11 = tmp0 - tmp10;
                        auto tmp13 = tmp12 * tmp5;
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = tmp7 * tmp15;
                        auto tmp17 = tmp14 * tmp16;
                        tmp17.store(out_ptr3 + (16*i2) + (64*i1) + (16384*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_out_ptr0[i2 + (64*i1) + (16384*i0)];
                        auto tmp1 = in_ptr4[i2 + (64*i1) + (16384*i0)];
                        auto tmp2 = in_ptr5[i1];
                        auto tmp4 = out_ptr1[i1];
                        auto tmp7 = in_ptr6[i1];
                        auto tmp12 = out_ptr0[i1];
                        auto tmp15 = in_ptr7[i1];
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.000244140625);
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp7 * tmp7;
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = tmp3 * tmp9;
                        auto tmp11 = tmp0 - tmp10;
                        auto tmp13 = tmp12 * tmp5;
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = tmp7 * tmp15;
                        auto tmp17 = tmp14 * tmp16;
                        out_ptr3[i2 + (64*i1) + (16384*i0)] = tmp17;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_46 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (4096*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i2) + (64*i0) + (4096*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i2) + (64*i0) + (4096*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=64; i2<64; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (64*i0) + (4096*i1)];
                            auto tmp4 = in_ptr1[i2 + (64*i0) + (4096*i1)];
                            auto tmp7 = in_ptr2[i2 + (64*i0) + (4096*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<4; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=64; i0<64; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<64; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.000244140625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i1) + (4096*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (64*i1) + (4096*i0)];
                        auto tmp6 = in_ptr2[i2 + (64*i1) + (4096*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.000244140625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (64*i1) + (4096*i0)] = tmp22;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_47 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (4096*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i2) + (64*i0) + (4096*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i2) + (64*i0) + (4096*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=64; i2<64; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (64*i0) + (4096*i1)];
                            auto tmp4 = in_ptr1[i2 + (64*i0) + (4096*i1)];
                            auto tmp7 = in_ptr2[i2 + (64*i0) + (4096*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<4; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=64; i0<64; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<64; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.000244140625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i1) + (4096*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (64*i1) + (4096*i0)];
                        auto tmp6 = in_ptr2[i2 + (64*i1) + (4096*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.000244140625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (64*i1) + (4096*i0)] = tmp22;
                    }
                }
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
                       const float* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       const float* __restrict__ in_ptr7,
                       const float* __restrict__ in_ptr8,
                       const float* __restrict__ in_ptr9,
                       const float* __restrict__ in_ptr10,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<256; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp8 = 0;
                    auto tmp8_vec = at::vec::Vectorized<float>(tmp8);
                    float tmp13 = 0;
                    auto tmp13_vec = at::vec::Vectorized<float>(tmp13);
                    float tmp18 = 0;
                    auto tmp18_vec = at::vec::Vectorized<float>(tmp18);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (16384*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i2) + (64*i0) + (16384*i1));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i2) + (64*i0) + (16384*i1));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + (16*i2) + (64*i0) + (16384*i1));
                            auto tmp10 = at::vec::Vectorized<float>(in_ptr4[i0]);
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + (16*i2) + (64*i0) + (16384*i1));
                            auto tmp15 = at::vec::Vectorized<float>(in_ptr6[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp2);
                            auto tmp11 = tmp9 - tmp10;
                            auto tmp12 = tmp7 * tmp11;
                            auto tmp16 = tmp14 - tmp15;
                            auto tmp17 = tmp7 * tmp16;
                            tmp8_vec += tmp7;
                            tmp13_vec += tmp12;
                            tmp18_vec += tmp17;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp8) reduction(+:tmp13) reduction(+:tmp18)
                        for(long i2=64; i2<64; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (64*i0) + (16384*i1)];
                            auto tmp4 = in_ptr1[i2 + (64*i0) + (16384*i1)];
                            auto tmp5 = in_ptr2[i2 + (64*i0) + (16384*i1)];
                            auto tmp9 = in_ptr3[i2 + (64*i0) + (16384*i1)];
                            auto tmp10 = in_ptr4[i0];
                            auto tmp14 = in_ptr5[i2 + (64*i0) + (16384*i1)];
                            auto tmp15 = in_ptr6[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp7 = tmp2 ? tmp3 : tmp6;
                            auto tmp11 = tmp9 - tmp10;
                            auto tmp12 = tmp7 * tmp11;
                            auto tmp16 = tmp14 - tmp15;
                            auto tmp17 = tmp7 * tmp16;
                            tmp8 += tmp7;
                            tmp13 += tmp12;
                            tmp18 += tmp17;
                        }
                    }
                    tmp8 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp8_vec);
                    tmp13 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp13_vec);
                    tmp18 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp18_vec);
                    out_ptr0[i0] = tmp8;
                    out_ptr1[i0] = tmp13;
                    out_ptr2[i0] = tmp18;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<256; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i1) + (16384*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i2) + (64*i1) + (16384*i0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i2) + (64*i1) + (16384*i0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + (16*i2) + (64*i1) + (16384*i0));
                        auto tmp9 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp11 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp14 = at::vec::Vectorized<float>(in_ptr7[i1]);
                        auto tmp19 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp22 = at::vec::Vectorized<float>(in_ptr8[i1]);
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + (16*i2) + (64*i1) + (16384*i0));
                        auto tmp26 = at::vec::Vectorized<float>(in_ptr6[i1]);
                        auto tmp28 = at::vec::Vectorized<float>(out_ptr2[i1]);
                        auto tmp30 = at::vec::Vectorized<float>(in_ptr9[i1]);
                        auto tmp36 = at::vec::Vectorized<float>(in_ptr10[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp2);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp12 = at::vec::Vectorized<float>(static_cast<float>(0.000244140625));
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp14 * tmp14;
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp10 * tmp16;
                        auto tmp18 = tmp7 - tmp17;
                        auto tmp20 = tmp19 * tmp12;
                        auto tmp21 = tmp18 - tmp20;
                        auto tmp23 = tmp14 * tmp22;
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp27 = tmp25 - tmp26;
                        auto tmp29 = tmp28 * tmp12;
                        auto tmp31 = tmp30 * tmp30;
                        auto tmp32 = tmp29 * tmp31;
                        auto tmp33 = tmp27 * tmp32;
                        auto tmp34 = tmp7 - tmp33;
                        auto tmp35 = tmp34 - tmp20;
                        auto tmp37 = tmp30 * tmp36;
                        auto tmp38 = tmp35 * tmp37;
                        tmp24.store(out_ptr3 + (16*i2) + (64*i1) + (16384*i0));
                        tmp38.store(out_ptr4 + (16*i2) + (64*i1) + (16384*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i1) + (16384*i0)];
                        auto tmp4 = in_ptr1[i2 + (64*i1) + (16384*i0)];
                        auto tmp5 = in_ptr2[i2 + (64*i1) + (16384*i0)];
                        auto tmp8 = in_ptr3[i2 + (64*i1) + (16384*i0)];
                        auto tmp9 = in_ptr4[i1];
                        auto tmp11 = out_ptr1[i1];
                        auto tmp14 = in_ptr7[i1];
                        auto tmp19 = out_ptr0[i1];
                        auto tmp22 = in_ptr8[i1];
                        auto tmp25 = in_ptr5[i2 + (64*i1) + (16384*i0)];
                        auto tmp26 = in_ptr6[i1];
                        auto tmp28 = out_ptr2[i1];
                        auto tmp30 = in_ptr9[i1];
                        auto tmp36 = in_ptr10[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = tmp2 ? tmp3 : tmp6;
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp12 = static_cast<float>(0.000244140625);
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp14 * tmp14;
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp10 * tmp16;
                        auto tmp18 = tmp7 - tmp17;
                        auto tmp20 = tmp19 * tmp12;
                        auto tmp21 = tmp18 - tmp20;
                        auto tmp23 = tmp14 * tmp22;
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp27 = tmp25 - tmp26;
                        auto tmp29 = tmp28 * tmp12;
                        auto tmp31 = tmp30 * tmp30;
                        auto tmp32 = tmp29 * tmp31;
                        auto tmp33 = tmp27 * tmp32;
                        auto tmp34 = tmp7 - tmp33;
                        auto tmp35 = tmp34 - tmp20;
                        auto tmp37 = tmp30 * tmp36;
                        auto tmp38 = tmp35 * tmp37;
                        out_ptr3[i2 + (64*i1) + (16384*i0)] = tmp24;
                        out_ptr4[i2 + (64*i1) + (16384*i0)] = tmp38;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<16; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=256; i0<256; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr7[i0];
                    auto tmp2 = tmp0 * tmp1;
                    in_out_ptr0[i0] = tmp2;
                }
            }
        }
    }
}
''')


kernel_cpp_49 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0)
{
    {
        for(long i0=0; i0<16; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=256; i0<256; i0+=1)
        {
            auto tmp0 = in_out_ptr0[i0];
            auto tmp1 = in_ptr0[i0];
            auto tmp2 = tmp0 * tmp1;
            in_out_ptr0[i0] = tmp2;
        }
    }
}
''')


kernel_cpp_50 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (4096*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i2) + (64*i0) + (4096*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i2) + (64*i0) + (4096*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=64; i2<64; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (64*i0) + (4096*i1)];
                            auto tmp4 = in_ptr1[i2 + (64*i0) + (4096*i1)];
                            auto tmp7 = in_ptr2[i2 + (64*i0) + (4096*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<4; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=64; i0<64; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<64; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.000244140625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i1) + (4096*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (64*i1) + (4096*i0)];
                        auto tmp6 = in_ptr2[i2 + (64*i1) + (4096*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.000244140625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (64*i1) + (4096*i0)] = tmp22;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_51 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<4; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i0) + (4096*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i2) + (64*i0) + (4096*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i2) + (64*i0) + (4096*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr3[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=64; i2<64; i2+=1)
                        {
                            auto tmp0 = in_ptr0[i2 + (64*i0) + (4096*i1)];
                            auto tmp4 = in_ptr1[i2 + (64*i0) + (4096*i1)];
                            auto tmp7 = in_ptr2[i2 + (64*i0) + (4096*i1)];
                            auto tmp8 = in_ptr3[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr0[i0] = tmp6;
                    out_ptr1[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<4; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=64; i0<64; i0+=1)
                {
                    auto tmp0 = out_ptr1[i0];
                    auto tmp1 = in_ptr4[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr2[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<64; i1+=1)
                {
                    for(long i2=0; i2<4; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i2) + (64*i1) + (4096*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr3[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr0[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.000244140625));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr0 + (16*i2) + (64*i1) + (4096*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=64; i2<64; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (64*i1) + (4096*i0)];
                        auto tmp4 = in_out_ptr0[i2 + (64*i1) + (4096*i0)];
                        auto tmp6 = in_ptr2[i2 + (64*i1) + (4096*i0)];
                        auto tmp7 = in_ptr3[i1];
                        auto tmp9 = out_ptr1[i1];
                        auto tmp12 = in_ptr4[i1];
                        auto tmp17 = out_ptr0[i1];
                        auto tmp20 = in_ptr5[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.000244140625);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr0[i2 + (64*i1) + (4096*i0)] = tmp22;
                    }
                }
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
                       const long* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<16384; i0+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + 16*i0);
            }
            #pragma omp for simd simdlen(8) 
            for(long i0=262144; i0<262144; i0+=1)
            {
                auto tmp0 = in_out_ptr0[i0];
                auto tmp1 = in_ptr0[i0];
                auto tmp2 = tmp0 + tmp1;
                in_out_ptr0[i0] = tmp2;
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<4096; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<16; i1+=1)
                {
                    #pragma GCC ivdep
                    for(long i2=0; i2<16; i2+=1)
                    {
                        auto tmp0 = static_cast<int>(i2 + (16*i1));
                        auto tmp1 = static_cast<int>((i1 / 2));
                        auto tmp2 = static_cast<int>((i2 / 2));
                        auto tmp3 = static_cast<int>(1 + (((1 + i1) / 2)));
                        auto tmp4 = static_cast<int>(1 + (((1 + i2) / 2)));
                        auto tmp5 = static_cast<int>(0);
                        auto tmp6 = (tmp5 != tmp5) ? tmp5 : std::max(tmp1, tmp5);
                        auto tmp7 = (tmp5 != tmp5) ? tmp5 : std::max(tmp2, tmp5);
                        auto tmp8 = static_cast<int>(8);
                        auto tmp9 = (tmp8 != tmp8) ? tmp8 : std::min(tmp3, tmp8);
                        auto tmp10 = (tmp8 != tmp8) ? tmp8 : std::min(tmp4, tmp8);
                        auto tmp11 = tmp6 + tmp5;
                        auto tmp12 = tmp7 + tmp5;
                        auto tmp13 = static_cast<int>(1);
                        auto tmp14 = tmp9 - tmp13;
                        auto tmp15 = (tmp14 != tmp14) ? tmp14 : std::min(tmp11, tmp14);
                        auto tmp16 = tmp10 - tmp13;
                        auto tmp17 = (tmp16 != tmp16) ? tmp16 : std::min(tmp12, tmp16);
                        auto tmp18 = in_ptr1[tmp17 + (8*tmp15) + (64*i0)];
                        auto tmp19 = in_out_ptr0[tmp17 + (8*tmp15) + (64*i0)];
                        auto tmp20 = tmp18 == tmp0;
                        auto tmp21 = static_cast<float>(0.0);
                        auto tmp22 = tmp20 ? tmp19 : tmp21;
                        auto tmp23 = tmp7 + tmp13;
                        auto tmp24 = (tmp16 != tmp16) ? tmp16 : std::min(tmp23, tmp16);
                        auto tmp25 = in_ptr1[tmp24 + (8*tmp15) + (64*i0)];
                        auto tmp26 = in_out_ptr0[tmp24 + (8*tmp15) + (64*i0)];
                        auto tmp27 = tmp25 == tmp0;
                        auto tmp28 = tmp11 < tmp9;
                        auto tmp29 = tmp23 < tmp10;
                        auto tmp30 = tmp28 & tmp29;
                        auto tmp31 = tmp30 & tmp27;
                        auto tmp32 = tmp22 + tmp26;
                        auto tmp33 = tmp31 ? tmp32 : tmp22;
                        auto tmp34 = tmp6 + tmp13;
                        auto tmp35 = (tmp14 != tmp14) ? tmp14 : std::min(tmp34, tmp14);
                        auto tmp36 = in_ptr1[tmp17 + (8*tmp35) + (64*i0)];
                        auto tmp37 = in_out_ptr0[tmp17 + (8*tmp35) + (64*i0)];
                        auto tmp38 = tmp36 == tmp0;
                        auto tmp39 = tmp34 < tmp9;
                        auto tmp40 = tmp12 < tmp10;
                        auto tmp41 = tmp39 & tmp40;
                        auto tmp42 = tmp41 & tmp38;
                        auto tmp43 = tmp33 + tmp37;
                        auto tmp44 = tmp42 ? tmp43 : tmp33;
                        auto tmp45 = in_ptr1[tmp24 + (8*tmp35) + (64*i0)];
                        auto tmp46 = in_out_ptr0[tmp24 + (8*tmp35) + (64*i0)];
                        auto tmp47 = tmp45 == tmp0;
                        auto tmp48 = tmp39 & tmp29;
                        auto tmp49 = tmp48 & tmp47;
                        auto tmp50 = tmp44 + tmp46;
                        auto tmp51 = tmp49 ? tmp50 : tmp44;
                        out_ptr0[i2 + (16*i1) + (256*i0)] = tmp51;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<64; i1+=1)
                    {
                        for(long i2=0; i2<16; i2+=1)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i2) + (256*i0) + (16384*i1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i2) + (256*i0) + (16384*i1));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + (16*i2) + (256*i0) + (16384*i1));
                            auto tmp8 = at::vec::Vectorized<float>(in_ptr4[i0]);
                            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                            auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6_vec += tmp5;
                            tmp11_vec += tmp10;
                        }
                        #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp11)
                        for(long i2=256; i2<256; i2+=1)
                        {
                            auto tmp0 = in_ptr2[i2 + (256*i0) + (16384*i1)];
                            auto tmp4 = out_ptr0[i2 + (256*i0) + (16384*i1)];
                            auto tmp7 = in_ptr3[i2 + (256*i0) + (16384*i1)];
                            auto tmp8 = in_ptr4[i0];
                            auto tmp1 = static_cast<float>(0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = tmp5 * tmp9;
                            tmp6 += tmp5;
                            tmp11 += tmp10;
                        }
                    }
                    tmp6 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                    tmp11 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp11_vec);
                    out_ptr1[i0] = tmp6;
                    out_ptr2[i0] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long i0=0; i0<4; i0+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + 16*i0);
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + 16*i0);
                }
                #pragma omp simd simdlen(8) 
                for(long i0=64; i0<64; i0+=1)
                {
                    auto tmp0 = out_ptr2[i0];
                    auto tmp1 = in_ptr5[i0];
                    auto tmp2 = tmp0 * tmp1;
                    out_ptr3[i0] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<64; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<64; i1+=1)
                {
                    for(long i2=0; i2<16; i2+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i2) + (256*i1) + (16384*i0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i2) + (256*i1) + (16384*i0));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + (16*i2) + (256*i1) + (16384*i0));
                        auto tmp7 = at::vec::Vectorized<float>(in_ptr4[i1]);
                        auto tmp9 = at::vec::Vectorized<float>(out_ptr2[i1]);
                        auto tmp12 = at::vec::Vectorized<float>(in_ptr5[i1]);
                        auto tmp17 = at::vec::Vectorized<float>(out_ptr1[i1]);
                        auto tmp20 = at::vec::Vectorized<float>(in_ptr6[i1]);
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                        auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(6.103515625e-05));
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        tmp22.store(in_out_ptr1 + (16*i2) + (256*i1) + (16384*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i2=256; i2<256; i2+=1)
                    {
                        auto tmp0 = in_ptr2[i2 + (256*i1) + (16384*i0)];
                        auto tmp4 = out_ptr0[i2 + (256*i1) + (16384*i0)];
                        auto tmp6 = in_ptr3[i2 + (256*i1) + (16384*i0)];
                        auto tmp7 = in_ptr4[i1];
                        auto tmp9 = out_ptr2[i1];
                        auto tmp12 = in_ptr5[i1];
                        auto tmp17 = out_ptr1[i1];
                        auto tmp20 = in_ptr6[i1];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(6.103515625e-05);
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp12 * tmp12;
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp8 * tmp14;
                        auto tmp16 = tmp5 - tmp15;
                        auto tmp18 = tmp17 * tmp10;
                        auto tmp19 = tmp16 - tmp18;
                        auto tmp21 = tmp12 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        in_out_ptr1[i2 + (256*i1) + (16384*i0)] = tmp22;
                    }
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_321, convolution, squeeze_1, relu, getitem_2, getitem_3, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, relu_4, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, relu_8, convolution_10, squeeze_31, relu_9, convolution_11, squeeze_34, relu_10, convolution_12, squeeze_37, relu_11, convolution_13, squeeze_40, convolution_14, squeeze_43, relu_12, convolution_15, squeeze_46, relu_13, convolution_16, squeeze_49, relu_14, convolution_17, squeeze_52, relu_15, convolution_18, squeeze_55, relu_16, convolution_19, squeeze_58, relu_17, convolution_20, squeeze_61, relu_18, convolution_21, squeeze_64, relu_19, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, relu_21, convolution_24, squeeze_73, relu_22, convolution_25, squeeze_76, relu_23, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_24, convolution_28, squeeze_85, relu_25, convolution_29, squeeze_88, relu_26, convolution_30, squeeze_91, relu_27, convolution_31, squeeze_94, relu_28, convolution_32, squeeze_97, relu_29, convolution_33, squeeze_100, relu_30, convolution_34, squeeze_103, relu_31, convolution_35, squeeze_106, relu_32, convolution_36, squeeze_109, relu_33, convolution_37, squeeze_112, relu_34, convolution_38, squeeze_115, relu_35, convolution_39, squeeze_118, relu_36, convolution_40, squeeze_121, relu_37, convolution_41, squeeze_124, relu_38, convolution_42, squeeze_127, relu_39, convolution_43, squeeze_130, relu_40, convolution_44, squeeze_133, relu_41, convolution_45, squeeze_136, convolution_46, squeeze_139, relu_42, convolution_47, squeeze_142, relu_43, convolution_48, squeeze_145, relu_44, convolution_49, squeeze_148, relu_45, convolution_50, squeeze_151, relu_46, convolution_51, squeeze_154, relu_47, convolution_52, squeeze_157, view, permute_1, le, unsqueeze_214, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50, tangents_51, tangents_52, tangents_53, tangents_54, tangents_55, tangents_56, tangents_57, tangents_58, tangents_59, tangents_60, tangents_61, tangents_62, tangents_63, tangents_64, tangents_65, tangents_66, tangents_67, tangents_68, tangents_69, tangents_70, tangents_71, tangents_72, tangents_73, tangents_74, tangents_75, tangents_76, tangents_77, tangents_78, tangents_79, tangents_80, tangents_81, tangents_82, tangents_83, tangents_84, tangents_85, tangents_86, tangents_87, tangents_88, tangents_89, tangents_90, tangents_91, tangents_92, tangents_93, tangents_94, tangents_95, tangents_96, tangents_97, tangents_98, tangents_99, tangents_100, tangents_101, tangents_102, tangents_103, tangents_104, tangents_105, tangents_106, tangents_107, tangents_108, tangents_109, tangents_110, tangents_111, tangents_112, tangents_113, tangents_114, tangents_115, tangents_116, tangents_117, tangents_118, tangents_119, tangents_120, tangents_121, tangents_122, tangents_123, tangents_124, tangents_125, tangents_126, tangents_127, tangents_128, tangents_129, tangents_130, tangents_131, tangents_132, tangents_133, tangents_134, tangents_135, tangents_136, tangents_137, tangents_138, tangents_139, tangents_140, tangents_141, tangents_142, tangents_143, tangents_144, tangents_145, tangents_146, tangents_147, tangents_148, tangents_149, tangents_150, tangents_151, tangents_152, tangents_153, tangents_154, tangents_155, tangents_156, tangents_157, tangents_158, tangents_159, tangents_160 = args
    args.clear()
    buf0 = empty_strided((64, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    extern_kernels.mm(tangents_160, permute_1, out=buf0)
    del permute_1
    buf1 = empty_strided((1000, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    extern_kernels.mm(as_strided(tangents_160, (1000, 64), (1, 1000)), view, out=buf1)
    del view
    buf2 = empty_strided((1, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((64, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_0(c_void_p(tangents_160.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(unsqueeze_214.data_ptr()), c_void_p(squeeze_157.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()))
    del convolution_52
    del primals_158
    del squeeze_157
    del tangents_160
    del unsqueeze_214
    buf7 = aten.convolution_backward(buf6, relu_47, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_157
    buf8 = buf7[0]
    assert_size_stride(buf8, (64, 512, 1, 1), (512, 1, 1, 1))
    buf9 = buf7[1]
    assert_size_stride(buf9, (2048, 512, 1, 1), (512, 1, 1, 1))
    del buf7
    buf10 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf11 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf12 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf13 = buf8; del buf8  # reuse
    kernel_cpp_1(c_void_p(buf13.data_ptr()), c_void_p(relu_47.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(unsqueeze_226.data_ptr()), c_void_p(squeeze_154.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()))
    del convolution_51
    del primals_155
    del relu_47
    del squeeze_154
    del unsqueeze_226
    buf14 = aten.convolution_backward(buf13, relu_46, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf13
    del primals_154
    buf15 = buf14[0]
    assert_size_stride(buf15, (64, 512, 1, 1), (512, 1, 1, 1))
    buf16 = buf14[1]
    assert_size_stride(buf16, (512, 512, 3, 3), (4608, 9, 3, 1))
    del buf14
    buf17 = buf11; del buf11  # reuse
    buf18 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf19 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf20 = buf15; del buf15  # reuse
    kernel_cpp_2(c_void_p(buf20.data_ptr()), c_void_p(relu_46.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(unsqueeze_238.data_ptr()), c_void_p(squeeze_151.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()))
    del convolution_50
    del primals_152
    del relu_46
    del squeeze_151
    del unsqueeze_238
    buf21 = aten.convolution_backward(buf20, relu_45, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf20
    del primals_151
    buf22 = buf21[0]
    assert_size_stride(buf22, (64, 2048, 1, 1), (2048, 1, 1, 1))
    buf23 = buf21[1]
    assert_size_stride(buf23, (512, 2048, 1, 1), (2048, 1, 1, 1))
    del buf21
    buf24 = buf4; del buf4  # reuse
    buf25 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf26 = as_strided(buf6, (64, 2048, 1, 1), (2048, 1, 131072, 131072)); del buf6  # reuse
    buf28 = as_strided(buf26, (64, 2048, 1, 1), (2048, 1, 1, 1)); del buf26  # reuse
    buf27 = buf25; del buf25  # reuse
    kernel_cpp_3(c_void_p(buf28.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(relu_45.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(unsqueeze_250.data_ptr()), c_void_p(squeeze_148.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(buf24.data_ptr()))
    del convolution_49
    del primals_149
    del squeeze_148
    del unsqueeze_250
    buf29 = aten.convolution_backward(buf28, relu_44, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_148
    buf30 = buf29[0]
    assert_size_stride(buf30, (64, 512, 1, 1), (512, 1, 1, 1))
    buf31 = buf29[1]
    assert_size_stride(buf31, (2048, 512, 1, 1), (512, 1, 1, 1))
    del buf29
    buf32 = buf18; del buf18  # reuse
    buf33 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf34 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf35 = buf30; del buf30  # reuse
    kernel_cpp_4(c_void_p(buf35.data_ptr()), c_void_p(relu_44.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(unsqueeze_262.data_ptr()), c_void_p(squeeze_145.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()))
    del convolution_48
    del primals_146
    del relu_44
    del squeeze_145
    del unsqueeze_262
    buf36 = aten.convolution_backward(buf35, relu_43, primals_145, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf35
    del primals_145
    buf37 = buf36[0]
    assert_size_stride(buf37, (64, 512, 1, 1), (512, 1, 1, 1))
    buf38 = buf36[1]
    assert_size_stride(buf38, (512, 512, 3, 3), (4608, 9, 3, 1))
    del buf36
    buf39 = buf33; del buf33  # reuse
    buf40 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf41 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf42 = buf37; del buf37  # reuse
    kernel_cpp_5(c_void_p(buf42.data_ptr()), c_void_p(relu_43.data_ptr()), c_void_p(convolution_47.data_ptr()), c_void_p(unsqueeze_274.data_ptr()), c_void_p(squeeze_142.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()))
    del convolution_47
    del primals_143
    del relu_43
    del squeeze_142
    del unsqueeze_274
    buf43 = aten.convolution_backward(buf42, relu_42, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf42
    del primals_142
    buf44 = buf43[0]
    assert_size_stride(buf44, (64, 2048, 1, 1), (2048, 1, 1, 1))
    buf45 = buf43[1]
    assert_size_stride(buf45, (512, 2048, 1, 1), (2048, 1, 1, 1))
    del buf43
    buf46 = as_strided(buf0, (64, 2048, 1, 1), (2048, 1, 131072, 131072)); del buf0  # reuse
    buf47 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf48 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf54 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf49 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf50 = buf28; del buf28  # reuse
    buf56 = empty_strided((64, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_6(c_void_p(buf46.data_ptr()), c_void_p(relu_42.data_ptr()), c_void_p(relu_45.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(unsqueeze_286.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(unsqueeze_298.data_ptr()), c_void_p(squeeze_139.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(squeeze_136.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf56.data_ptr()))
    del buf22
    del buf44
    del buf46
    del buf48
    del convolution_45
    del convolution_46
    del le
    del primals_137
    del primals_140
    del relu_42
    del relu_45
    del squeeze_139
    del unsqueeze_286
    del unsqueeze_298
    buf51 = aten.convolution_backward(buf50, relu_39, primals_139, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf50
    del primals_139
    buf52 = buf51[0]
    assert_size_stride(buf52, (64, 1024, 2, 2), (4096, 4, 2, 1))
    buf53 = buf51[1]
    assert_size_stride(buf53, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    del buf51
    buf55 = buf54; del buf54  # reuse
    kernel_cpp_7(c_void_p(buf55.data_ptr()), c_void_p(squeeze_136.data_ptr()))
    del squeeze_136
    buf57 = aten.convolution_backward(buf56, relu_41, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf56
    del primals_136
    buf58 = buf57[0]
    assert_size_stride(buf58, (64, 512, 1, 1), (512, 1, 1, 1))
    buf59 = buf57[1]
    assert_size_stride(buf59, (2048, 512, 1, 1), (512, 1, 1, 1))
    del buf57
    buf60 = buf40; del buf40  # reuse
    buf61 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf62 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf63 = buf58; del buf58  # reuse
    kernel_cpp_8(c_void_p(buf63.data_ptr()), c_void_p(relu_41.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(unsqueeze_310.data_ptr()), c_void_p(squeeze_133.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()))
    del convolution_44
    del primals_134
    del relu_41
    del squeeze_133
    del unsqueeze_310
    buf64 = aten.convolution_backward(buf63, relu_40, primals_133, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf63
    del primals_133
    buf65 = buf64[0]
    assert_size_stride(buf65, (64, 512, 2, 2), (2048, 4, 2, 1))
    buf66 = buf64[1]
    assert_size_stride(buf66, (512, 512, 3, 3), (4608, 9, 3, 1))
    del buf64
    buf67 = buf61; del buf61  # reuse
    buf68 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf69 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf70 = buf65; del buf65  # reuse
    kernel_cpp_9(c_void_p(buf70.data_ptr()), c_void_p(relu_40.data_ptr()), c_void_p(convolution_43.data_ptr()), c_void_p(unsqueeze_322.data_ptr()), c_void_p(squeeze_130.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()))
    del convolution_43
    del primals_131
    del relu_40
    del squeeze_130
    del unsqueeze_322
    buf71 = aten.convolution_backward(buf70, relu_39, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf70
    del primals_130
    buf72 = buf71[0]
    assert_size_stride(buf72, (64, 1024, 2, 2), (4096, 4, 2, 1))
    buf73 = buf71[1]
    assert_size_stride(buf73, (512, 1024, 1, 1), (1024, 1, 1, 1))
    del buf71
    buf74 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf75 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf76 = empty_strided((64, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    buf77 = buf75; del buf75  # reuse
    kernel_cpp_10(c_void_p(buf77.data_ptr()), c_void_p(relu_39.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(unsqueeze_334.data_ptr()), c_void_p(squeeze_127.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf76.data_ptr()))
    del convolution_42
    del primals_128
    del squeeze_127
    del unsqueeze_334
    buf78 = aten.convolution_backward(buf76, relu_38, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_127
    buf79 = buf78[0]
    assert_size_stride(buf79, (64, 256, 2, 2), (1024, 4, 2, 1))
    buf80 = buf78[1]
    assert_size_stride(buf80, (1024, 256, 1, 1), (256, 1, 1, 1))
    del buf78
    buf81 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf82 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf83 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf84 = buf79; del buf79  # reuse
    kernel_cpp_11(c_void_p(buf84.data_ptr()), c_void_p(relu_38.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(unsqueeze_346.data_ptr()), c_void_p(squeeze_124.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()))
    del convolution_41
    del primals_125
    del relu_38
    del squeeze_124
    del unsqueeze_346
    buf85 = aten.convolution_backward(buf84, relu_37, primals_124, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf84
    del primals_124
    buf86 = buf85[0]
    assert_size_stride(buf86, (64, 256, 2, 2), (1024, 4, 2, 1))
    buf87 = buf85[1]
    assert_size_stride(buf87, (256, 256, 3, 3), (2304, 9, 3, 1))
    del buf85
    buf88 = buf82; del buf82  # reuse
    buf89 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf90 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf91 = buf86; del buf86  # reuse
    kernel_cpp_12(c_void_p(buf91.data_ptr()), c_void_p(relu_37.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(unsqueeze_358.data_ptr()), c_void_p(squeeze_121.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()))
    del convolution_40
    del primals_122
    del relu_37
    del squeeze_121
    del unsqueeze_358
    buf92 = aten.convolution_backward(buf91, relu_36, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf91
    del primals_121
    buf93 = buf92[0]
    assert_size_stride(buf93, (64, 1024, 2, 2), (4096, 4, 2, 1))
    buf94 = buf92[1]
    assert_size_stride(buf94, (256, 1024, 1, 1), (1024, 1, 1, 1))
    del buf92
    buf95 = buf52; del buf52  # reuse
    buf96 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf97 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf98 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf99 = buf76; del buf76  # reuse
    kernel_cpp_13(c_void_p(buf95.data_ptr()), c_void_p(relu_36.data_ptr()), c_void_p(relu_39.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(unsqueeze_370.data_ptr()), c_void_p(squeeze_118.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()))
    del buf72
    del buf93
    del convolution_39
    del primals_119
    del relu_36
    del relu_39
    del squeeze_118
    del unsqueeze_370
    buf100 = aten.convolution_backward(buf99, relu_35, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_118
    buf101 = buf100[0]
    assert_size_stride(buf101, (64, 256, 2, 2), (1024, 4, 2, 1))
    buf102 = buf100[1]
    assert_size_stride(buf102, (1024, 256, 1, 1), (256, 1, 1, 1))
    del buf100
    buf103 = buf89; del buf89  # reuse
    buf104 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf105 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf106 = buf101; del buf101  # reuse
    kernel_cpp_14(c_void_p(buf106.data_ptr()), c_void_p(relu_35.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(unsqueeze_382.data_ptr()), c_void_p(squeeze_115.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()))
    del convolution_38
    del primals_116
    del relu_35
    del squeeze_115
    del unsqueeze_382
    buf107 = aten.convolution_backward(buf106, relu_34, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf106
    del primals_115
    buf108 = buf107[0]
    assert_size_stride(buf108, (64, 256, 2, 2), (1024, 4, 2, 1))
    buf109 = buf107[1]
    assert_size_stride(buf109, (256, 256, 3, 3), (2304, 9, 3, 1))
    del buf107
    buf110 = buf104; del buf104  # reuse
    buf111 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf112 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf113 = buf108; del buf108  # reuse
    kernel_cpp_15(c_void_p(buf113.data_ptr()), c_void_p(relu_34.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(unsqueeze_394.data_ptr()), c_void_p(squeeze_112.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()))
    del convolution_37
    del primals_113
    del relu_34
    del squeeze_112
    del unsqueeze_394
    buf114 = aten.convolution_backward(buf113, relu_33, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf113
    del primals_112
    buf115 = buf114[0]
    assert_size_stride(buf115, (64, 1024, 2, 2), (4096, 4, 2, 1))
    buf116 = buf114[1]
    assert_size_stride(buf116, (256, 1024, 1, 1), (1024, 1, 1, 1))
    del buf114
    buf117 = buf97; del buf97  # reuse
    buf118 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf119 = buf99; del buf99  # reuse
    buf120 = buf118; del buf118  # reuse
    kernel_cpp_16(c_void_p(buf120.data_ptr()), c_void_p(relu_33.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(unsqueeze_406.data_ptr()), c_void_p(squeeze_109.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf119.data_ptr()))
    del convolution_36
    del primals_110
    del squeeze_109
    del unsqueeze_406
    buf121 = aten.convolution_backward(buf119, relu_32, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_109
    buf122 = buf121[0]
    assert_size_stride(buf122, (64, 256, 2, 2), (1024, 4, 2, 1))
    buf123 = buf121[1]
    assert_size_stride(buf123, (1024, 256, 1, 1), (256, 1, 1, 1))
    del buf121
    buf124 = buf111; del buf111  # reuse
    buf125 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf126 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf127 = buf122; del buf122  # reuse
    kernel_cpp_17(c_void_p(buf127.data_ptr()), c_void_p(relu_32.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(unsqueeze_418.data_ptr()), c_void_p(squeeze_106.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()))
    del convolution_35
    del primals_107
    del relu_32
    del squeeze_106
    del unsqueeze_418
    buf128 = aten.convolution_backward(buf127, relu_31, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf127
    del primals_106
    buf129 = buf128[0]
    assert_size_stride(buf129, (64, 256, 2, 2), (1024, 4, 2, 1))
    buf130 = buf128[1]
    assert_size_stride(buf130, (256, 256, 3, 3), (2304, 9, 3, 1))
    del buf128
    buf131 = buf125; del buf125  # reuse
    buf132 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf133 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf134 = buf129; del buf129  # reuse
    kernel_cpp_18(c_void_p(buf134.data_ptr()), c_void_p(relu_31.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(unsqueeze_430.data_ptr()), c_void_p(squeeze_103.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()))
    del convolution_34
    del primals_104
    del relu_31
    del squeeze_103
    del unsqueeze_430
    buf135 = aten.convolution_backward(buf134, relu_30, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf134
    del primals_103
    buf136 = buf135[0]
    assert_size_stride(buf136, (64, 1024, 2, 2), (4096, 4, 2, 1))
    buf137 = buf135[1]
    assert_size_stride(buf137, (256, 1024, 1, 1), (1024, 1, 1, 1))
    del buf135
    buf138 = buf115; del buf115  # reuse
    buf139 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf140 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf141 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf142 = buf119; del buf119  # reuse
    kernel_cpp_19(c_void_p(buf138.data_ptr()), c_void_p(relu_30.data_ptr()), c_void_p(relu_33.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(unsqueeze_442.data_ptr()), c_void_p(squeeze_100.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()))
    del buf136
    del convolution_33
    del primals_101
    del relu_30
    del relu_33
    del squeeze_100
    del unsqueeze_442
    buf143 = aten.convolution_backward(buf142, relu_29, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_100
    buf144 = buf143[0]
    assert_size_stride(buf144, (64, 256, 2, 2), (1024, 4, 2, 1))
    buf145 = buf143[1]
    assert_size_stride(buf145, (1024, 256, 1, 1), (256, 1, 1, 1))
    del buf143
    buf146 = buf132; del buf132  # reuse
    buf147 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf148 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf149 = buf144; del buf144  # reuse
    kernel_cpp_20(c_void_p(buf149.data_ptr()), c_void_p(relu_29.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(unsqueeze_454.data_ptr()), c_void_p(squeeze_97.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()))
    del convolution_32
    del primals_98
    del relu_29
    del squeeze_97
    del unsqueeze_454
    buf150 = aten.convolution_backward(buf149, relu_28, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf149
    del primals_97
    buf151 = buf150[0]
    assert_size_stride(buf151, (64, 256, 2, 2), (1024, 4, 2, 1))
    buf152 = buf150[1]
    assert_size_stride(buf152, (256, 256, 3, 3), (2304, 9, 3, 1))
    del buf150
    buf153 = buf147; del buf147  # reuse
    buf154 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf155 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf156 = buf151; del buf151  # reuse
    kernel_cpp_21(c_void_p(buf156.data_ptr()), c_void_p(relu_28.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(unsqueeze_466.data_ptr()), c_void_p(squeeze_94.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()))
    del convolution_31
    del primals_95
    del relu_28
    del squeeze_94
    del unsqueeze_466
    buf157 = aten.convolution_backward(buf156, relu_27, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf156
    del primals_94
    buf158 = buf157[0]
    assert_size_stride(buf158, (64, 1024, 2, 2), (4096, 4, 2, 1))
    buf159 = buf157[1]
    assert_size_stride(buf159, (256, 1024, 1, 1), (1024, 1, 1, 1))
    del buf157
    buf160 = buf140; del buf140  # reuse
    buf161 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf162 = buf142; del buf142  # reuse
    buf163 = buf161; del buf161  # reuse
    kernel_cpp_22(c_void_p(buf163.data_ptr()), c_void_p(relu_27.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(unsqueeze_478.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf162.data_ptr()))
    del convolution_30
    del primals_92
    del squeeze_91
    del unsqueeze_478
    buf164 = aten.convolution_backward(buf162, relu_26, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_91
    buf165 = buf164[0]
    assert_size_stride(buf165, (64, 256, 2, 2), (1024, 4, 2, 1))
    buf166 = buf164[1]
    assert_size_stride(buf166, (1024, 256, 1, 1), (256, 1, 1, 1))
    del buf164
    buf167 = buf154; del buf154  # reuse
    buf168 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf169 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf170 = buf165; del buf165  # reuse
    kernel_cpp_23(c_void_p(buf170.data_ptr()), c_void_p(relu_26.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(unsqueeze_490.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()))
    del convolution_29
    del primals_89
    del relu_26
    del squeeze_88
    del unsqueeze_490
    buf171 = aten.convolution_backward(buf170, relu_25, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf170
    del primals_88
    buf172 = buf171[0]
    assert_size_stride(buf172, (64, 256, 2, 2), (1024, 4, 2, 1))
    buf173 = buf171[1]
    assert_size_stride(buf173, (256, 256, 3, 3), (2304, 9, 3, 1))
    del buf171
    buf174 = buf168; del buf168  # reuse
    buf175 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf176 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf177 = buf172; del buf172  # reuse
    kernel_cpp_24(c_void_p(buf177.data_ptr()), c_void_p(relu_25.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(unsqueeze_502.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()))
    del convolution_28
    del primals_86
    del relu_25
    del squeeze_85
    del unsqueeze_502
    buf178 = aten.convolution_backward(buf177, relu_24, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf177
    del primals_85
    buf179 = buf178[0]
    assert_size_stride(buf179, (64, 1024, 2, 2), (4096, 4, 2, 1))
    buf180 = buf178[1]
    assert_size_stride(buf180, (256, 1024, 1, 1), (1024, 1, 1, 1))
    del buf178
    buf181 = buf138; del buf138  # reuse
    buf182 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf183 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf189 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf184 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf185 = buf162; del buf162  # reuse
    buf191 = buf95; del buf95  # reuse
    kernel_cpp_25(c_void_p(buf181.data_ptr()), c_void_p(relu_24.data_ptr()), c_void_p(relu_27.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(unsqueeze_514.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(unsqueeze_526.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf191.data_ptr()))
    del buf158
    del buf179
    del buf181
    del buf183
    del convolution_26
    del convolution_27
    del primals_80
    del primals_83
    del relu_24
    del relu_27
    del squeeze_82
    del unsqueeze_514
    del unsqueeze_526
    buf186 = aten.convolution_backward(buf185, relu_21, primals_82, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf185
    del primals_82
    buf187 = buf186[0]
    assert_size_stride(buf187, (64, 512, 4, 4), (8192, 16, 4, 1))
    buf188 = buf186[1]
    assert_size_stride(buf188, (1024, 512, 1, 1), (512, 1, 1, 1))
    del buf186
    buf190 = buf189; del buf189  # reuse
    kernel_cpp_26(c_void_p(buf190.data_ptr()), c_void_p(squeeze_79.data_ptr()))
    del squeeze_79
    buf192 = aten.convolution_backward(buf191, relu_23, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf191
    del primals_79
    buf193 = buf192[0]
    assert_size_stride(buf193, (64, 256, 2, 2), (1024, 4, 2, 1))
    buf194 = buf192[1]
    assert_size_stride(buf194, (1024, 256, 1, 1), (256, 1, 1, 1))
    del buf192
    buf195 = buf175; del buf175  # reuse
    buf196 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf197 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf198 = buf193; del buf193  # reuse
    kernel_cpp_27(c_void_p(buf198.data_ptr()), c_void_p(relu_23.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(unsqueeze_538.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()))
    del convolution_25
    del primals_77
    del relu_23
    del squeeze_76
    del unsqueeze_538
    buf199 = aten.convolution_backward(buf198, relu_22, primals_76, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf198
    del primals_76
    buf200 = buf199[0]
    assert_size_stride(buf200, (64, 256, 4, 4), (4096, 16, 4, 1))
    buf201 = buf199[1]
    assert_size_stride(buf201, (256, 256, 3, 3), (2304, 9, 3, 1))
    del buf199
    buf202 = buf196; del buf196  # reuse
    buf203 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf204 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf205 = buf200; del buf200  # reuse
    kernel_cpp_28(c_void_p(buf205.data_ptr()), c_void_p(relu_22.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(unsqueeze_550.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()))
    del convolution_24
    del primals_74
    del relu_22
    del squeeze_73
    del unsqueeze_550
    buf206 = aten.convolution_backward(buf205, relu_21, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf205
    del primals_73
    buf207 = buf206[0]
    assert_size_stride(buf207, (64, 512, 4, 4), (8192, 16, 4, 1))
    buf208 = buf206[1]
    assert_size_stride(buf208, (256, 512, 1, 1), (512, 1, 1, 1))
    del buf206
    buf209 = buf68; del buf68  # reuse
    buf210 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf211 = empty_strided((64, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    buf212 = buf210; del buf210  # reuse
    kernel_cpp_29(c_void_p(buf212.data_ptr()), c_void_p(relu_21.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(unsqueeze_562.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf211.data_ptr()))
    del convolution_23
    del primals_71
    del squeeze_70
    del unsqueeze_562
    buf213 = aten.convolution_backward(buf211, relu_20, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_70
    buf214 = buf213[0]
    assert_size_stride(buf214, (64, 128, 4, 4), (2048, 16, 4, 1))
    buf215 = buf213[1]
    assert_size_stride(buf215, (512, 128, 1, 1), (128, 1, 1, 1))
    del buf213
    buf216 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf217 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf218 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf219 = buf214; del buf214  # reuse
    kernel_cpp_30(c_void_p(buf219.data_ptr()), c_void_p(relu_20.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(unsqueeze_574.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()))
    del convolution_22
    del primals_68
    del relu_20
    del squeeze_67
    del unsqueeze_574
    buf220 = aten.convolution_backward(buf219, relu_19, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf219
    del primals_67
    buf221 = buf220[0]
    assert_size_stride(buf221, (64, 128, 4, 4), (2048, 16, 4, 1))
    buf222 = buf220[1]
    assert_size_stride(buf222, (128, 128, 3, 3), (1152, 9, 3, 1))
    del buf220
    buf223 = buf217; del buf217  # reuse
    buf224 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf225 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf226 = buf221; del buf221  # reuse
    kernel_cpp_31(c_void_p(buf226.data_ptr()), c_void_p(relu_19.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(unsqueeze_586.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()))
    del convolution_21
    del primals_65
    del relu_19
    del squeeze_64
    del unsqueeze_586
    buf227 = aten.convolution_backward(buf226, relu_18, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf226
    del primals_64
    buf228 = buf227[0]
    assert_size_stride(buf228, (64, 512, 4, 4), (8192, 16, 4, 1))
    buf229 = buf227[1]
    assert_size_stride(buf229, (128, 512, 1, 1), (512, 1, 1, 1))
    del buf227
    buf230 = buf187; del buf187  # reuse
    buf231 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf232 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf233 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf234 = buf211; del buf211  # reuse
    kernel_cpp_32(c_void_p(buf230.data_ptr()), c_void_p(relu_18.data_ptr()), c_void_p(relu_21.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(unsqueeze_598.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf234.data_ptr()))
    del buf207
    del convolution_20
    del primals_62
    del relu_18
    del relu_21
    del squeeze_61
    del unsqueeze_598
    buf235 = aten.convolution_backward(buf234, relu_17, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_61
    buf236 = buf235[0]
    assert_size_stride(buf236, (64, 128, 4, 4), (2048, 16, 4, 1))
    buf237 = buf235[1]
    assert_size_stride(buf237, (512, 128, 1, 1), (128, 1, 1, 1))
    del buf235
    buf238 = buf224; del buf224  # reuse
    buf239 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf240 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf241 = buf236; del buf236  # reuse
    kernel_cpp_33(c_void_p(buf241.data_ptr()), c_void_p(relu_17.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(unsqueeze_610.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()))
    del convolution_19
    del primals_59
    del relu_17
    del squeeze_58
    del unsqueeze_610
    buf242 = aten.convolution_backward(buf241, relu_16, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf241
    del primals_58
    buf243 = buf242[0]
    assert_size_stride(buf243, (64, 128, 4, 4), (2048, 16, 4, 1))
    buf244 = buf242[1]
    assert_size_stride(buf244, (128, 128, 3, 3), (1152, 9, 3, 1))
    del buf242
    buf245 = buf239; del buf239  # reuse
    buf246 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf247 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf248 = buf243; del buf243  # reuse
    kernel_cpp_34(c_void_p(buf248.data_ptr()), c_void_p(relu_16.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(unsqueeze_622.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()))
    del convolution_18
    del primals_56
    del relu_16
    del squeeze_55
    del unsqueeze_622
    buf249 = aten.convolution_backward(buf248, relu_15, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf248
    del primals_55
    buf250 = buf249[0]
    assert_size_stride(buf250, (64, 512, 4, 4), (8192, 16, 4, 1))
    buf251 = buf249[1]
    assert_size_stride(buf251, (128, 512, 1, 1), (512, 1, 1, 1))
    del buf249
    buf252 = buf232; del buf232  # reuse
    buf253 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf254 = buf234; del buf234  # reuse
    buf255 = buf253; del buf253  # reuse
    kernel_cpp_35(c_void_p(buf255.data_ptr()), c_void_p(relu_15.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(unsqueeze_634.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf254.data_ptr()))
    del convolution_17
    del primals_53
    del squeeze_52
    del unsqueeze_634
    buf256 = aten.convolution_backward(buf254, relu_14, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_52
    buf257 = buf256[0]
    assert_size_stride(buf257, (64, 128, 4, 4), (2048, 16, 4, 1))
    buf258 = buf256[1]
    assert_size_stride(buf258, (512, 128, 1, 1), (128, 1, 1, 1))
    del buf256
    buf259 = buf246; del buf246  # reuse
    buf260 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf261 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf262 = buf257; del buf257  # reuse
    kernel_cpp_36(c_void_p(buf262.data_ptr()), c_void_p(relu_14.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(unsqueeze_646.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()))
    del convolution_16
    del primals_50
    del relu_14
    del squeeze_49
    del unsqueeze_646
    buf263 = aten.convolution_backward(buf262, relu_13, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf262
    del primals_49
    buf264 = buf263[0]
    assert_size_stride(buf264, (64, 128, 4, 4), (2048, 16, 4, 1))
    buf265 = buf263[1]
    assert_size_stride(buf265, (128, 128, 3, 3), (1152, 9, 3, 1))
    del buf263
    buf266 = buf260; del buf260  # reuse
    buf267 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf268 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf269 = buf264; del buf264  # reuse
    kernel_cpp_37(c_void_p(buf269.data_ptr()), c_void_p(relu_13.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_658.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()))
    del convolution_15
    del primals_47
    del relu_13
    del squeeze_46
    del unsqueeze_658
    buf270 = aten.convolution_backward(buf269, relu_12, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf269
    del primals_46
    buf271 = buf270[0]
    assert_size_stride(buf271, (64, 512, 4, 4), (8192, 16, 4, 1))
    buf272 = buf270[1]
    assert_size_stride(buf272, (128, 512, 1, 1), (512, 1, 1, 1))
    del buf270
    buf273 = buf230; del buf230  # reuse
    buf274 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf275 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf281 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf276 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf277 = buf254; del buf254  # reuse
    buf283 = buf228; del buf228  # reuse
    kernel_cpp_38(c_void_p(buf273.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(relu_15.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(unsqueeze_670.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(unsqueeze_682.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf283.data_ptr()))
    del buf250
    del buf271
    del buf273
    del buf275
    del convolution_13
    del convolution_14
    del primals_41
    del primals_44
    del relu_12
    del relu_15
    del squeeze_43
    del unsqueeze_670
    del unsqueeze_682
    buf278 = aten.convolution_backward(buf277, relu_9, primals_43, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf277
    del primals_43
    buf279 = buf278[0]
    assert_size_stride(buf279, (64, 256, 8, 8), (16384, 64, 8, 1))
    buf280 = buf278[1]
    assert_size_stride(buf280, (512, 256, 1, 1), (256, 1, 1, 1))
    del buf278
    buf282 = buf281; del buf281  # reuse
    kernel_cpp_39(c_void_p(buf282.data_ptr()), c_void_p(squeeze_40.data_ptr()))
    del squeeze_40
    buf284 = aten.convolution_backward(buf283, relu_11, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf283
    del primals_40
    buf285 = buf284[0]
    assert_size_stride(buf285, (64, 128, 4, 4), (2048, 16, 4, 1))
    buf286 = buf284[1]
    assert_size_stride(buf286, (512, 128, 1, 1), (128, 1, 1, 1))
    del buf284
    buf287 = buf267; del buf267  # reuse
    buf288 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf289 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf290 = buf285; del buf285  # reuse
    kernel_cpp_40(c_void_p(buf290.data_ptr()), c_void_p(relu_11.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(unsqueeze_694.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()))
    del convolution_12
    del primals_38
    del relu_11
    del squeeze_37
    del unsqueeze_694
    buf291 = aten.convolution_backward(buf290, relu_10, primals_37, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf290
    del primals_37
    buf292 = buf291[0]
    assert_size_stride(buf292, (64, 128, 8, 8), (8192, 64, 8, 1))
    buf293 = buf291[1]
    assert_size_stride(buf293, (128, 128, 3, 3), (1152, 9, 3, 1))
    del buf291
    buf294 = buf288; del buf288  # reuse
    buf295 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf296 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf297 = buf292; del buf292  # reuse
    kernel_cpp_41(c_void_p(buf297.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(unsqueeze_706.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()))
    del buf295
    del convolution_11
    del primals_35
    del relu_10
    del squeeze_34
    del unsqueeze_706
    buf298 = aten.convolution_backward(buf297, relu_9, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf297
    del primals_34
    buf299 = buf298[0]
    assert_size_stride(buf299, (64, 256, 8, 8), (16384, 64, 8, 1))
    buf300 = buf298[1]
    assert_size_stride(buf300, (128, 256, 1, 1), (256, 1, 1, 1))
    del buf298
    buf301 = buf203; del buf203  # reuse
    buf302 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf303 = empty_strided((64, 256, 8, 8), (16384, 64, 8, 1), device='cpu', dtype=torch.float32)
    buf304 = buf302; del buf302  # reuse
    kernel_cpp_42(c_void_p(buf304.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(unsqueeze_718.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf303.data_ptr()))
    del convolution_10
    del primals_32
    del squeeze_31
    del unsqueeze_718
    buf305 = aten.convolution_backward(buf303, relu_8, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_31
    buf306 = buf305[0]
    assert_size_stride(buf306, (64, 64, 8, 8), (4096, 64, 8, 1))
    buf307 = buf305[1]
    assert_size_stride(buf307, (256, 64, 1, 1), (64, 1, 1, 1))
    del buf305
    buf308 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf309 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf310 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf311 = buf306; del buf306  # reuse
    kernel_cpp_43(c_void_p(buf311.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_730.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()))
    del convolution_9
    del primals_29
    del relu_8
    del squeeze_28
    del unsqueeze_730
    buf312 = aten.convolution_backward(buf311, relu_7, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf311
    del primals_28
    buf313 = buf312[0]
    assert_size_stride(buf313, (64, 64, 8, 8), (4096, 64, 8, 1))
    buf314 = buf312[1]
    assert_size_stride(buf314, (64, 64, 3, 3), (576, 9, 3, 1))
    del buf312
    buf315 = buf309; del buf309  # reuse
    buf316 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf317 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf318 = buf313; del buf313  # reuse
    kernel_cpp_44(c_void_p(buf318.data_ptr()), c_void_p(relu_7.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(unsqueeze_742.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()))
    del convolution_8
    del primals_26
    del relu_7
    del squeeze_25
    del unsqueeze_742
    buf319 = aten.convolution_backward(buf318, relu_6, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf318
    del primals_25
    buf320 = buf319[0]
    assert_size_stride(buf320, (64, 256, 8, 8), (16384, 64, 8, 1))
    buf321 = buf319[1]
    assert_size_stride(buf321, (64, 256, 1, 1), (256, 1, 1, 1))
    del buf319
    buf322 = buf279; del buf279  # reuse
    buf323 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf324 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf325 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf326 = buf303; del buf303  # reuse
    kernel_cpp_45(c_void_p(buf322.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(unsqueeze_754.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()))
    del buf299
    del convolution_7
    del primals_23
    del relu_6
    del relu_9
    del squeeze_22
    del unsqueeze_754
    buf327 = aten.convolution_backward(buf326, relu_5, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_22
    buf328 = buf327[0]
    assert_size_stride(buf328, (64, 64, 8, 8), (4096, 64, 8, 1))
    buf329 = buf327[1]
    assert_size_stride(buf329, (256, 64, 1, 1), (64, 1, 1, 1))
    del buf327
    buf330 = buf316; del buf316  # reuse
    buf331 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf332 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf333 = buf328; del buf328  # reuse
    kernel_cpp_46(c_void_p(buf333.data_ptr()), c_void_p(relu_5.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(unsqueeze_766.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()))
    del convolution_6
    del primals_20
    del relu_5
    del squeeze_19
    del unsqueeze_766
    buf334 = aten.convolution_backward(buf333, relu_4, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf333
    del primals_19
    buf335 = buf334[0]
    assert_size_stride(buf335, (64, 64, 8, 8), (4096, 64, 8, 1))
    buf336 = buf334[1]
    assert_size_stride(buf336, (64, 64, 3, 3), (576, 9, 3, 1))
    del buf334
    buf337 = buf331; del buf331  # reuse
    buf338 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf339 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf340 = buf335; del buf335  # reuse
    kernel_cpp_47(c_void_p(buf340.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(unsqueeze_778.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()))
    del convolution_5
    del primals_17
    del relu_4
    del squeeze_16
    del unsqueeze_778
    buf341 = aten.convolution_backward(buf340, relu_3, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf340
    del primals_16
    buf342 = buf341[0]
    assert_size_stride(buf342, (64, 256, 8, 8), (16384, 64, 8, 1))
    buf343 = buf341[1]
    assert_size_stride(buf343, (64, 256, 1, 1), (256, 1, 1, 1))
    del buf341
    buf344 = buf324; del buf324  # reuse
    buf345 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf351 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf346 = buf326; del buf326  # reuse
    buf352 = buf320; del buf320  # reuse
    buf347 = buf345; del buf345  # reuse
    kernel_cpp_48(c_void_p(buf347.data_ptr()), c_void_p(relu_3.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_790.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(unsqueeze_802.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf352.data_ptr()))
    del buf322
    del buf342
    del convolution_3
    del convolution_4
    del primals_11
    del primals_14
    del relu_3
    del squeeze_13
    del unsqueeze_790
    del unsqueeze_802
    buf348 = aten.convolution_backward(buf346, getitem_2, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf346
    del primals_13
    buf349 = buf348[0]
    assert_size_stride(buf349, (64, 64, 8, 8), (4096, 64, 8, 1))
    buf350 = buf348[1]
    assert_size_stride(buf350, (256, 64, 1, 1), (64, 1, 1, 1))
    del buf348
    buf353 = buf351; del buf351  # reuse
    kernel_cpp_49(c_void_p(buf353.data_ptr()), c_void_p(squeeze_10.data_ptr()))
    del squeeze_10
    buf354 = aten.convolution_backward(buf352, relu_2, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_10
    buf355 = buf354[0]
    assert_size_stride(buf355, (64, 64, 8, 8), (4096, 64, 8, 1))
    buf356 = buf354[1]
    assert_size_stride(buf356, (256, 64, 1, 1), (64, 1, 1, 1))
    del buf354
    buf357 = buf338; del buf338  # reuse
    buf358 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf359 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf360 = buf355; del buf355  # reuse
    kernel_cpp_50(c_void_p(buf360.data_ptr()), c_void_p(relu_2.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_814.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()))
    del convolution_2
    del primals_8
    del relu_2
    del squeeze_7
    del unsqueeze_814
    buf361 = aten.convolution_backward(buf360, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf360
    del primals_7
    buf362 = buf361[0]
    assert_size_stride(buf362, (64, 64, 8, 8), (4096, 64, 8, 1))
    buf363 = buf361[1]
    assert_size_stride(buf363, (64, 64, 3, 3), (576, 9, 3, 1))
    del buf361
    buf364 = buf358; del buf358  # reuse
    buf365 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf366 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf367 = buf362; del buf362  # reuse
    kernel_cpp_51(c_void_p(buf367.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_826.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf366.data_ptr()))
    del convolution_1
    del primals_5
    del relu_1
    del squeeze_4
    del unsqueeze_826
    buf368 = aten.convolution_backward(buf367, getitem_2, primals_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf367
    del getitem_2
    del primals_4
    buf369 = buf368[0]
    assert_size_stride(buf369, (64, 64, 8, 8), (4096, 64, 8, 1))
    buf370 = buf368[1]
    assert_size_stride(buf370, (64, 64, 1, 1), (64, 1, 1, 1))
    del buf368
    buf371 = buf349; del buf349  # reuse
    buf372 = as_strided(buf352, (64, 64, 16, 16), (16384, 256, 16, 1)); del buf352  # reuse
    buf373 = buf365; del buf365  # reuse
    buf374 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf375 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf376 = buf372; del buf372  # reuse
    kernel_cpp_52(c_void_p(buf371.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(getitem_3.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_838.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()))
    del buf369
    del buf371
    del buf374
    del convolution
    del getitem_3
    del primals_2
    del relu
    del squeeze_1
    del unsqueeze_838
    buf377 = aten.convolution_backward(buf376, primals_321, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf376
    del primals_1
    del primals_321
    buf378 = buf377[1]
    assert_size_stride(buf378, (64, 3, 7, 7), (147, 49, 7, 1))
    del buf377
    return (buf378, buf375, buf373, buf370, buf366, buf364, buf363, buf359, buf357, buf356, buf353, buf344, buf350, buf347, buf344, buf343, buf339, buf337, buf336, buf332, buf330, buf329, buf325, buf323, buf321, buf317, buf315, buf314, buf310, buf308, buf307, buf304, buf301, buf300, buf296, buf294, buf293, buf289, buf287, buf286, buf282, buf274, buf280, buf276, buf274, buf272, buf268, buf266, buf265, buf261, buf259, buf258, buf255, buf252, buf251, buf247, buf245, buf244, buf240, buf238, buf237, buf233, buf231, buf229, buf225, buf223, buf222, buf218, buf216, buf215, buf212, buf209, buf208, buf204, buf202, buf201, buf197, buf195, buf194, buf190, buf182, buf188, buf184, buf182, buf180, buf176, buf174, buf173, buf169, buf167, buf166, buf163, buf160, buf159, buf155, buf153, buf152, buf148, buf146, buf145, buf141, buf139, buf137, buf133, buf131, buf130, buf126, buf124, buf123, buf120, buf117, buf116, buf112, buf110, buf109, buf105, buf103, buf102, buf98, buf96, buf94, buf90, buf88, buf87, buf83, buf81, buf80, buf77, buf74, buf73, buf69, buf67, buf66, buf62, buf60, buf59, buf55, buf47, buf53, buf49, buf47, buf45, buf41, buf39, buf38, buf34, buf32, buf31, buf27, buf24, buf23, buf19, buf17, buf16, buf12, buf10, buf9, buf5, buf3, as_strided(buf1, (1000, 2048), (2048, 1)), as_strided(buf2, (1000, ), (1, )), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_321 = rand_strided((64, 3, 32, 32), (3072, 1024, 32, 1), device='cpu', dtype=torch.float32)
    convolution = rand_strided((64, 64, 16, 16), (16384, 256, 16, 1), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu = rand_strided((64, 64, 16, 16), (16384, 256, 16, 1), device='cpu', dtype=torch.float32)
    getitem_2 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.int64)
    convolution_1 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((64, 256, 8, 8), (16384, 64, 8, 1), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((64, 256, 8, 8), (16384, 64, 8, 1), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((64, 256, 8, 8), (16384, 64, 8, 1), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((64, 256, 8, 8), (16384, 64, 8, 1), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((64, 256, 8, 8), (16384, 64, 8, 1), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_7 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((64, 256, 8, 8), (16384, 64, 8, 1), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((64, 256, 8, 8), (16384, 64, 8, 1), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((64, 128, 8, 8), (8192, 64, 8, 1), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((64, 128, 8, 8), (8192, 64, 8, 1), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((64, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_11 = rand_strided((64, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((64, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((64, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_12 = rand_strided((64, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((64, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_13 = rand_strided((64, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((64, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_14 = rand_strided((64, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((64, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_15 = rand_strided((64, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((64, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_16 = rand_strided((64, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((64, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_17 = rand_strided((64, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((64, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_18 = rand_strided((64, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((64, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_19 = rand_strided((64, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((64, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_20 = rand_strided((64, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((64, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_21 = rand_strided((64, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((64, 256, 4, 4), (4096, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_22 = rand_strided((64, 256, 4, 4), (4096, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((64, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_23 = rand_strided((64, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((64, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((64, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    relu_24 = rand_strided((64, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((64, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_25 = rand_strided((64, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((64, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_26 = rand_strided((64, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((64, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    relu_27 = rand_strided((64, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((64, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_94 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_28 = rand_strided((64, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((64, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_97 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_29 = rand_strided((64, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((64, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_100 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    relu_30 = rand_strided((64, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((64, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_103 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_31 = rand_strided((64, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((64, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_106 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_32 = rand_strided((64, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((64, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_109 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    relu_33 = rand_strided((64, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((64, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_112 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_34 = rand_strided((64, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((64, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_115 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_35 = rand_strided((64, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((64, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_118 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    relu_36 = rand_strided((64, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((64, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_121 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_37 = rand_strided((64, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((64, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_124 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_38 = rand_strided((64, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((64, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_127 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    relu_39 = rand_strided((64, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((64, 512, 2, 2), (2048, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_130 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_40 = rand_strided((64, 512, 2, 2), (2048, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((64, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    squeeze_133 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_41 = rand_strided((64, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((64, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    squeeze_136 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((64, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    squeeze_139 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    relu_42 = rand_strided((64, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((64, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    squeeze_142 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_43 = rand_strided((64, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((64, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    squeeze_145 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_44 = rand_strided((64, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((64, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    squeeze_148 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    relu_45 = rand_strided((64, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((64, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    squeeze_151 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_46 = rand_strided((64, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((64, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    squeeze_154 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_47 = rand_strided((64, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    convolution_52 = rand_strided((64, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    squeeze_157 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    view = rand_strided((64, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le = rand_strided((64, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.bool)
    unsqueeze_214 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_226 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_238 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_250 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_262 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_274 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_286 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_298 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_310 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_322 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_334 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_358 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_370 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_382 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_394 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_406 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_418 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_430 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_442 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_454 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_466 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_478 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_490 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_502 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_514 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_526 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_538 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_550 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_562 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_574 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_586 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_598 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_610 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_622 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_634 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_646 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_658 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_670 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_682 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_694 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_706 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_718 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_730 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_742 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_754 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_766 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_778 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_790 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_802 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_814 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_826 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_838 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_2 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_3 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_4 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_5 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_6 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_7 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_8 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_9 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_10 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_11 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_12 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_13 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_14 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_15 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_16 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_17 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_18 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_19 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_20 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_21 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_22 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_23 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_24 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_25 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_26 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_27 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_28 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_29 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_30 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_31 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_32 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_33 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_34 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_35 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_36 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_37 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_38 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_39 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_40 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_41 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_42 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_43 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_44 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_45 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_46 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_47 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_48 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_49 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_50 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_51 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_52 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_53 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_54 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_55 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_56 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_57 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_58 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_59 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_60 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_61 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_62 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_63 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_64 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_65 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_66 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_67 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_68 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_69 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_70 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_71 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_72 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_73 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_74 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_75 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_76 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_77 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_78 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_79 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_80 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_81 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_82 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_83 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_84 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_85 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_86 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_87 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_88 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_89 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_90 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_91 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_92 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_93 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_94 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_95 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_96 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_97 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_98 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_99 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_100 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_101 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_102 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_103 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_104 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_105 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_106 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_107 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_108 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_109 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_110 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_111 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_112 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_113 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_114 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_115 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_116 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_117 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_118 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_119 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_120 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_121 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_122 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_123 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_124 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_125 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_126 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_127 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_128 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_129 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_130 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_131 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_132 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_133 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_134 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_135 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_136 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_137 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_138 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_139 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_140 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_141 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_142 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_143 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_144 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_145 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_146 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_147 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_148 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_149 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_150 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_151 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_152 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_153 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_154 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_155 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_156 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_157 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_158 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_159 = rand_strided((), (), device='cpu', dtype=torch.int64)
    tangents_160 = rand_strided((64, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_321, convolution, squeeze_1, relu, getitem_2, getitem_3, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, relu_4, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, relu_8, convolution_10, squeeze_31, relu_9, convolution_11, squeeze_34, relu_10, convolution_12, squeeze_37, relu_11, convolution_13, squeeze_40, convolution_14, squeeze_43, relu_12, convolution_15, squeeze_46, relu_13, convolution_16, squeeze_49, relu_14, convolution_17, squeeze_52, relu_15, convolution_18, squeeze_55, relu_16, convolution_19, squeeze_58, relu_17, convolution_20, squeeze_61, relu_18, convolution_21, squeeze_64, relu_19, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, relu_21, convolution_24, squeeze_73, relu_22, convolution_25, squeeze_76, relu_23, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_24, convolution_28, squeeze_85, relu_25, convolution_29, squeeze_88, relu_26, convolution_30, squeeze_91, relu_27, convolution_31, squeeze_94, relu_28, convolution_32, squeeze_97, relu_29, convolution_33, squeeze_100, relu_30, convolution_34, squeeze_103, relu_31, convolution_35, squeeze_106, relu_32, convolution_36, squeeze_109, relu_33, convolution_37, squeeze_112, relu_34, convolution_38, squeeze_115, relu_35, convolution_39, squeeze_118, relu_36, convolution_40, squeeze_121, relu_37, convolution_41, squeeze_124, relu_38, convolution_42, squeeze_127, relu_39, convolution_43, squeeze_130, relu_40, convolution_44, squeeze_133, relu_41, convolution_45, squeeze_136, convolution_46, squeeze_139, relu_42, convolution_47, squeeze_142, relu_43, convolution_48, squeeze_145, relu_44, convolution_49, squeeze_148, relu_45, convolution_50, squeeze_151, relu_46, convolution_51, squeeze_154, relu_47, convolution_52, squeeze_157, view, permute_1, le, unsqueeze_214, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50, tangents_51, tangents_52, tangents_53, tangents_54, tangents_55, tangents_56, tangents_57, tangents_58, tangents_59, tangents_60, tangents_61, tangents_62, tangents_63, tangents_64, tangents_65, tangents_66, tangents_67, tangents_68, tangents_69, tangents_70, tangents_71, tangents_72, tangents_73, tangents_74, tangents_75, tangents_76, tangents_77, tangents_78, tangents_79, tangents_80, tangents_81, tangents_82, tangents_83, tangents_84, tangents_85, tangents_86, tangents_87, tangents_88, tangents_89, tangents_90, tangents_91, tangents_92, tangents_93, tangents_94, tangents_95, tangents_96, tangents_97, tangents_98, tangents_99, tangents_100, tangents_101, tangents_102, tangents_103, tangents_104, tangents_105, tangents_106, tangents_107, tangents_108, tangents_109, tangents_110, tangents_111, tangents_112, tangents_113, tangents_114, tangents_115, tangents_116, tangents_117, tangents_118, tangents_119, tangents_120, tangents_121, tangents_122, tangents_123, tangents_124, tangents_125, tangents_126, tangents_127, tangents_128, tangents_129, tangents_130, tangents_131, tangents_132, tangents_133, tangents_134, tangents_135, tangents_136, tangents_137, tangents_138, tangents_139, tangents_140, tangents_141, tangents_142, tangents_143, tangents_144, tangents_145, tangents_146, tangents_147, tangents_148, tangents_149, tangents_150, tangents_151, tangents_152, tangents_153, tangents_154, tangents_155, tangents_156, tangents_157, tangents_158, tangents_159, tangents_160]))
