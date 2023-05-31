
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
                       float* __restrict__ in_out_ptr2,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    auto out_ptr2 = in_out_ptr2;
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
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            tmp2.store(in_out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            in_out_ptr0[i0] = tmp2;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<2048; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr1[i0 + (2048*i1)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<128; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            tmp2.store(in_out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = out_ptr1[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            in_out_ptr1[i0] = tmp2;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<2048; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr2[i0 + (2048*i1)];
                    tmp1 += tmp0;
                }
                out_ptr2[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<128; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            tmp2.store(in_out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            in_out_ptr2[i0] = tmp2;
        }
    }
}
''')


kernel_cpp_1 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       float* __restrict__ in_out_ptr2,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const bool* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5,
                       float* __restrict__ out_ptr6)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    auto out_ptr2 = in_out_ptr2;
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
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            tmp2.store(in_out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            in_out_ptr0[i0] = tmp2;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr1[i0 + (512*i1)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            tmp2.store(in_out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr1[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            in_out_ptr1[i0] = tmp2;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<2048; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr2[i0 + (2048*i1)];
                    tmp1 += tmp0;
                }
                out_ptr2[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<128; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            tmp2.store(in_out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            in_out_ptr2[i0] = tmp2;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<2048; i0+=1)
        {
            {
                float tmp4 = 0;
                float tmp11 = 0;
                float tmp13 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr2[i0 + (2048*i1)];
                    auto tmp1 = in_out_ptr2[i0];
                    auto tmp5 = in_ptr3[i0 + (2048*i1)];
                    auto tmp7 = in_ptr4[i0 + (2048*i1)];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    auto tmp6 = static_cast<float>(0.0);
                    auto tmp8 = static_cast<float>(1);
                    auto tmp9 = tmp7 / tmp8;
                    auto tmp10 = tmp5 ? tmp6 : tmp9;
                    auto tmp12 = tmp10 * tmp2;
                    tmp4 += tmp3;
                    tmp11 += tmp10;
                    tmp13 += tmp12;
                }
                out_ptr3[i0] = tmp4;
                out_ptr4[i0] = tmp11;
                out_ptr5[i0] = tmp13;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            for(long i1=0; i1<128; i1+=1)
            {
                float g_tmp_buffer_in_ptr3[16] = {0};
                flag_to_float(in_ptr3 + (16*i1) + (2048*i0), g_tmp_buffer_in_ptr3, 16);
                auto tmp0 = at::vec::Vectorized<float>::loadu(g_tmp_buffer_in_ptr3);
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr4 + (16*i1) + (2048*i0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (2048*i0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + 16*i1);
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr5 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr3 + 16*i1);
                auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr4 + 16*i1);
                auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + 16*i1);
                auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(1));
                auto tmp4 = tmp2 / tmp3;
                auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.125));
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = at::vec::Vectorized<float>(static_cast<float>(8));
                auto tmp14 = tmp12 / tmp13;
                auto tmp15 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = tmp16.rsqrt();
                auto tmp18 = tmp17 * tmp17;
                auto tmp19 = tmp11 * tmp18;
                auto tmp20 = tmp8 * tmp19;
                auto tmp21 = tmp5 - tmp20;
                auto tmp23 = tmp22 * tmp10;
                auto tmp24 = tmp21 - tmp23;
                auto tmp26 = tmp17 * tmp25;
                auto tmp27 = tmp24 * tmp26;
                tmp27.store(out_ptr6 + (16*i1) + (2048*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=2048; i1<2048; i1+=1)
            {
                auto tmp0 = in_ptr3[i1 + (2048*i0)];
                auto tmp2 = in_ptr4[i1 + (2048*i0)];
                auto tmp6 = in_ptr2[i1 + (2048*i0)];
                auto tmp7 = in_out_ptr2[i1];
                auto tmp9 = out_ptr5[i1];
                auto tmp12 = out_ptr3[i1];
                auto tmp22 = out_ptr4[i1];
                auto tmp25 = in_ptr5[i1];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp3 = static_cast<float>(1);
                auto tmp4 = tmp2 / tmp3;
                auto tmp5 = tmp0 ? tmp1 : tmp4;
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.125);
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = static_cast<float>(8);
                auto tmp14 = tmp12 / tmp13;
                auto tmp15 = static_cast<float>(1e-05);
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = 1 / std::sqrt(tmp16);
                auto tmp18 = tmp17 * tmp17;
                auto tmp19 = tmp11 * tmp18;
                auto tmp20 = tmp8 * tmp19;
                auto tmp21 = tmp5 - tmp20;
                auto tmp23 = tmp22 * tmp10;
                auto tmp24 = tmp21 - tmp23;
                auto tmp26 = tmp17 * tmp25;
                auto tmp27 = tmp24 * tmp26;
                out_ptr6[i1 + (2048*i0)] = tmp27;
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
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr3 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp4 = 0;
                float tmp11 = 0;
                float tmp13 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    auto tmp1 = in_ptr1[i0];
                    auto tmp5 = in_ptr2[i0 + (512*i1)];
                    auto tmp9 = in_ptr3[i0 + (512*i1)];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    auto tmp6 = static_cast<float>(0);
                    auto tmp7 = tmp5 <= tmp6;
                    auto tmp8 = static_cast<float>(0.0);
                    auto tmp10 = tmp7 ? tmp8 : tmp9;
                    auto tmp12 = tmp10 * tmp2;
                    tmp4 += tmp3;
                    tmp11 += tmp10;
                    tmp13 += tmp12;
                }
                out_ptr0[i0] = tmp4;
                out_ptr1[i0] = tmp11;
                out_ptr2[i0] = tmp13;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (512*i0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i1) + (512*i0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (512*i0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i1);
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i1);
                auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i1);
                auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i1);
                auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.125));
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = at::vec::Vectorized<float>(static_cast<float>(8));
                auto tmp14 = tmp12 / tmp13;
                auto tmp15 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = tmp16.rsqrt();
                auto tmp18 = tmp17 * tmp17;
                auto tmp19 = tmp11 * tmp18;
                auto tmp20 = tmp8 * tmp19;
                auto tmp21 = tmp5 - tmp20;
                auto tmp23 = tmp22 * tmp10;
                auto tmp24 = tmp21 - tmp23;
                auto tmp26 = tmp17 * tmp25;
                auto tmp27 = tmp24 * tmp26;
                tmp27.store(in_out_ptr0 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = in_ptr2[i1 + (512*i0)];
                auto tmp4 = in_out_ptr0[i1 + (512*i0)];
                auto tmp6 = in_ptr0[i1 + (512*i0)];
                auto tmp7 = in_ptr1[i1];
                auto tmp9 = out_ptr2[i1];
                auto tmp12 = out_ptr0[i1];
                auto tmp22 = out_ptr1[i1];
                auto tmp25 = in_ptr4[i1];
                auto tmp1 = static_cast<float>(0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = static_cast<float>(0.0);
                auto tmp5 = tmp2 ? tmp3 : tmp4;
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.125);
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = static_cast<float>(8);
                auto tmp14 = tmp12 / tmp13;
                auto tmp15 = static_cast<float>(1e-05);
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = 1 / std::sqrt(tmp16);
                auto tmp18 = tmp17 * tmp17;
                auto tmp19 = tmp11 * tmp18;
                auto tmp20 = tmp8 * tmp19;
                auto tmp21 = tmp5 - tmp20;
                auto tmp23 = tmp22 * tmp10;
                auto tmp24 = tmp21 - tmp23;
                auto tmp26 = tmp17 * tmp25;
                auto tmp27 = tmp24 * tmp26;
                in_out_ptr0[i1 + (512*i0)] = tmp27;
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
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr3 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp4 = 0;
                float tmp11 = 0;
                float tmp13 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    auto tmp1 = in_ptr1[i0];
                    auto tmp5 = in_ptr2[i0 + (512*i1)];
                    auto tmp9 = in_ptr3[i0 + (512*i1)];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    auto tmp6 = static_cast<float>(0);
                    auto tmp7 = tmp5 <= tmp6;
                    auto tmp8 = static_cast<float>(0.0);
                    auto tmp10 = tmp7 ? tmp8 : tmp9;
                    auto tmp12 = tmp10 * tmp2;
                    tmp4 += tmp3;
                    tmp11 += tmp10;
                    tmp13 += tmp12;
                }
                out_ptr0[i0] = tmp4;
                out_ptr1[i0] = tmp11;
                out_ptr2[i0] = tmp13;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (512*i0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i1) + (512*i0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (512*i0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i1);
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i1);
                auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i1);
                auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i1);
                auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.125));
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = at::vec::Vectorized<float>(static_cast<float>(8));
                auto tmp14 = tmp12 / tmp13;
                auto tmp15 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = tmp16.rsqrt();
                auto tmp18 = tmp17 * tmp17;
                auto tmp19 = tmp11 * tmp18;
                auto tmp20 = tmp8 * tmp19;
                auto tmp21 = tmp5 - tmp20;
                auto tmp23 = tmp22 * tmp10;
                auto tmp24 = tmp21 - tmp23;
                auto tmp26 = tmp17 * tmp25;
                auto tmp27 = tmp24 * tmp26;
                tmp27.store(in_out_ptr0 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = in_ptr2[i1 + (512*i0)];
                auto tmp4 = in_out_ptr0[i1 + (512*i0)];
                auto tmp6 = in_ptr0[i1 + (512*i0)];
                auto tmp7 = in_ptr1[i1];
                auto tmp9 = out_ptr2[i1];
                auto tmp12 = out_ptr0[i1];
                auto tmp22 = out_ptr1[i1];
                auto tmp25 = in_ptr4[i1];
                auto tmp1 = static_cast<float>(0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = static_cast<float>(0.0);
                auto tmp5 = tmp2 ? tmp3 : tmp4;
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.125);
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = static_cast<float>(8);
                auto tmp14 = tmp12 / tmp13;
                auto tmp15 = static_cast<float>(1e-05);
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = 1 / std::sqrt(tmp16);
                auto tmp18 = tmp17 * tmp17;
                auto tmp19 = tmp11 * tmp18;
                auto tmp20 = tmp8 * tmp19;
                auto tmp21 = tmp5 - tmp20;
                auto tmp23 = tmp22 * tmp10;
                auto tmp24 = tmp21 - tmp23;
                auto tmp26 = tmp17 * tmp25;
                auto tmp27 = tmp24 * tmp26;
                in_out_ptr0[i1 + (512*i0)] = tmp27;
            }
        }
    }
}
''')


kernel_cpp_4 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1,
                       float* __restrict__ in_out_ptr2,
                       float* __restrict__ in_out_ptr3,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const bool* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       const float* __restrict__ in_ptr7,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    auto out_ptr2 = in_out_ptr2;
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
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            tmp2.store(in_out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr0[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            in_out_ptr0[i0] = tmp2;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr1[i0 + (512*i1)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            tmp2.store(in_out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = out_ptr1[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            in_out_ptr1[i0] = tmp2;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<2048; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr2[i0 + (2048*i1)];
                    tmp1 += tmp0;
                }
                out_ptr2[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<128; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp2 = tmp0 / tmp1;
            tmp2.store(in_out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = out_ptr2[i0];
            auto tmp1 = static_cast<float>(8);
            auto tmp2 = tmp0 / tmp1;
            in_out_ptr2[i0] = tmp2;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<2048; i0+=1)
        {
            {
                float tmp4 = 0;
                float tmp17 = 0;
                float tmp19 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr2[i0 + (2048*i1)];
                    auto tmp1 = in_out_ptr2[i0];
                    auto tmp5 = in_ptr3[i0 + (2048*i1)];
                    auto tmp9 = in_ptr4[i0 + (2048*i1)];
                    auto tmp10 = in_ptr5[i0 + (2048*i1)];
                    auto tmp14 = in_ptr6[i0 + (2048*i1)];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    auto tmp6 = static_cast<float>(0);
                    auto tmp7 = tmp5 <= tmp6;
                    auto tmp8 = static_cast<float>(0.0);
                    auto tmp11 = static_cast<float>(1);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = tmp9 ? tmp8 : tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = tmp7 ? tmp8 : tmp15;
                    auto tmp18 = tmp16 * tmp2;
                    tmp4 += tmp3;
                    tmp17 += tmp16;
                    tmp19 += tmp18;
                }
                out_ptr3[i0] = tmp4;
                out_ptr4[i0] = tmp17;
                out_ptr5[i0] = tmp19;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            for(long i1=0; i1<128; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + (16*i1) + (2048*i0));
                float g_tmp_buffer_in_ptr4[16] = {0};
                flag_to_float(in_ptr4 + (16*i1) + (2048*i0), g_tmp_buffer_in_ptr4, 16);
                auto tmp4 = at::vec::Vectorized<float>::loadu(g_tmp_buffer_in_ptr4);
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + (16*i1) + (2048*i0));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + (16*i1) + (2048*i0));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (2048*i0));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + 16*i1);
                auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr5 + 16*i1);
                auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr3 + 16*i1);
                auto tmp28 = at::vec::Vectorized<float>::loadu(out_ptr4 + 16*i1);
                auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr7 + 16*i1);
                auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                auto tmp6 = at::vec::Vectorized<float>(static_cast<float>(1));
                auto tmp7 = tmp5 / tmp6;
                auto tmp8 = decltype(tmp3)::blendv(tmp7, tmp3, tmp4);
                auto tmp10 = tmp8 + tmp9;
                auto tmp11 = decltype(tmp3)::blendv(tmp10, tmp3, tmp2);
                auto tmp14 = tmp12 - tmp13;
                auto tmp16 = at::vec::Vectorized<float>(static_cast<float>(0.125));
                auto tmp17 = tmp15 * tmp16;
                auto tmp19 = at::vec::Vectorized<float>(static_cast<float>(8));
                auto tmp20 = tmp18 / tmp19;
                auto tmp21 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp22 = tmp20 + tmp21;
                auto tmp23 = tmp22.rsqrt();
                auto tmp24 = tmp23 * tmp23;
                auto tmp25 = tmp17 * tmp24;
                auto tmp26 = tmp14 * tmp25;
                auto tmp27 = tmp11 - tmp26;
                auto tmp29 = tmp28 * tmp16;
                auto tmp30 = tmp27 - tmp29;
                auto tmp32 = tmp23 * tmp31;
                auto tmp33 = tmp30 * tmp32;
                tmp33.store(in_out_ptr3 + (16*i1) + (2048*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=2048; i1<2048; i1+=1)
            {
                auto tmp0 = in_ptr3[i1 + (2048*i0)];
                auto tmp4 = in_ptr4[i1 + (2048*i0)];
                auto tmp5 = in_ptr5[i1 + (2048*i0)];
                auto tmp9 = in_ptr6[i1 + (2048*i0)];
                auto tmp12 = in_ptr2[i1 + (2048*i0)];
                auto tmp13 = in_out_ptr2[i1];
                auto tmp15 = out_ptr5[i1];
                auto tmp18 = out_ptr3[i1];
                auto tmp28 = out_ptr4[i1];
                auto tmp31 = in_ptr7[i1];
                auto tmp1 = static_cast<float>(0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = static_cast<float>(0.0);
                auto tmp6 = static_cast<float>(1);
                auto tmp7 = tmp5 / tmp6;
                auto tmp8 = tmp4 ? tmp3 : tmp7;
                auto tmp10 = tmp8 + tmp9;
                auto tmp11 = tmp2 ? tmp3 : tmp10;
                auto tmp14 = tmp12 - tmp13;
                auto tmp16 = static_cast<float>(0.125);
                auto tmp17 = tmp15 * tmp16;
                auto tmp19 = static_cast<float>(8);
                auto tmp20 = tmp18 / tmp19;
                auto tmp21 = static_cast<float>(1e-05);
                auto tmp22 = tmp20 + tmp21;
                auto tmp23 = 1 / std::sqrt(tmp22);
                auto tmp24 = tmp23 * tmp23;
                auto tmp25 = tmp17 * tmp24;
                auto tmp26 = tmp14 * tmp25;
                auto tmp27 = tmp11 - tmp26;
                auto tmp29 = tmp28 * tmp16;
                auto tmp30 = tmp27 - tmp29;
                auto tmp32 = tmp23 * tmp31;
                auto tmp33 = tmp30 * tmp32;
                in_out_ptr3[i1 + (2048*i0)] = tmp33;
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
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr3 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp4 = 0;
                float tmp11 = 0;
                float tmp13 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    auto tmp1 = in_ptr1[i0];
                    auto tmp5 = in_ptr2[i0 + (512*i1)];
                    auto tmp9 = in_ptr3[i0 + (512*i1)];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    auto tmp6 = static_cast<float>(0);
                    auto tmp7 = tmp5 <= tmp6;
                    auto tmp8 = static_cast<float>(0.0);
                    auto tmp10 = tmp7 ? tmp8 : tmp9;
                    auto tmp12 = tmp10 * tmp2;
                    tmp4 += tmp3;
                    tmp11 += tmp10;
                    tmp13 += tmp12;
                }
                out_ptr0[i0] = tmp4;
                out_ptr1[i0] = tmp11;
                out_ptr2[i0] = tmp13;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (512*i0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i1) + (512*i0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (512*i0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i1);
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i1);
                auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i1);
                auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i1);
                auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.125));
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = at::vec::Vectorized<float>(static_cast<float>(8));
                auto tmp14 = tmp12 / tmp13;
                auto tmp15 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = tmp16.rsqrt();
                auto tmp18 = tmp17 * tmp17;
                auto tmp19 = tmp11 * tmp18;
                auto tmp20 = tmp8 * tmp19;
                auto tmp21 = tmp5 - tmp20;
                auto tmp23 = tmp22 * tmp10;
                auto tmp24 = tmp21 - tmp23;
                auto tmp26 = tmp17 * tmp25;
                auto tmp27 = tmp24 * tmp26;
                tmp27.store(in_out_ptr0 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = in_ptr2[i1 + (512*i0)];
                auto tmp4 = in_out_ptr0[i1 + (512*i0)];
                auto tmp6 = in_ptr0[i1 + (512*i0)];
                auto tmp7 = in_ptr1[i1];
                auto tmp9 = out_ptr2[i1];
                auto tmp12 = out_ptr0[i1];
                auto tmp22 = out_ptr1[i1];
                auto tmp25 = in_ptr4[i1];
                auto tmp1 = static_cast<float>(0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = static_cast<float>(0.0);
                auto tmp5 = tmp2 ? tmp3 : tmp4;
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.125);
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = static_cast<float>(8);
                auto tmp14 = tmp12 / tmp13;
                auto tmp15 = static_cast<float>(1e-05);
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = 1 / std::sqrt(tmp16);
                auto tmp18 = tmp17 * tmp17;
                auto tmp19 = tmp11 * tmp18;
                auto tmp20 = tmp8 * tmp19;
                auto tmp21 = tmp5 - tmp20;
                auto tmp23 = tmp22 * tmp10;
                auto tmp24 = tmp21 - tmp23;
                auto tmp26 = tmp17 * tmp25;
                auto tmp27 = tmp24 * tmp26;
                in_out_ptr0[i1 + (512*i0)] = tmp27;
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
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    auto in_ptr3 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp4 = 0;
                float tmp11 = 0;
                float tmp13 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    auto tmp1 = in_ptr1[i0];
                    auto tmp5 = in_ptr2[i0 + (512*i1)];
                    auto tmp9 = in_ptr3[i0 + (512*i1)];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    auto tmp6 = static_cast<float>(0);
                    auto tmp7 = tmp5 <= tmp6;
                    auto tmp8 = static_cast<float>(0.0);
                    auto tmp10 = tmp7 ? tmp8 : tmp9;
                    auto tmp12 = tmp10 * tmp2;
                    tmp4 += tmp3;
                    tmp11 += tmp10;
                    tmp13 += tmp12;
                }
                out_ptr0[i0] = tmp4;
                out_ptr1[i0] = tmp11;
                out_ptr2[i0] = tmp13;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (512*i0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i1) + (512*i0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i1) + (512*i0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i1);
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i1);
                auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i1);
                auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i1);
                auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.125));
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = at::vec::Vectorized<float>(static_cast<float>(8));
                auto tmp14 = tmp12 / tmp13;
                auto tmp15 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = tmp16.rsqrt();
                auto tmp18 = tmp17 * tmp17;
                auto tmp19 = tmp11 * tmp18;
                auto tmp20 = tmp8 * tmp19;
                auto tmp21 = tmp5 - tmp20;
                auto tmp23 = tmp22 * tmp10;
                auto tmp24 = tmp21 - tmp23;
                auto tmp26 = tmp17 * tmp25;
                auto tmp27 = tmp24 * tmp26;
                tmp27.store(in_out_ptr0 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = in_ptr2[i1 + (512*i0)];
                auto tmp4 = in_out_ptr0[i1 + (512*i0)];
                auto tmp6 = in_ptr0[i1 + (512*i0)];
                auto tmp7 = in_ptr1[i1];
                auto tmp9 = out_ptr2[i1];
                auto tmp12 = out_ptr0[i1];
                auto tmp22 = out_ptr1[i1];
                auto tmp25 = in_ptr4[i1];
                auto tmp1 = static_cast<float>(0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = static_cast<float>(0.0);
                auto tmp5 = tmp2 ? tmp3 : tmp4;
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.125);
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = static_cast<float>(8);
                auto tmp14 = tmp12 / tmp13;
                auto tmp15 = static_cast<float>(1e-05);
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = 1 / std::sqrt(tmp16);
                auto tmp18 = tmp17 * tmp17;
                auto tmp19 = tmp11 * tmp18;
                auto tmp20 = tmp8 * tmp19;
                auto tmp21 = tmp5 - tmp20;
                auto tmp23 = tmp22 * tmp10;
                auto tmp24 = tmp21 - tmp23;
                auto tmp26 = tmp17 * tmp25;
                auto tmp27 = tmp24 * tmp26;
                in_out_ptr0[i1 + (512*i0)] = tmp27;
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
                       const bool* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       const float* __restrict__ in_ptr7,
                       const float* __restrict__ in_ptr8,
                       const float* __restrict__ in_ptr9,
                       const float* __restrict__ in_ptr10,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       float* __restrict__ out_ptr5,
                       float* __restrict__ out_ptr6)
{
    {
        for(long i0=0; i0<1024; i0+=1)
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
        #pragma omp simd simdlen(8) 
        for(long i0=16384; i0<16384; i0+=1)
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
        #pragma GCC ivdep
        for(long i0=0; i0<2048; i0+=1)
        {
            {
                float tmp1 = 0;
                float tmp6 = 0;
                float tmp8 = 0;
                float tmp13 = 0;
                float tmp15 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_out_ptr0[i0 + (2048*i1)];
                    auto tmp2 = in_ptr5[i0 + (2048*i1)];
                    auto tmp3 = in_ptr6[i0];
                    auto tmp9 = in_ptr7[i0 + (2048*i1)];
                    auto tmp10 = in_ptr8[i0];
                    auto tmp4 = tmp2 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    auto tmp7 = tmp0 * tmp4;
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp12 = tmp11 * tmp11;
                    auto tmp14 = tmp0 * tmp11;
                    tmp1 += tmp0;
                    tmp6 += tmp5;
                    tmp8 += tmp7;
                    tmp13 += tmp12;
                    tmp15 += tmp14;
                }
                out_ptr0[i0] = tmp1;
                out_ptr1[i0] = tmp6;
                out_ptr2[i0] = tmp8;
                out_ptr3[i0] = tmp13;
                out_ptr4[i0] = tmp15;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            for(long i1=0; i1<128; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i1) + (2048*i0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + (16*i1) + (2048*i0));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + 16*i1);
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + 16*i1);
                auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr1 + 16*i1);
                auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + 16*i1);
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr9 + 16*i1);
                auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr7 + (16*i1) + (2048*i0));
                auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr8 + 16*i1);
                auto tmp26 = at::vec::Vectorized<float>::loadu(out_ptr4 + 16*i1);
                auto tmp28 = at::vec::Vectorized<float>::loadu(out_ptr3 + 16*i1);
                auto tmp37 = at::vec::Vectorized<float>::loadu(in_ptr10 + 16*i1);
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.125));
                auto tmp6 = tmp4 * tmp5;
                auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(8));
                auto tmp9 = tmp7 / tmp8;
                auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = tmp11.rsqrt();
                auto tmp13 = tmp12 * tmp12;
                auto tmp14 = tmp6 * tmp13;
                auto tmp15 = tmp3 * tmp14;
                auto tmp16 = tmp0 - tmp15;
                auto tmp18 = tmp17 * tmp5;
                auto tmp19 = tmp16 - tmp18;
                auto tmp21 = tmp12 * tmp20;
                auto tmp22 = tmp19 * tmp21;
                auto tmp25 = tmp23 - tmp24;
                auto tmp27 = tmp26 * tmp5;
                auto tmp29 = tmp28 / tmp8;
                auto tmp30 = tmp29 + tmp10;
                auto tmp31 = tmp30.rsqrt();
                auto tmp32 = tmp31 * tmp31;
                auto tmp33 = tmp27 * tmp32;
                auto tmp34 = tmp25 * tmp33;
                auto tmp35 = tmp0 - tmp34;
                auto tmp36 = tmp35 - tmp18;
                auto tmp38 = tmp31 * tmp37;
                auto tmp39 = tmp36 * tmp38;
                tmp22.store(out_ptr5 + (16*i1) + (2048*i0));
                tmp39.store(out_ptr6 + (16*i1) + (2048*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=2048; i1<2048; i1+=1)
            {
                auto tmp0 = in_out_ptr0[i1 + (2048*i0)];
                auto tmp1 = in_ptr5[i1 + (2048*i0)];
                auto tmp2 = in_ptr6[i1];
                auto tmp4 = out_ptr2[i1];
                auto tmp7 = out_ptr1[i1];
                auto tmp17 = out_ptr0[i1];
                auto tmp20 = in_ptr9[i1];
                auto tmp23 = in_ptr7[i1 + (2048*i0)];
                auto tmp24 = in_ptr8[i1];
                auto tmp26 = out_ptr4[i1];
                auto tmp28 = out_ptr3[i1];
                auto tmp37 = in_ptr10[i1];
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(0.125);
                auto tmp6 = tmp4 * tmp5;
                auto tmp8 = static_cast<float>(8);
                auto tmp9 = tmp7 / tmp8;
                auto tmp10 = static_cast<float>(1e-05);
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = 1 / std::sqrt(tmp11);
                auto tmp13 = tmp12 * tmp12;
                auto tmp14 = tmp6 * tmp13;
                auto tmp15 = tmp3 * tmp14;
                auto tmp16 = tmp0 - tmp15;
                auto tmp18 = tmp17 * tmp5;
                auto tmp19 = tmp16 - tmp18;
                auto tmp21 = tmp12 * tmp20;
                auto tmp22 = tmp19 * tmp21;
                auto tmp25 = tmp23 - tmp24;
                auto tmp27 = tmp26 * tmp5;
                auto tmp29 = tmp28 / tmp8;
                auto tmp30 = tmp29 + tmp10;
                auto tmp31 = 1 / std::sqrt(tmp30);
                auto tmp32 = tmp31 * tmp31;
                auto tmp33 = tmp27 * tmp32;
                auto tmp34 = tmp25 * tmp33;
                auto tmp35 = tmp0 - tmp34;
                auto tmp36 = tmp35 - tmp18;
                auto tmp38 = tmp31 * tmp37;
                auto tmp39 = tmp36 * tmp38;
                out_ptr5[i1 + (2048*i0)] = tmp22;
                out_ptr6[i1 + (2048*i0)] = tmp39;
            }
        }
    }
}
''')


kernel_cpp_8 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp4 = 0;
                float tmp11 = 0;
                float tmp13 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
                    auto tmp1 = in_ptr1[i0];
                    auto tmp5 = in_ptr2[i0 + (512*i1)];
                    auto tmp9 = in_ptr3[i0 + (512*i1)];
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp3 = tmp2 * tmp2;
                    auto tmp6 = static_cast<float>(0);
                    auto tmp7 = tmp5 <= tmp6;
                    auto tmp8 = static_cast<float>(0.0);
                    auto tmp10 = tmp7 ? tmp8 : tmp9;
                    auto tmp12 = tmp10 * tmp2;
                    tmp4 += tmp3;
                    tmp11 += tmp10;
                    tmp13 += tmp12;
                }
                out_ptr0[i0] = tmp4;
                out_ptr1[i0] = tmp11;
                out_ptr2[i0] = tmp13;
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
                       float* __restrict__ out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long i0=0; i0<1000; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<8; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (1000*i1)];
                    tmp1 += tmp0;
                }
                out_ptr0[i0] = tmp1;
            }
        }
    }
    {
        for(long i0=0; i0<128; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
            auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp5 = tmp3 + tmp4;
            auto tmp6 = tmp5.rsqrt();
            auto tmp7 = tmp0 * tmp6;
            tmp7.store(in_out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = in_ptr1[i0];
            auto tmp1 = in_out_ptr0[i0];
            auto tmp2 = static_cast<float>(8);
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = tmp3 + tmp4;
            auto tmp6 = 1 / std::sqrt(tmp5);
            auto tmp7 = tmp0 * tmp6;
            in_out_ptr0[i0] = tmp7;
        }
    }
}
''')


kernel_cpp_10 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0)
{
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
            auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp5 = tmp3 + tmp4;
            auto tmp6 = tmp5.rsqrt();
            auto tmp7 = tmp0 * tmp6;
            tmp7.store(in_out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_out_ptr0[i0];
            auto tmp2 = static_cast<float>(8);
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = tmp3 + tmp4;
            auto tmp6 = 1 / std::sqrt(tmp5);
            auto tmp7 = tmp0 * tmp6;
            in_out_ptr0[i0] = tmp7;
        }
    }
}
''')


kernel_cpp_11 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0)
{
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
            auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp5 = tmp3 + tmp4;
            auto tmp6 = tmp5.rsqrt();
            auto tmp7 = tmp0 * tmp6;
            tmp7.store(in_out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_out_ptr0[i0];
            auto tmp2 = static_cast<float>(8);
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = tmp3 + tmp4;
            auto tmp6 = 1 / std::sqrt(tmp5);
            auto tmp7 = tmp0 * tmp6;
            in_out_ptr0[i0] = tmp7;
        }
    }
}
''')


kernel_cpp_12 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0)
{
    {
        for(long i0=0; i0<128; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
            auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp5 = tmp3 + tmp4;
            auto tmp6 = tmp5.rsqrt();
            auto tmp7 = tmp0 * tmp6;
            tmp7.store(in_out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_out_ptr0[i0];
            auto tmp2 = static_cast<float>(8);
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = tmp3 + tmp4;
            auto tmp6 = 1 / std::sqrt(tmp5);
            auto tmp7 = tmp0 * tmp6;
            in_out_ptr0[i0] = tmp7;
        }
    }
}
''')


kernel_cpp_13 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0)
{
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
            auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp5 = tmp3 + tmp4;
            auto tmp6 = tmp5.rsqrt();
            auto tmp7 = tmp0 * tmp6;
            tmp7.store(in_out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_out_ptr0[i0];
            auto tmp2 = static_cast<float>(8);
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = tmp3 + tmp4;
            auto tmp6 = 1 / std::sqrt(tmp5);
            auto tmp7 = tmp0 * tmp6;
            in_out_ptr0[i0] = tmp7;
        }
    }
}
''')


kernel_cpp_14 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0)
{
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
            auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp5 = tmp3 + tmp4;
            auto tmp6 = tmp5.rsqrt();
            auto tmp7 = tmp0 * tmp6;
            tmp7.store(in_out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_out_ptr0[i0];
            auto tmp2 = static_cast<float>(8);
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = tmp3 + tmp4;
            auto tmp6 = 1 / std::sqrt(tmp5);
            auto tmp7 = tmp0 * tmp6;
            in_out_ptr0[i0] = tmp7;
        }
    }
}
''')


kernel_cpp_15 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0)
{
    {
        for(long i0=0; i0<128; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp5 = tmp3 + tmp4;
            auto tmp6 = tmp5.rsqrt();
            auto tmp7 = tmp0 * tmp6;
            tmp7.store(in_out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = in_out_ptr0[i0];
            auto tmp1 = in_ptr0[i0];
            auto tmp2 = static_cast<float>(8);
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = tmp3 + tmp4;
            auto tmp6 = 1 / std::sqrt(tmp5);
            auto tmp7 = tmp0 * tmp6;
            in_out_ptr0[i0] = tmp7;
        }
    }
}
''')


kernel_cpp_16 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0)
{
    {
        for(long i0=0; i0<128; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
            auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp5 = tmp3 + tmp4;
            auto tmp6 = tmp5.rsqrt();
            auto tmp7 = tmp0 * tmp6;
            tmp7.store(in_out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=2048; i0<2048; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_out_ptr0[i0];
            auto tmp2 = static_cast<float>(8);
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = tmp3 + tmp4;
            auto tmp6 = 1 / std::sqrt(tmp5);
            auto tmp7 = tmp0 * tmp6;
            in_out_ptr0[i0] = tmp7;
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
                       const float* __restrict__ in_ptr6,
                       float* __restrict__ out_ptr0)
{
    {
        for(long i0=0; i0<32; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(8));
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
            auto tmp5 = tmp3 + tmp4;
            auto tmp6 = tmp5.rsqrt();
            auto tmp7 = tmp0 * tmp6;
            tmp7.store(out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=512; i0<512; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = static_cast<float>(8);
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = tmp3 + tmp4;
            auto tmp6 = 1 / std::sqrt(tmp5);
            auto tmp7 = tmp0 * tmp6;
            out_ptr0[i0] = tmp7;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i1) + (512*i0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + (16*i1) + (512*i0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + (16*i1) + (512*i0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + 16*i1);
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i1);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i1);
                auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + 16*i1);
                auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + 16*i1);
                auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = at::vec::Vectorized<float>(static_cast<float>(0.0));
                auto tmp5 = decltype(tmp3)::blendv(tmp4, tmp3, tmp2);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.125));
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = at::vec::Vectorized<float>(static_cast<float>(8));
                auto tmp14 = tmp12 / tmp13;
                auto tmp15 = at::vec::Vectorized<float>(static_cast<float>(1e-05));
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = tmp16.rsqrt();
                auto tmp18 = tmp17 * tmp17;
                auto tmp19 = tmp11 * tmp18;
                auto tmp20 = tmp8 * tmp19;
                auto tmp21 = tmp5 - tmp20;
                auto tmp23 = tmp22 * tmp10;
                auto tmp24 = tmp21 - tmp23;
                auto tmp26 = tmp17 * tmp25;
                auto tmp27 = tmp24 * tmp26;
                tmp27.store(in_out_ptr0 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = in_ptr2[i1 + (512*i0)];
                auto tmp4 = in_out_ptr0[i1 + (512*i0)];
                auto tmp6 = in_ptr3[i1 + (512*i0)];
                auto tmp7 = in_ptr4[i1];
                auto tmp9 = in_ptr0[i1];
                auto tmp12 = in_ptr1[i1];
                auto tmp22 = in_ptr5[i1];
                auto tmp25 = in_ptr6[i1];
                auto tmp1 = static_cast<float>(0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = static_cast<float>(0.0);
                auto tmp5 = tmp2 ? tmp3 : tmp4;
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.125);
                auto tmp11 = tmp9 * tmp10;
                auto tmp13 = static_cast<float>(8);
                auto tmp14 = tmp12 / tmp13;
                auto tmp15 = static_cast<float>(1e-05);
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = 1 / std::sqrt(tmp16);
                auto tmp18 = tmp17 * tmp17;
                auto tmp19 = tmp11 * tmp18;
                auto tmp20 = tmp8 * tmp19;
                auto tmp21 = tmp5 - tmp20;
                auto tmp23 = tmp22 * tmp10;
                auto tmp24 = tmp21 - tmp23;
                auto tmp26 = tmp17 * tmp25;
                auto tmp27 = tmp24 * tmp26;
                in_out_ptr0[i1 + (512*i0)] = tmp27;
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
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
        for(long i0=0; i0<8; i0+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.03125));
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
                    auto tmp10 = static_cast<float>(0.03125);
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
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<1024; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp8 = 0;
                auto tmp8_vec = at::vec::Vectorized<float>(tmp8);
                float tmp13 = 0;
                auto tmp13_vec = at::vec::Vectorized<float>(tmp13);
                for(long i1=0; i1<8; i1+=1)
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
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
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
                    auto tmp12 = at::vec::Vectorized<float>(static_cast<float>(0.03125));
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
                    auto tmp12 = static_cast<float>(0.03125);
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
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.03125));
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
                    auto tmp10 = static_cast<float>(0.03125);
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
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.03125));
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
                    auto tmp10 = static_cast<float>(0.03125);
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
                       const float* __restrict__ in_ptr7,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    {
        for(long i0=0; i0<2048; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i0);
            auto tmp10 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
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
        #pragma omp simd simdlen(8) 
        for(long i0=32768; i0<32768; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp4 = in_ptr1[i0];
            auto tmp6 = in_ptr2[i0];
            auto tmp7 = in_ptr3[i0];
            auto tmp10 = in_out_ptr0[i0];
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
        #pragma GCC ivdep
        for(long i0=0; i0<1024; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                for(long i1=0; i1<8; i1+=1)
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
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
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
                    auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.03125));
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
                    auto tmp5 = static_cast<float>(0.03125);
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
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.03125));
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
                    auto tmp10 = static_cast<float>(0.03125);
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
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.03125));
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
                    auto tmp10 = static_cast<float>(0.03125);
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
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=0; i0<1024; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp8 = 0;
                auto tmp8_vec = at::vec::Vectorized<float>(tmp8);
                float tmp13 = 0;
                auto tmp13_vec = at::vec::Vectorized<float>(tmp13);
                for(long i1=0; i1<8; i1+=1)
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
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
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
                    auto tmp12 = at::vec::Vectorized<float>(static_cast<float>(0.03125));
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
                    auto tmp12 = static_cast<float>(0.03125);
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
''')


kernel_cpp_26 = async_compile.cpp('''
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
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.03125));
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
                    auto tmp10 = static_cast<float>(0.03125);
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
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.03125));
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
                    auto tmp10 = static_cast<float>(0.03125);
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
''')


kernel_cpp_28 = async_compile.cpp('''
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
    {
        for(long i0=0; i0<2048; i0+=1)
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
        #pragma omp simd simdlen(8) 
        for(long i0=32768; i0<32768; i0+=1)
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
        #pragma GCC ivdep
        for(long i0=0; i0<1024; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                for(long i1=0; i1<8; i1+=1)
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
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
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
                    auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.03125));
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
                    auto tmp5 = static_cast<float>(0.03125);
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
''')


kernel_cpp_29 = async_compile.cpp('''
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
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.03125));
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
                    auto tmp10 = static_cast<float>(0.03125);
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
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.03125));
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
                    auto tmp10 = static_cast<float>(0.03125);
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
''')


kernel_cpp_31 = async_compile.cpp('''
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
    {
        #pragma GCC ivdep
        for(long i0=0; i0<1024; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp8 = 0;
                auto tmp8_vec = at::vec::Vectorized<float>(tmp8);
                float tmp13 = 0;
                auto tmp13_vec = at::vec::Vectorized<float>(tmp13);
                for(long i1=0; i1<8; i1+=1)
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
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
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
                    auto tmp12 = at::vec::Vectorized<float>(static_cast<float>(0.03125));
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
                    auto tmp12 = static_cast<float>(0.03125);
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
''')


kernel_cpp_32 = async_compile.cpp('''
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
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.03125));
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
                    auto tmp10 = static_cast<float>(0.03125);
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
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.03125));
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
                    auto tmp10 = static_cast<float>(0.03125);
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
''')


kernel_cpp_34 = async_compile.cpp('''
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
    {
        for(long i0=0; i0<2048; i0+=1)
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
        #pragma omp simd simdlen(8) 
        for(long i0=32768; i0<32768; i0+=1)
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
        #pragma GCC ivdep
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
                for(long i1=0; i1<8; i1+=1)
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
    {
        #pragma GCC ivdep
        for(long i0=0; i0<8; i0+=1)
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
                    auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.03125));
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
                    auto tmp5 = static_cast<float>(0.03125);
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
''')


kernel_cpp_35 = async_compile.cpp('''
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
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.03125));
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
                    auto tmp10 = static_cast<float>(0.03125);
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
    {
        #pragma GCC ivdep
        for(long i0=0; i0<256; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.0078125));
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
                    auto tmp10 = static_cast<float>(0.0078125);
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
                    for(long i1=0; i1<8; i1+=1)
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
            for(long i0=0; i0<8; i0+=1)
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
                        auto tmp12 = at::vec::Vectorized<float>(static_cast<float>(0.0078125));
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
                        auto tmp12 = static_cast<float>(0.0078125);
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


kernel_cpp_39 = async_compile.cpp('''
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
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.0078125));
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
                    auto tmp10 = static_cast<float>(0.0078125);
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
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.0078125));
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
                    auto tmp10 = static_cast<float>(0.0078125);
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
''')


kernel_cpp_41 = async_compile.cpp('''
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
            for(long i0=0; i0<4096; i0+=1)
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
            for(long i0=65536; i0<65536; i0+=1)
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
                    for(long i1=0; i1<8; i1+=1)
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
            for(long i0=0; i0<8; i0+=1)
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
                        auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.0078125));
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
                        auto tmp5 = static_cast<float>(0.0078125);
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


kernel_cpp_42 = async_compile.cpp('''
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
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.0078125));
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
                    auto tmp10 = static_cast<float>(0.0078125);
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
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.0078125));
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
                    auto tmp10 = static_cast<float>(0.0078125);
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
''')


kernel_cpp_44 = async_compile.cpp('''
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
                    for(long i1=0; i1<8; i1+=1)
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
            for(long i0=0; i0<8; i0+=1)
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
                        auto tmp12 = at::vec::Vectorized<float>(static_cast<float>(0.0078125));
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
                        auto tmp12 = static_cast<float>(0.0078125);
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


kernel_cpp_45 = async_compile.cpp('''
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
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.0078125));
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
                    auto tmp10 = static_cast<float>(0.0078125);
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
    {
        #pragma GCC ivdep
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.0078125));
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
                    auto tmp10 = static_cast<float>(0.0078125);
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
''')


kernel_cpp_47 = async_compile.cpp('''
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
            for(long i0=0; i0<4096; i0+=1)
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
            for(long i0=65536; i0<65536; i0+=1)
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
                    for(long i1=0; i1<8; i1+=1)
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
            for(long i0=0; i0<8; i0+=1)
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
                        auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.0078125));
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
                        auto tmp5 = static_cast<float>(0.0078125);
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


kernel_cpp_48 = async_compile.cpp('''
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


kernel_cpp_49 = async_compile.cpp('''
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
        for(long i0=0; i0<128; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.0078125));
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
                    auto tmp10 = static_cast<float>(0.0078125);
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
            for(long i0=0; i0<128; i0+=1)
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                    float tmp6 = 0;
                    auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                    float tmp11 = 0;
                    auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                    for(long i1=0; i1<8; i1+=1)
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
            for(long i0=0; i0<8; i0+=1)
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
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.001953125));
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
                        auto tmp10 = static_cast<float>(0.001953125);
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


kernel_cpp_51 = async_compile.cpp('''
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
                    for(long i1=0; i1<8; i1+=1)
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
            for(long i0=0; i0<8; i0+=1)
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
                        auto tmp12 = at::vec::Vectorized<float>(static_cast<float>(0.001953125));
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
                        auto tmp12 = static_cast<float>(0.001953125);
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


kernel_cpp_52 = async_compile.cpp('''
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
        for(long i0=0; i0<64; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.001953125));
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
                    auto tmp10 = static_cast<float>(0.001953125);
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
''')


kernel_cpp_53 = async_compile.cpp('''
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
        for(long i0=0; i0<64; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.001953125));
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
                    auto tmp10 = static_cast<float>(0.001953125);
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
''')


kernel_cpp_54 = async_compile.cpp('''
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
            for(long i0=0; i0<8192; i0+=1)
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
            for(long i0=131072; i0<131072; i0+=1)
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
                    for(long i1=0; i1<8; i1+=1)
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
            for(long i0=0; i0<8; i0+=1)
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
                        auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.001953125));
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
                        auto tmp5 = static_cast<float>(0.001953125);
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


kernel_cpp_55 = async_compile.cpp('''
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
        for(long i0=0; i0<64; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.001953125));
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
                    auto tmp10 = static_cast<float>(0.001953125);
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
''')


kernel_cpp_56 = async_compile.cpp('''
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
        for(long i0=0; i0<64; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.001953125));
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
                    auto tmp10 = static_cast<float>(0.001953125);
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
''')


kernel_cpp_57 = async_compile.cpp('''
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
                    for(long i1=0; i1<8; i1+=1)
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
            for(long i0=0; i0<8; i0+=1)
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
                        auto tmp12 = at::vec::Vectorized<float>(static_cast<float>(0.001953125));
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
                        auto tmp12 = static_cast<float>(0.001953125);
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


kernel_cpp_58 = async_compile.cpp('''
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


kernel_cpp_59 = async_compile.cpp('''
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
        for(long i0=0; i0<64; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.001953125));
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
                    auto tmp10 = static_cast<float>(0.001953125);
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
''')


kernel_cpp_60 = async_compile.cpp('''
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
        for(long i0=0; i0<64; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp11 = 0;
                auto tmp11_vec = at::vec::Vectorized<float>(tmp11);
                for(long i1=0; i1<8; i1+=1)
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
                    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.001953125));
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
                    auto tmp10 = static_cast<float>(0.001953125);
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
''')


kernel_cpp_61 = async_compile.cpp('''
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
    {
        for(long i0=0; i0<2048; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=32768; i0<32768; i0+=1)
        {
            auto tmp0 = in_out_ptr0[i0];
            auto tmp1 = in_ptr0[i0];
            auto tmp2 = tmp0 + tmp1;
            in_out_ptr0[i0] = tmp2;
        }
    }
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<512; i0+=1)
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
                    for(long i1=0; i1<8; i1+=1)
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
            for(long i0=0; i0<8; i0+=1)
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
                        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(0.00048828125));
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
                        auto tmp10 = static_cast<float>(0.00048828125);
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
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_321, convolution, squeeze_1, relu, getitem_2, getitem_3, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, relu_4, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, relu_8, convolution_10, squeeze_31, relu_9, convolution_11, squeeze_34, relu_10, convolution_12, squeeze_37, relu_11, convolution_13, squeeze_40, convolution_14, squeeze_43, relu_12, convolution_15, squeeze_46, relu_13, convolution_16, squeeze_49, relu_14, convolution_17, squeeze_52, relu_15, convolution_18, squeeze_55, relu_16, convolution_19, squeeze_58, relu_17, convolution_20, squeeze_61, relu_18, convolution_21, squeeze_64, relu_19, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, relu_21, convolution_24, squeeze_73, relu_22, convolution_25, squeeze_76, relu_23, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_24, convolution_28, squeeze_85, relu_25, convolution_29, squeeze_88, relu_26, convolution_30, squeeze_91, relu_27, convolution_31, squeeze_94, relu_28, convolution_32, squeeze_97, relu_29, convolution_33, squeeze_100, relu_30, convolution_34, squeeze_103, relu_31, convolution_35, squeeze_106, relu_32, convolution_36, squeeze_109, relu_33, convolution_37, squeeze_112, relu_34, convolution_38, squeeze_115, relu_35, convolution_39, squeeze_118, relu_36, convolution_40, squeeze_121, relu_37, convolution_41, squeeze_124, relu_38, convolution_42, squeeze_127, relu_39, convolution_43, squeeze_130, relu_40, convolution_44, relu_41, convolution_45, convolution_46, relu_42, convolution_47, relu_43, convolution_48, relu_44, convolution_49, relu_45, convolution_50, relu_46, convolution_51, relu_47, convolution_52, view, permute_1, le, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50, tangents_51, tangents_52, tangents_53, tangents_54, tangents_55, tangents_56, tangents_57, tangents_58, tangents_59, tangents_60, tangents_61, tangents_62, tangents_63, tangents_64, tangents_65, tangents_66, tangents_67, tangents_68, tangents_69, tangents_70, tangents_71, tangents_72, tangents_73, tangents_74, tangents_75, tangents_76, tangents_77, tangents_78, tangents_79, tangents_80, tangents_81, tangents_82, tangents_83, tangents_84, tangents_85, tangents_86, tangents_87, tangents_88, tangents_89, tangents_90, tangents_91, tangents_92, tangents_93, tangents_94, tangents_95, tangents_96, tangents_97, tangents_98, tangents_99, tangents_100, tangents_101, tangents_102, tangents_103, tangents_104, tangents_105, tangents_106, tangents_107, tangents_108, tangents_109, tangents_110, tangents_111, tangents_112, tangents_113, tangents_114, tangents_115, tangents_116, tangents_117, tangents_118, tangents_119, tangents_120, tangents_121, tangents_122, tangents_123, tangents_124, tangents_125, tangents_126, tangents_127, tangents_128, tangents_129, tangents_130, tangents_131, tangents_132, tangents_133, tangents_134, tangents_135, tangents_136, tangents_137, tangents_138, tangents_139, tangents_140, tangents_141, tangents_142, tangents_143, tangents_144, tangents_145, tangents_146, tangents_147, tangents_148, tangents_149, tangents_150, tangents_151, tangents_152, tangents_153, tangents_154, tangents_155, tangents_156, tangents_157, tangents_158, tangents_159, tangents_160 = args
    args.clear()
    buf0 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf1 = buf0; del buf0  # reuse
    buf3 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf4 = buf3; del buf3  # reuse
    buf6 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf7 = buf6; del buf6  # reuse
    kernel_cpp_0(c_void_p(buf1.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(convolution_46.data_ptr()))
    buf27 = empty_strided((8, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    extern_kernels.mm(tangents_160, permute_1, out=buf27)
    del permute_1
    buf18 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf19 = buf18; del buf18  # reuse
    buf21 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf22 = buf21; del buf21  # reuse
    buf24 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf25 = buf24; del buf24  # reuse
    buf26 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf30 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf31 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf33 = empty_strided((8, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_1(c_void_p(buf19.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf33.data_ptr()))
    del convolution_52
    del primals_158
    buf34 = aten.convolution_backward(buf33, relu_47, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_157
    buf35 = buf34[0]
    assert_size_stride(buf35, (8, 512, 1, 1), (512, 1, 1, 1))
    buf23 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf37 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf38 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf40 = buf35; del buf35  # reuse
    kernel_cpp_2(c_void_p(buf40.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(relu_47.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()))
    del convolution_51
    del primals_155
    del relu_47
    buf41 = aten.convolution_backward(buf40, relu_46, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf40
    del primals_154
    buf42 = buf41[0]
    assert_size_stride(buf42, (8, 512, 1, 1), (512, 1, 1, 1))
    buf20 = buf22; del buf22  # reuse
    buf44 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf45 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf47 = buf42; del buf42  # reuse
    kernel_cpp_3(c_void_p(buf47.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(relu_46.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()))
    del convolution_50
    del primals_152
    del relu_46
    buf48 = aten.convolution_backward(buf47, relu_45, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf47
    del primals_151
    buf49 = buf48[0]
    assert_size_stride(buf49, (8, 2048, 1, 1), (2048, 1, 1, 1))
    buf9 = buf19; del buf19  # reuse
    buf10 = buf9; del buf9  # reuse
    buf12 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf13 = buf12; del buf12  # reuse
    buf15 = buf25; del buf25  # reuse
    buf16 = buf15; del buf15  # reuse
    buf17 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf51 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf52 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf53 = as_strided(buf33, (8, 2048, 1, 1), (2048, 1, 16384, 16384)); del buf33  # reuse
    buf55 = as_strided(buf53, (8, 2048, 1, 1), (2048, 1, 1, 1)); del buf53  # reuse
    kernel_cpp_4(c_void_p(buf10.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(convolution_47.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(relu_45.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()))
    del convolution_49
    del primals_149
    buf56 = aten.convolution_backward(buf55, relu_44, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_148
    buf57 = buf56[0]
    assert_size_stride(buf57, (8, 512, 1, 1), (512, 1, 1, 1))
    buf14 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf59 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf60 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf62 = buf57; del buf57  # reuse
    kernel_cpp_5(c_void_p(buf62.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(relu_44.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()))
    del convolution_48
    del primals_146
    del relu_44
    buf63 = aten.convolution_backward(buf62, relu_43, primals_145, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf62
    del primals_145
    buf64 = buf63[0]
    assert_size_stride(buf64, (8, 512, 1, 1), (512, 1, 1, 1))
    buf11 = buf13; del buf13  # reuse
    buf66 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf67 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf69 = buf64; del buf64  # reuse
    kernel_cpp_6(c_void_p(buf69.data_ptr()), c_void_p(convolution_47.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(relu_43.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()))
    del convolution_47
    del primals_143
    del relu_43
    buf70 = aten.convolution_backward(buf69, relu_42, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf69
    del primals_142
    buf71 = buf70[0]
    assert_size_stride(buf71, (8, 2048, 1, 1), (2048, 1, 1, 1))
    buf73 = as_strided(buf27, (8, 2048, 1, 1), (2048, 1, 16384, 16384)); del buf27  # reuse
    buf74 = as_strided(buf16, (2048, ), (1, )); del buf16  # reuse
    buf8 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf75 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf81 = empty_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    buf77 = buf55; del buf55  # reuse
    buf83 = empty_strided((8, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_7(c_void_p(buf73.data_ptr()), c_void_p(relu_42.data_ptr()), c_void_p(relu_45.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf83.data_ptr()))
    del buf4
    del buf49
    del buf7
    del buf71
    del buf73
    del convolution_45
    del convolution_46
    del le
    del primals_137
    del primals_140
    del relu_42
    del relu_45
    buf84 = aten.convolution_backward(buf83, relu_41, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf83
    del primals_136
    buf85 = buf84[0]
    assert_size_stride(buf85, (8, 512, 1, 1), (512, 1, 1, 1))
    buf2 = buf10; del buf10  # reuse
    buf87 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf88 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    kernel_cpp_8(c_void_p(convolution_44.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(relu_41.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()))
    buf28 = empty_strided((1000, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    extern_kernels.mm(as_strided(tangents_160, (1000, 8), (1, 1000)), view, out=buf28)
    del view
    buf29 = empty_strided((1, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    buf32 = as_strided(buf26, (2048, ), (1, )); del buf26  # reuse
    kernel_cpp_9(c_void_p(buf32.data_ptr()), c_void_p(tangents_160.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf29.data_ptr()))
    del buf31
    del tangents_160
    buf36 = buf34[1]
    assert_size_stride(buf36, (2048, 512, 1, 1), (512, 1, 1, 1))
    del buf34
    buf39 = as_strided(buf23, (512, ), (1, )); del buf23  # reuse
    kernel_cpp_10(c_void_p(buf39.data_ptr()), c_void_p(buf38.data_ptr()))
    buf43 = buf41[1]
    assert_size_stride(buf43, (512, 512, 3, 3), (4608, 9, 3, 1))
    del buf41
    buf46 = as_strided(buf20, (512, ), (1, )); del buf20  # reuse
    kernel_cpp_11(c_void_p(buf46.data_ptr()), c_void_p(buf45.data_ptr()))
    buf50 = buf48[1]
    assert_size_stride(buf50, (512, 2048, 1, 1), (2048, 1, 1, 1))
    del buf48
    buf54 = as_strided(buf17, (2048, ), (1, )); del buf17  # reuse
    kernel_cpp_12(c_void_p(buf54.data_ptr()), c_void_p(buf52.data_ptr()))
    del buf52
    buf58 = buf56[1]
    assert_size_stride(buf58, (2048, 512, 1, 1), (512, 1, 1, 1))
    del buf56
    buf61 = as_strided(buf14, (512, ), (1, )); del buf14  # reuse
    kernel_cpp_13(c_void_p(buf61.data_ptr()), c_void_p(buf60.data_ptr()))
    buf65 = buf63[1]
    assert_size_stride(buf65, (512, 512, 3, 3), (4608, 9, 3, 1))
    del buf63
    buf68 = as_strided(buf11, (512, ), (1, )); del buf11  # reuse
    kernel_cpp_14(c_void_p(buf68.data_ptr()), c_void_p(buf67.data_ptr()))
    buf72 = buf70[1]
    assert_size_stride(buf72, (512, 2048, 1, 1), (2048, 1, 1, 1))
    del buf70
    buf76 = buf75; del buf75  # reuse
    kernel_cpp_15(c_void_p(buf76.data_ptr()), c_void_p(buf8.data_ptr()))
    del buf8
    buf78 = aten.convolution_backward(buf77, relu_39, primals_139, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf77
    del primals_139
    buf79 = buf78[0]
    assert_size_stride(buf79, (8, 1024, 2, 2), (4096, 4, 2, 1))
    buf80 = buf78[1]
    assert_size_stride(buf80, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    del buf78
    buf82 = as_strided(buf5, (2048, ), (1, )); del buf5  # reuse
    kernel_cpp_16(c_void_p(buf82.data_ptr()), c_void_p(buf81.data_ptr()))
    del buf81
    buf86 = buf84[1]
    assert_size_stride(buf86, (2048, 512, 1, 1), (512, 1, 1, 1))
    del buf84
    buf89 = buf67; del buf67  # reuse
    buf90 = buf85; del buf85  # reuse
    kernel_cpp_17(c_void_p(buf90.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(relu_41.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(buf89.data_ptr()))
    del convolution_44
    del primals_134
    del relu_41
    buf91 = aten.convolution_backward(buf90, relu_40, primals_133, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf90
    del primals_133
    buf92 = buf91[0]
    assert_size_stride(buf92, (8, 512, 2, 2), (2048, 4, 2, 1))
    buf93 = buf91[1]
    assert_size_stride(buf93, (512, 512, 3, 3), (4608, 9, 3, 1))
    del buf91
    buf94 = buf88; del buf88  # reuse
    buf95 = as_strided(buf2, (512, ), (1, )); del buf2  # reuse
    buf96 = as_strided(buf1, (512, ), (1, )); del buf1  # reuse
    buf97 = buf92; del buf92  # reuse
    kernel_cpp_18(c_void_p(buf97.data_ptr()), c_void_p(relu_40.data_ptr()), c_void_p(convolution_43.data_ptr()), c_void_p(unsqueeze_322.data_ptr()), c_void_p(squeeze_130.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()))
    del convolution_43
    del primals_131
    del relu_40
    del squeeze_130
    del unsqueeze_322
    buf98 = aten.convolution_backward(buf97, relu_39, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf97
    del primals_130
    buf99 = buf98[0]
    assert_size_stride(buf99, (8, 1024, 2, 2), (4096, 4, 2, 1))
    buf100 = buf98[1]
    assert_size_stride(buf100, (512, 1024, 1, 1), (1024, 1, 1, 1))
    del buf98
    buf101 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf102 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf103 = empty_strided((8, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    buf104 = buf102; del buf102  # reuse
    kernel_cpp_19(c_void_p(buf104.data_ptr()), c_void_p(relu_39.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(unsqueeze_334.data_ptr()), c_void_p(squeeze_127.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf103.data_ptr()))
    del convolution_42
    del primals_128
    del squeeze_127
    del unsqueeze_334
    buf105 = aten.convolution_backward(buf103, relu_38, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_127
    buf106 = buf105[0]
    assert_size_stride(buf106, (8, 256, 2, 2), (1024, 4, 2, 1))
    buf107 = buf105[1]
    assert_size_stride(buf107, (1024, 256, 1, 1), (256, 1, 1, 1))
    del buf105
    buf108 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf109 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf110 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf111 = buf106; del buf106  # reuse
    kernel_cpp_20(c_void_p(buf111.data_ptr()), c_void_p(relu_38.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(unsqueeze_346.data_ptr()), c_void_p(squeeze_124.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()))
    del convolution_41
    del primals_125
    del relu_38
    del squeeze_124
    del unsqueeze_346
    buf112 = aten.convolution_backward(buf111, relu_37, primals_124, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf111
    del primals_124
    buf113 = buf112[0]
    assert_size_stride(buf113, (8, 256, 2, 2), (1024, 4, 2, 1))
    buf114 = buf112[1]
    assert_size_stride(buf114, (256, 256, 3, 3), (2304, 9, 3, 1))
    del buf112
    buf115 = buf109; del buf109  # reuse
    buf116 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf117 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf118 = buf113; del buf113  # reuse
    kernel_cpp_21(c_void_p(buf118.data_ptr()), c_void_p(relu_37.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(unsqueeze_358.data_ptr()), c_void_p(squeeze_121.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()))
    del convolution_40
    del primals_122
    del relu_37
    del squeeze_121
    del unsqueeze_358
    buf119 = aten.convolution_backward(buf118, relu_36, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf118
    del primals_121
    buf120 = buf119[0]
    assert_size_stride(buf120, (8, 1024, 2, 2), (4096, 4, 2, 1))
    buf121 = buf119[1]
    assert_size_stride(buf121, (256, 1024, 1, 1), (1024, 1, 1, 1))
    del buf119
    buf122 = buf120; del buf120  # reuse
    buf123 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf124 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf125 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf126 = buf103; del buf103  # reuse
    kernel_cpp_22(c_void_p(buf122.data_ptr()), c_void_p(relu_36.data_ptr()), c_void_p(relu_39.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(unsqueeze_370.data_ptr()), c_void_p(squeeze_118.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()))
    del buf79
    del buf99
    del convolution_39
    del primals_119
    del relu_36
    del relu_39
    del squeeze_118
    del unsqueeze_370
    buf127 = aten.convolution_backward(buf126, relu_35, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_118
    buf128 = buf127[0]
    assert_size_stride(buf128, (8, 256, 2, 2), (1024, 4, 2, 1))
    buf129 = buf127[1]
    assert_size_stride(buf129, (1024, 256, 1, 1), (256, 1, 1, 1))
    del buf127
    buf130 = buf116; del buf116  # reuse
    buf131 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf132 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf133 = buf128; del buf128  # reuse
    kernel_cpp_23(c_void_p(buf133.data_ptr()), c_void_p(relu_35.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(unsqueeze_382.data_ptr()), c_void_p(squeeze_115.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()))
    del convolution_38
    del primals_116
    del relu_35
    del squeeze_115
    del unsqueeze_382
    buf134 = aten.convolution_backward(buf133, relu_34, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf133
    del primals_115
    buf135 = buf134[0]
    assert_size_stride(buf135, (8, 256, 2, 2), (1024, 4, 2, 1))
    buf136 = buf134[1]
    assert_size_stride(buf136, (256, 256, 3, 3), (2304, 9, 3, 1))
    del buf134
    buf137 = buf131; del buf131  # reuse
    buf138 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf139 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf140 = buf135; del buf135  # reuse
    kernel_cpp_24(c_void_p(buf140.data_ptr()), c_void_p(relu_34.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(unsqueeze_394.data_ptr()), c_void_p(squeeze_112.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()))
    del convolution_37
    del primals_113
    del relu_34
    del squeeze_112
    del unsqueeze_394
    buf141 = aten.convolution_backward(buf140, relu_33, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf140
    del primals_112
    buf142 = buf141[0]
    assert_size_stride(buf142, (8, 1024, 2, 2), (4096, 4, 2, 1))
    buf143 = buf141[1]
    assert_size_stride(buf143, (256, 1024, 1, 1), (1024, 1, 1, 1))
    del buf141
    buf144 = buf124; del buf124  # reuse
    buf145 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf146 = buf126; del buf126  # reuse
    buf147 = buf145; del buf145  # reuse
    kernel_cpp_25(c_void_p(buf147.data_ptr()), c_void_p(relu_33.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(unsqueeze_406.data_ptr()), c_void_p(squeeze_109.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf146.data_ptr()))
    del convolution_36
    del primals_110
    del squeeze_109
    del unsqueeze_406
    buf148 = aten.convolution_backward(buf146, relu_32, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_109
    buf149 = buf148[0]
    assert_size_stride(buf149, (8, 256, 2, 2), (1024, 4, 2, 1))
    buf150 = buf148[1]
    assert_size_stride(buf150, (1024, 256, 1, 1), (256, 1, 1, 1))
    del buf148
    buf151 = buf138; del buf138  # reuse
    buf152 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf153 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf154 = buf149; del buf149  # reuse
    kernel_cpp_26(c_void_p(buf154.data_ptr()), c_void_p(relu_32.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(unsqueeze_418.data_ptr()), c_void_p(squeeze_106.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()))
    del convolution_35
    del primals_107
    del relu_32
    del squeeze_106
    del unsqueeze_418
    buf155 = aten.convolution_backward(buf154, relu_31, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf154
    del primals_106
    buf156 = buf155[0]
    assert_size_stride(buf156, (8, 256, 2, 2), (1024, 4, 2, 1))
    buf157 = buf155[1]
    assert_size_stride(buf157, (256, 256, 3, 3), (2304, 9, 3, 1))
    del buf155
    buf158 = buf152; del buf152  # reuse
    buf159 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf160 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf161 = buf156; del buf156  # reuse
    kernel_cpp_27(c_void_p(buf161.data_ptr()), c_void_p(relu_31.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(unsqueeze_430.data_ptr()), c_void_p(squeeze_103.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()))
    del convolution_34
    del primals_104
    del relu_31
    del squeeze_103
    del unsqueeze_430
    buf162 = aten.convolution_backward(buf161, relu_30, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf161
    del primals_103
    buf163 = buf162[0]
    assert_size_stride(buf163, (8, 1024, 2, 2), (4096, 4, 2, 1))
    buf164 = buf162[1]
    assert_size_stride(buf164, (256, 1024, 1, 1), (1024, 1, 1, 1))
    del buf162
    buf165 = buf122; del buf122  # reuse
    buf166 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf167 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf168 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf169 = buf146; del buf146  # reuse
    kernel_cpp_28(c_void_p(buf165.data_ptr()), c_void_p(relu_30.data_ptr()), c_void_p(relu_33.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(unsqueeze_442.data_ptr()), c_void_p(squeeze_100.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()))
    del buf142
    del convolution_33
    del primals_101
    del relu_30
    del relu_33
    del squeeze_100
    del unsqueeze_442
    buf170 = aten.convolution_backward(buf169, relu_29, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_100
    buf171 = buf170[0]
    assert_size_stride(buf171, (8, 256, 2, 2), (1024, 4, 2, 1))
    buf172 = buf170[1]
    assert_size_stride(buf172, (1024, 256, 1, 1), (256, 1, 1, 1))
    del buf170
    buf173 = buf159; del buf159  # reuse
    buf174 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf175 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf176 = buf171; del buf171  # reuse
    kernel_cpp_29(c_void_p(buf176.data_ptr()), c_void_p(relu_29.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(unsqueeze_454.data_ptr()), c_void_p(squeeze_97.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()))
    del convolution_32
    del primals_98
    del relu_29
    del squeeze_97
    del unsqueeze_454
    buf177 = aten.convolution_backward(buf176, relu_28, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf176
    del primals_97
    buf178 = buf177[0]
    assert_size_stride(buf178, (8, 256, 2, 2), (1024, 4, 2, 1))
    buf179 = buf177[1]
    assert_size_stride(buf179, (256, 256, 3, 3), (2304, 9, 3, 1))
    del buf177
    buf180 = buf174; del buf174  # reuse
    buf181 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf182 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf183 = buf178; del buf178  # reuse
    kernel_cpp_30(c_void_p(buf183.data_ptr()), c_void_p(relu_28.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(unsqueeze_466.data_ptr()), c_void_p(squeeze_94.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()))
    del convolution_31
    del primals_95
    del relu_28
    del squeeze_94
    del unsqueeze_466
    buf184 = aten.convolution_backward(buf183, relu_27, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf183
    del primals_94
    buf185 = buf184[0]
    assert_size_stride(buf185, (8, 1024, 2, 2), (4096, 4, 2, 1))
    buf186 = buf184[1]
    assert_size_stride(buf186, (256, 1024, 1, 1), (1024, 1, 1, 1))
    del buf184
    buf187 = buf167; del buf167  # reuse
    buf188 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf189 = buf169; del buf169  # reuse
    buf190 = buf188; del buf188  # reuse
    kernel_cpp_31(c_void_p(buf190.data_ptr()), c_void_p(relu_27.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(unsqueeze_478.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf189.data_ptr()))
    del convolution_30
    del primals_92
    del squeeze_91
    del unsqueeze_478
    buf191 = aten.convolution_backward(buf189, relu_26, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_91
    buf192 = buf191[0]
    assert_size_stride(buf192, (8, 256, 2, 2), (1024, 4, 2, 1))
    buf193 = buf191[1]
    assert_size_stride(buf193, (1024, 256, 1, 1), (256, 1, 1, 1))
    del buf191
    buf194 = buf181; del buf181  # reuse
    buf195 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf196 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf197 = buf192; del buf192  # reuse
    kernel_cpp_32(c_void_p(buf197.data_ptr()), c_void_p(relu_26.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(unsqueeze_490.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()))
    del convolution_29
    del primals_89
    del relu_26
    del squeeze_88
    del unsqueeze_490
    buf198 = aten.convolution_backward(buf197, relu_25, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf197
    del primals_88
    buf199 = buf198[0]
    assert_size_stride(buf199, (8, 256, 2, 2), (1024, 4, 2, 1))
    buf200 = buf198[1]
    assert_size_stride(buf200, (256, 256, 3, 3), (2304, 9, 3, 1))
    del buf198
    buf201 = buf195; del buf195  # reuse
    buf202 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf203 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf204 = buf199; del buf199  # reuse
    kernel_cpp_33(c_void_p(buf204.data_ptr()), c_void_p(relu_25.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(unsqueeze_502.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()))
    del convolution_28
    del primals_86
    del relu_25
    del squeeze_85
    del unsqueeze_502
    buf205 = aten.convolution_backward(buf204, relu_24, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf204
    del primals_85
    buf206 = buf205[0]
    assert_size_stride(buf206, (8, 1024, 2, 2), (4096, 4, 2, 1))
    buf207 = buf205[1]
    assert_size_stride(buf207, (256, 1024, 1, 1), (1024, 1, 1, 1))
    del buf205
    buf208 = buf165; del buf165  # reuse
    buf209 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf210 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf216 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf211 = empty_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    buf212 = buf189; del buf189  # reuse
    buf218 = buf163; del buf163  # reuse
    kernel_cpp_34(c_void_p(buf208.data_ptr()), c_void_p(relu_24.data_ptr()), c_void_p(relu_27.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(unsqueeze_514.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(unsqueeze_526.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf218.data_ptr()))
    del buf185
    del buf206
    del buf208
    del buf210
    del convolution_26
    del convolution_27
    del primals_80
    del primals_83
    del relu_24
    del relu_27
    del squeeze_82
    del unsqueeze_514
    del unsqueeze_526
    buf213 = aten.convolution_backward(buf212, relu_21, primals_82, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf212
    del primals_82
    buf214 = buf213[0]
    assert_size_stride(buf214, (8, 512, 4, 4), (8192, 16, 4, 1))
    buf215 = buf213[1]
    assert_size_stride(buf215, (1024, 512, 1, 1), (512, 1, 1, 1))
    del buf213
    buf217 = buf216; del buf216  # reuse
    kernel_cpp_35(c_void_p(buf217.data_ptr()), c_void_p(squeeze_79.data_ptr()))
    del squeeze_79
    buf219 = aten.convolution_backward(buf218, relu_23, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf218
    del primals_79
    buf220 = buf219[0]
    assert_size_stride(buf220, (8, 256, 2, 2), (1024, 4, 2, 1))
    buf221 = buf219[1]
    assert_size_stride(buf221, (1024, 256, 1, 1), (256, 1, 1, 1))
    del buf219
    buf222 = buf202; del buf202  # reuse
    buf223 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf224 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf225 = buf220; del buf220  # reuse
    kernel_cpp_36(c_void_p(buf225.data_ptr()), c_void_p(relu_23.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(unsqueeze_538.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()))
    del convolution_25
    del primals_77
    del relu_23
    del squeeze_76
    del unsqueeze_538
    buf226 = aten.convolution_backward(buf225, relu_22, primals_76, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf225
    del primals_76
    buf227 = buf226[0]
    assert_size_stride(buf227, (8, 256, 4, 4), (4096, 16, 4, 1))
    buf228 = buf226[1]
    assert_size_stride(buf228, (256, 256, 3, 3), (2304, 9, 3, 1))
    del buf226
    buf229 = buf223; del buf223  # reuse
    buf230 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf231 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf232 = buf227; del buf227  # reuse
    kernel_cpp_37(c_void_p(buf232.data_ptr()), c_void_p(relu_22.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(unsqueeze_550.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()))
    del convolution_24
    del primals_74
    del relu_22
    del squeeze_73
    del unsqueeze_550
    buf233 = aten.convolution_backward(buf232, relu_21, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf232
    del primals_73
    buf234 = buf233[0]
    assert_size_stride(buf234, (8, 512, 4, 4), (8192, 16, 4, 1))
    buf235 = buf233[1]
    assert_size_stride(buf235, (256, 512, 1, 1), (512, 1, 1, 1))
    del buf233
    buf236 = buf95; del buf95  # reuse
    buf237 = buf60; del buf60  # reuse
    buf238 = empty_strided((8, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    buf239 = buf237; del buf237  # reuse
    kernel_cpp_38(c_void_p(buf239.data_ptr()), c_void_p(relu_21.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(unsqueeze_562.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf238.data_ptr()))
    del convolution_23
    del primals_71
    del squeeze_70
    del unsqueeze_562
    buf240 = aten.convolution_backward(buf238, relu_20, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_70
    buf241 = buf240[0]
    assert_size_stride(buf241, (8, 128, 4, 4), (2048, 16, 4, 1))
    buf242 = buf240[1]
    assert_size_stride(buf242, (512, 128, 1, 1), (128, 1, 1, 1))
    del buf240
    buf243 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf244 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf245 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf246 = buf241; del buf241  # reuse
    kernel_cpp_39(c_void_p(buf246.data_ptr()), c_void_p(relu_20.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(unsqueeze_574.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()))
    del convolution_22
    del primals_68
    del relu_20
    del squeeze_67
    del unsqueeze_574
    buf247 = aten.convolution_backward(buf246, relu_19, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf246
    del primals_67
    buf248 = buf247[0]
    assert_size_stride(buf248, (8, 128, 4, 4), (2048, 16, 4, 1))
    buf249 = buf247[1]
    assert_size_stride(buf249, (128, 128, 3, 3), (1152, 9, 3, 1))
    del buf247
    buf250 = buf244; del buf244  # reuse
    buf251 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf252 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf253 = buf248; del buf248  # reuse
    kernel_cpp_40(c_void_p(buf253.data_ptr()), c_void_p(relu_19.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(unsqueeze_586.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()))
    del convolution_21
    del primals_65
    del relu_19
    del squeeze_64
    del unsqueeze_586
    buf254 = aten.convolution_backward(buf253, relu_18, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf253
    del primals_64
    buf255 = buf254[0]
    assert_size_stride(buf255, (8, 512, 4, 4), (8192, 16, 4, 1))
    buf256 = buf254[1]
    assert_size_stride(buf256, (128, 512, 1, 1), (512, 1, 1, 1))
    del buf254
    buf257 = buf214; del buf214  # reuse
    buf258 = buf45; del buf45  # reuse
    buf259 = buf38; del buf38  # reuse
    buf260 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf261 = buf238; del buf238  # reuse
    kernel_cpp_41(c_void_p(buf257.data_ptr()), c_void_p(relu_18.data_ptr()), c_void_p(relu_21.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(unsqueeze_598.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()))
    del buf234
    del convolution_20
    del primals_62
    del relu_18
    del relu_21
    del squeeze_61
    del unsqueeze_598
    buf262 = aten.convolution_backward(buf261, relu_17, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_61
    buf263 = buf262[0]
    assert_size_stride(buf263, (8, 128, 4, 4), (2048, 16, 4, 1))
    buf264 = buf262[1]
    assert_size_stride(buf264, (512, 128, 1, 1), (128, 1, 1, 1))
    del buf262
    buf265 = buf251; del buf251  # reuse
    buf266 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf267 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf268 = buf263; del buf263  # reuse
    kernel_cpp_42(c_void_p(buf268.data_ptr()), c_void_p(relu_17.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(unsqueeze_610.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()))
    del convolution_19
    del primals_59
    del relu_17
    del squeeze_58
    del unsqueeze_610
    buf269 = aten.convolution_backward(buf268, relu_16, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf268
    del primals_58
    buf270 = buf269[0]
    assert_size_stride(buf270, (8, 128, 4, 4), (2048, 16, 4, 1))
    buf271 = buf269[1]
    assert_size_stride(buf271, (128, 128, 3, 3), (1152, 9, 3, 1))
    del buf269
    buf272 = buf266; del buf266  # reuse
    buf273 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf274 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf275 = buf270; del buf270  # reuse
    kernel_cpp_43(c_void_p(buf275.data_ptr()), c_void_p(relu_16.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(unsqueeze_622.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()))
    del convolution_18
    del primals_56
    del relu_16
    del squeeze_55
    del unsqueeze_622
    buf276 = aten.convolution_backward(buf275, relu_15, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf275
    del primals_55
    buf277 = buf276[0]
    assert_size_stride(buf277, (8, 512, 4, 4), (8192, 16, 4, 1))
    buf278 = buf276[1]
    assert_size_stride(buf278, (128, 512, 1, 1), (512, 1, 1, 1))
    del buf276
    buf279 = buf259; del buf259  # reuse
    buf280 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf281 = buf261; del buf261  # reuse
    buf282 = buf280; del buf280  # reuse
    kernel_cpp_44(c_void_p(buf282.data_ptr()), c_void_p(relu_15.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(unsqueeze_634.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf281.data_ptr()))
    del convolution_17
    del primals_53
    del squeeze_52
    del unsqueeze_634
    buf283 = aten.convolution_backward(buf281, relu_14, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_52
    buf284 = buf283[0]
    assert_size_stride(buf284, (8, 128, 4, 4), (2048, 16, 4, 1))
    buf285 = buf283[1]
    assert_size_stride(buf285, (512, 128, 1, 1), (128, 1, 1, 1))
    del buf283
    buf286 = buf273; del buf273  # reuse
    buf287 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf288 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf289 = buf284; del buf284  # reuse
    kernel_cpp_45(c_void_p(buf289.data_ptr()), c_void_p(relu_14.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(unsqueeze_646.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()))
    del convolution_16
    del primals_50
    del relu_14
    del squeeze_49
    del unsqueeze_646
    buf290 = aten.convolution_backward(buf289, relu_13, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf289
    del primals_49
    buf291 = buf290[0]
    assert_size_stride(buf291, (8, 128, 4, 4), (2048, 16, 4, 1))
    buf292 = buf290[1]
    assert_size_stride(buf292, (128, 128, 3, 3), (1152, 9, 3, 1))
    del buf290
    buf293 = buf287; del buf287  # reuse
    buf294 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf295 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf296 = buf291; del buf291  # reuse
    kernel_cpp_46(c_void_p(buf296.data_ptr()), c_void_p(relu_13.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_658.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()))
    del convolution_15
    del primals_47
    del relu_13
    del squeeze_46
    del unsqueeze_658
    buf297 = aten.convolution_backward(buf296, relu_12, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf296
    del primals_46
    buf298 = buf297[0]
    assert_size_stride(buf298, (8, 512, 4, 4), (8192, 16, 4, 1))
    buf299 = buf297[1]
    assert_size_stride(buf299, (128, 512, 1, 1), (512, 1, 1, 1))
    del buf297
    buf300 = buf257; del buf257  # reuse
    buf301 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf302 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf308 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf303 = empty_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    buf304 = buf281; del buf281  # reuse
    buf310 = buf255; del buf255  # reuse
    kernel_cpp_47(c_void_p(buf300.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(relu_15.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(unsqueeze_670.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(unsqueeze_682.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf310.data_ptr()))
    del buf277
    del buf298
    del buf300
    del buf302
    del convolution_13
    del convolution_14
    del primals_41
    del primals_44
    del relu_12
    del relu_15
    del squeeze_43
    del unsqueeze_670
    del unsqueeze_682
    buf305 = aten.convolution_backward(buf304, relu_9, primals_43, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf304
    del primals_43
    buf306 = buf305[0]
    assert_size_stride(buf306, (8, 256, 8, 8), (16384, 64, 8, 1))
    buf307 = buf305[1]
    assert_size_stride(buf307, (512, 256, 1, 1), (256, 1, 1, 1))
    del buf305
    buf309 = buf308; del buf308  # reuse
    kernel_cpp_48(c_void_p(buf309.data_ptr()), c_void_p(squeeze_40.data_ptr()))
    del squeeze_40
    buf311 = aten.convolution_backward(buf310, relu_11, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf310
    del primals_40
    buf312 = buf311[0]
    assert_size_stride(buf312, (8, 128, 4, 4), (2048, 16, 4, 1))
    buf313 = buf311[1]
    assert_size_stride(buf313, (512, 128, 1, 1), (128, 1, 1, 1))
    del buf311
    buf314 = buf294; del buf294  # reuse
    buf315 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf316 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf317 = buf312; del buf312  # reuse
    kernel_cpp_49(c_void_p(buf317.data_ptr()), c_void_p(relu_11.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(unsqueeze_694.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()))
    del convolution_12
    del primals_38
    del relu_11
    del squeeze_37
    del unsqueeze_694
    buf318 = aten.convolution_backward(buf317, relu_10, primals_37, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf317
    del primals_37
    buf319 = buf318[0]
    assert_size_stride(buf319, (8, 128, 8, 8), (8192, 64, 8, 1))
    buf320 = buf318[1]
    assert_size_stride(buf320, (128, 128, 3, 3), (1152, 9, 3, 1))
    del buf318
    buf321 = buf315; del buf315  # reuse
    buf322 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf323 = empty_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    buf324 = buf319; del buf319  # reuse
    kernel_cpp_50(c_void_p(buf324.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(unsqueeze_706.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()))
    del buf322
    del convolution_11
    del primals_35
    del relu_10
    del squeeze_34
    del unsqueeze_706
    buf325 = aten.convolution_backward(buf324, relu_9, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf324
    del primals_34
    buf326 = buf325[0]
    assert_size_stride(buf326, (8, 256, 8, 8), (16384, 64, 8, 1))
    buf327 = buf325[1]
    assert_size_stride(buf327, (128, 256, 1, 1), (256, 1, 1, 1))
    del buf325
    buf328 = buf230; del buf230  # reuse
    buf329 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf330 = empty_strided((8, 256, 8, 8), (16384, 64, 8, 1), device='cpu', dtype=torch.float32)
    buf331 = buf329; del buf329  # reuse
    kernel_cpp_51(c_void_p(buf331.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(unsqueeze_718.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf330.data_ptr()))
    del convolution_10
    del primals_32
    del squeeze_31
    del unsqueeze_718
    buf332 = aten.convolution_backward(buf330, relu_8, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_31
    buf333 = buf332[0]
    assert_size_stride(buf333, (8, 64, 8, 8), (4096, 64, 8, 1))
    buf334 = buf332[1]
    assert_size_stride(buf334, (256, 64, 1, 1), (64, 1, 1, 1))
    del buf332
    buf335 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf336 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf337 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf338 = buf333; del buf333  # reuse
    kernel_cpp_52(c_void_p(buf338.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_730.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf337.data_ptr()))
    del convolution_9
    del primals_29
    del relu_8
    del squeeze_28
    del unsqueeze_730
    buf339 = aten.convolution_backward(buf338, relu_7, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf338
    del primals_28
    buf340 = buf339[0]
    assert_size_stride(buf340, (8, 64, 8, 8), (4096, 64, 8, 1))
    buf341 = buf339[1]
    assert_size_stride(buf341, (64, 64, 3, 3), (576, 9, 3, 1))
    del buf339
    buf342 = buf336; del buf336  # reuse
    buf343 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf344 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf345 = buf340; del buf340  # reuse
    kernel_cpp_53(c_void_p(buf345.data_ptr()), c_void_p(relu_7.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(unsqueeze_742.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()))
    del convolution_8
    del primals_26
    del relu_7
    del squeeze_25
    del unsqueeze_742
    buf346 = aten.convolution_backward(buf345, relu_6, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf345
    del primals_25
    buf347 = buf346[0]
    assert_size_stride(buf347, (8, 256, 8, 8), (16384, 64, 8, 1))
    buf348 = buf346[1]
    assert_size_stride(buf348, (64, 256, 1, 1), (256, 1, 1, 1))
    del buf346
    buf349 = buf306; del buf306  # reuse
    buf350 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf351 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf352 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf353 = buf330; del buf330  # reuse
    kernel_cpp_54(c_void_p(buf349.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(unsqueeze_754.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()))
    del buf326
    del convolution_7
    del primals_23
    del relu_6
    del relu_9
    del squeeze_22
    del unsqueeze_754
    buf354 = aten.convolution_backward(buf353, relu_5, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_22
    buf355 = buf354[0]
    assert_size_stride(buf355, (8, 64, 8, 8), (4096, 64, 8, 1))
    buf356 = buf354[1]
    assert_size_stride(buf356, (256, 64, 1, 1), (64, 1, 1, 1))
    del buf354
    buf357 = buf343; del buf343  # reuse
    buf358 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf359 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf360 = buf355; del buf355  # reuse
    kernel_cpp_55(c_void_p(buf360.data_ptr()), c_void_p(relu_5.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(unsqueeze_766.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()))
    del convolution_6
    del primals_20
    del relu_5
    del squeeze_19
    del unsqueeze_766
    buf361 = aten.convolution_backward(buf360, relu_4, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf360
    del primals_19
    buf362 = buf361[0]
    assert_size_stride(buf362, (8, 64, 8, 8), (4096, 64, 8, 1))
    buf363 = buf361[1]
    assert_size_stride(buf363, (64, 64, 3, 3), (576, 9, 3, 1))
    del buf361
    buf364 = buf358; del buf358  # reuse
    buf365 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf366 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf367 = buf362; del buf362  # reuse
    kernel_cpp_56(c_void_p(buf367.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(unsqueeze_778.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf366.data_ptr()))
    del convolution_5
    del primals_17
    del relu_4
    del squeeze_16
    del unsqueeze_778
    buf368 = aten.convolution_backward(buf367, relu_3, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf367
    del primals_16
    buf369 = buf368[0]
    assert_size_stride(buf369, (8, 256, 8, 8), (16384, 64, 8, 1))
    buf370 = buf368[1]
    assert_size_stride(buf370, (64, 256, 1, 1), (256, 1, 1, 1))
    del buf368
    buf371 = buf351; del buf351  # reuse
    buf372 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf378 = empty_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    buf373 = buf353; del buf353  # reuse
    buf379 = buf347; del buf347  # reuse
    buf374 = buf372; del buf372  # reuse
    kernel_cpp_57(c_void_p(buf374.data_ptr()), c_void_p(relu_3.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_790.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(unsqueeze_802.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf379.data_ptr()))
    del buf349
    del buf369
    del convolution_3
    del convolution_4
    del primals_11
    del primals_14
    del relu_3
    del squeeze_13
    del unsqueeze_790
    del unsqueeze_802
    buf375 = aten.convolution_backward(buf373, getitem_2, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf373
    del primals_13
    buf376 = buf375[0]
    assert_size_stride(buf376, (8, 64, 8, 8), (4096, 64, 8, 1))
    buf377 = buf375[1]
    assert_size_stride(buf377, (256, 64, 1, 1), (64, 1, 1, 1))
    del buf375
    buf380 = buf378; del buf378  # reuse
    kernel_cpp_58(c_void_p(buf380.data_ptr()), c_void_p(squeeze_10.data_ptr()))
    del squeeze_10
    buf381 = aten.convolution_backward(buf379, relu_2, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_10
    buf382 = buf381[0]
    assert_size_stride(buf382, (8, 64, 8, 8), (4096, 64, 8, 1))
    buf383 = buf381[1]
    assert_size_stride(buf383, (256, 64, 1, 1), (64, 1, 1, 1))
    del buf381
    buf384 = buf365; del buf365  # reuse
    buf385 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf386 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf387 = buf382; del buf382  # reuse
    kernel_cpp_59(c_void_p(buf387.data_ptr()), c_void_p(relu_2.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_814.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()))
    del convolution_2
    del primals_8
    del relu_2
    del squeeze_7
    del unsqueeze_814
    buf388 = aten.convolution_backward(buf387, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf387
    del primals_7
    buf389 = buf388[0]
    assert_size_stride(buf389, (8, 64, 8, 8), (4096, 64, 8, 1))
    buf390 = buf388[1]
    assert_size_stride(buf390, (64, 64, 3, 3), (576, 9, 3, 1))
    del buf388
    buf391 = buf385; del buf385  # reuse
    buf392 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf393 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf394 = buf389; del buf389  # reuse
    kernel_cpp_60(c_void_p(buf394.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_826.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf393.data_ptr()))
    del convolution_1
    del primals_5
    del relu_1
    del squeeze_4
    del unsqueeze_826
    buf395 = aten.convolution_backward(buf394, getitem_2, primals_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf394
    del getitem_2
    del primals_4
    buf396 = buf395[0]
    assert_size_stride(buf396, (8, 64, 8, 8), (4096, 64, 8, 1))
    buf397 = buf395[1]
    assert_size_stride(buf397, (64, 64, 1, 1), (64, 1, 1, 1))
    del buf395
    buf398 = buf376; del buf376  # reuse
    buf399 = as_strided(buf379, (8, 64, 16, 16), (16384, 256, 16, 1)); del buf379  # reuse
    buf400 = buf392; del buf392  # reuse
    buf401 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf402 = empty_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    buf403 = buf399; del buf399  # reuse
    kernel_cpp_61(c_void_p(buf398.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(getitem_3.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_838.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf402.data_ptr()))
    del buf396
    del buf398
    del buf401
    del convolution
    del getitem_3
    del primals_2
    del relu
    del squeeze_1
    del unsqueeze_838
    buf404 = aten.convolution_backward(buf403, primals_321, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf403
    del primals_1
    del primals_321
    buf405 = buf404[1]
    assert_size_stride(buf405, (64, 3, 7, 7), (147, 49, 7, 1))
    del buf404
    return (buf405, buf402, buf400, buf397, buf393, buf391, buf390, buf386, buf384, buf383, buf380, buf371, buf377, buf374, buf371, buf370, buf366, buf364, buf363, buf359, buf357, buf356, buf352, buf350, buf348, buf344, buf342, buf341, buf337, buf335, buf334, buf331, buf328, buf327, buf323, buf321, buf320, buf316, buf314, buf313, buf309, buf301, buf307, buf303, buf301, buf299, buf295, buf293, buf292, buf288, buf286, buf285, buf282, buf279, buf278, buf274, buf272, buf271, buf267, buf265, buf264, buf260, buf258, buf256, buf252, buf250, buf249, buf245, buf243, buf242, buf239, buf236, buf235, buf231, buf229, buf228, buf224, buf222, buf221, buf217, buf209, buf215, buf211, buf209, buf207, buf203, buf201, buf200, buf196, buf194, buf193, buf190, buf187, buf186, buf182, buf180, buf179, buf175, buf173, buf172, buf168, buf166, buf164, buf160, buf158, buf157, buf153, buf151, buf150, buf147, buf144, buf143, buf139, buf137, buf136, buf132, buf130, buf129, buf125, buf123, buf121, buf117, buf115, buf114, buf110, buf108, buf107, buf104, buf101, buf100, buf96, buf94, buf93, buf89, buf87, buf86, buf82, buf74, buf80, buf76, buf74, buf72, buf68, buf66, buf65, buf61, buf59, buf58, buf54, buf51, buf50, buf46, buf44, buf43, buf39, buf37, buf36, buf32, buf30, as_strided(buf28, (1000, 2048), (2048, 1)), as_strided(buf29, (1000, ), (1, )), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


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
    primals_321 = rand_strided((8, 3, 32, 32), (3072, 1024, 32, 1), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 64, 16, 16), (16384, 256, 16, 1), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu = rand_strided((8, 64, 16, 16), (16384, 256, 16, 1), device='cpu', dtype=torch.float32)
    getitem_2 = rand_strided((8, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((8, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.int64)
    convolution_1 = rand_strided((8, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((8, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((8, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((8, 256, 8, 8), (16384, 64, 8, 1), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 256, 8, 8), (16384, 64, 8, 1), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((8, 256, 8, 8), (16384, 64, 8, 1), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((8, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((8, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 256, 8, 8), (16384, 64, 8, 1), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((8, 256, 8, 8), (16384, 64, 8, 1), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_7 = rand_strided((8, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((8, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 256, 8, 8), (16384, 64, 8, 1), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((8, 256, 8, 8), (16384, 64, 8, 1), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((8, 128, 8, 8), (8192, 64, 8, 1), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((8, 128, 8, 8), (8192, 64, 8, 1), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_11 = rand_strided((8, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_12 = rand_strided((8, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_13 = rand_strided((8, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((8, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_14 = rand_strided((8, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((8, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_15 = rand_strided((8, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_16 = rand_strided((8, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_17 = rand_strided((8, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_18 = rand_strided((8, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_19 = rand_strided((8, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((8, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_20 = rand_strided((8, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_21 = rand_strided((8, 512, 4, 4), (8192, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((8, 256, 4, 4), (4096, 16, 4, 1), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_22 = rand_strided((8, 256, 4, 4), (4096, 16, 4, 1), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_23 = rand_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((8, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    relu_24 = rand_strided((8, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_25 = rand_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_26 = rand_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    relu_27 = rand_strided((8, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_94 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_28 = rand_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_97 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_29 = rand_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_100 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    relu_30 = rand_strided((8, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_103 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_31 = rand_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_106 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_32 = rand_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((8, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_109 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    relu_33 = rand_strided((8, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_112 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_34 = rand_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_115 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_35 = rand_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((8, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_118 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    relu_36 = rand_strided((8, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_121 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_37 = rand_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_124 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_38 = rand_strided((8, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((8, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_127 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    relu_39 = rand_strided((8, 1024, 2, 2), (4096, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((8, 512, 2, 2), (2048, 4, 2, 1), device='cpu', dtype=torch.float32)
    squeeze_130 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_40 = rand_strided((8, 512, 2, 2), (2048, 4, 2, 1), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    relu_41 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((8, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((8, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    relu_42 = rand_strided((8, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    relu_43 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    relu_44 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((8, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    relu_45 = rand_strided((8, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    relu_46 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    relu_47 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    convolution_52 = rand_strided((8, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    view = rand_strided((8, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le = rand_strided((8, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.bool)
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
    tangents_160 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_321, convolution, squeeze_1, relu, getitem_2, getitem_3, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, relu_4, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, relu_8, convolution_10, squeeze_31, relu_9, convolution_11, squeeze_34, relu_10, convolution_12, squeeze_37, relu_11, convolution_13, squeeze_40, convolution_14, squeeze_43, relu_12, convolution_15, squeeze_46, relu_13, convolution_16, squeeze_49, relu_14, convolution_17, squeeze_52, relu_15, convolution_18, squeeze_55, relu_16, convolution_19, squeeze_58, relu_17, convolution_20, squeeze_61, relu_18, convolution_21, squeeze_64, relu_19, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, relu_21, convolution_24, squeeze_73, relu_22, convolution_25, squeeze_76, relu_23, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_24, convolution_28, squeeze_85, relu_25, convolution_29, squeeze_88, relu_26, convolution_30, squeeze_91, relu_27, convolution_31, squeeze_94, relu_28, convolution_32, squeeze_97, relu_29, convolution_33, squeeze_100, relu_30, convolution_34, squeeze_103, relu_31, convolution_35, squeeze_106, relu_32, convolution_36, squeeze_109, relu_33, convolution_37, squeeze_112, relu_34, convolution_38, squeeze_115, relu_35, convolution_39, squeeze_118, relu_36, convolution_40, squeeze_121, relu_37, convolution_41, squeeze_124, relu_38, convolution_42, squeeze_127, relu_39, convolution_43, squeeze_130, relu_40, convolution_44, relu_41, convolution_45, convolution_46, relu_42, convolution_47, relu_43, convolution_48, relu_44, convolution_49, relu_45, convolution_50, relu_46, convolution_51, relu_47, convolution_52, view, permute_1, le, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50, tangents_51, tangents_52, tangents_53, tangents_54, tangents_55, tangents_56, tangents_57, tangents_58, tangents_59, tangents_60, tangents_61, tangents_62, tangents_63, tangents_64, tangents_65, tangents_66, tangents_67, tangents_68, tangents_69, tangents_70, tangents_71, tangents_72, tangents_73, tangents_74, tangents_75, tangents_76, tangents_77, tangents_78, tangents_79, tangents_80, tangents_81, tangents_82, tangents_83, tangents_84, tangents_85, tangents_86, tangents_87, tangents_88, tangents_89, tangents_90, tangents_91, tangents_92, tangents_93, tangents_94, tangents_95, tangents_96, tangents_97, tangents_98, tangents_99, tangents_100, tangents_101, tangents_102, tangents_103, tangents_104, tangents_105, tangents_106, tangents_107, tangents_108, tangents_109, tangents_110, tangents_111, tangents_112, tangents_113, tangents_114, tangents_115, tangents_116, tangents_117, tangents_118, tangents_119, tangents_120, tangents_121, tangents_122, tangents_123, tangents_124, tangents_125, tangents_126, tangents_127, tangents_128, tangents_129, tangents_130, tangents_131, tangents_132, tangents_133, tangents_134, tangents_135, tangents_136, tangents_137, tangents_138, tangents_139, tangents_140, tangents_141, tangents_142, tangents_143, tangents_144, tangents_145, tangents_146, tangents_147, tangents_148, tangents_149, tangents_150, tangents_151, tangents_152, tangents_153, tangents_154, tangents_155, tangents_156, tangents_157, tangents_158, tangents_159, tangents_160]))
