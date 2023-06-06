
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
                       float* __restrict__ out_ptr0,
                       long* __restrict__ out_ptr1)
{
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<75264; i0+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + 16*i0);
            }
            #pragma omp for simd simdlen(8) 
            for(long i0=1204224; i0<1204224; i0+=1)
            {
                auto tmp0 = in_out_ptr0[i0];
                auto tmp1 = tmp0 * (tmp0>0);
                in_out_ptr0[i0] = tmp1;
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<21504; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<14; i1+=1)
                {
                    auto tmp0 = in_out_ptr0[(2*i1) + (56*i0)];
                    auto tmp1 = in_out_ptr0[1 + (2*i1) + (56*i0)];
                    auto tmp3 = in_out_ptr0[28 + (2*i1) + (56*i0)];
                    auto tmp5 = in_out_ptr0[29 + (2*i1) + (56*i0)];
                    auto tmp2 = (tmp0 != tmp0) ? tmp0 : std::max(tmp1, tmp0);
                    auto tmp4 = (tmp2 != tmp2) ? tmp2 : std::max(tmp3, tmp2);
                    auto tmp6 = (tmp4 != tmp4) ? tmp4 : std::max(tmp5, tmp4);
                    out_ptr0[i1 + (14*i0)] = tmp6;
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<1536; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<14; i1+=1)
                {
                    #pragma GCC ivdep
                    for(long i2=0; i2<14; i2+=1)
                    {
                        auto tmp0 = in_out_ptr0[(2*i2) + (56*i1) + (784*i0)];
                        auto tmp2 = in_out_ptr0[1 + (2*i2) + (56*i1) + (784*i0)];
                        auto tmp7 = in_out_ptr0[28 + (2*i2) + (56*i1) + (784*i0)];
                        auto tmp12 = in_out_ptr0[29 + (2*i2) + (56*i1) + (784*i0)];
                        auto tmp1 = static_cast<long>((2*i2) + (56*i1));
                        auto tmp3 = static_cast<long>(1 + (2*i2) + (56*i1));
                        auto tmp4 = tmp2 > tmp0;
                        auto tmp5 = tmp4 ? tmp3 : tmp1;
                        auto tmp6 = (tmp0 != tmp0) ? tmp0 : std::max(tmp2, tmp0);
                        auto tmp8 = static_cast<long>(28 + (2*i2) + (56*i1));
                        auto tmp9 = tmp7 > tmp6;
                        auto tmp10 = tmp9 ? tmp8 : tmp5;
                        auto tmp11 = (tmp6 != tmp6) ? tmp6 : std::max(tmp7, tmp6);
                        auto tmp13 = static_cast<long>(29 + (2*i2) + (56*i1));
                        auto tmp14 = tmp12 > tmp11;
                        auto tmp15 = tmp14 ? tmp13 : tmp10;
                        auto tmp16 = (tmp11 != tmp11) ? tmp11 : std::max(tmp12, tmp11);
                        out_ptr1[i2 + (14*i1) + (196*i0)] = tmp15;
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
                       long* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1)
{
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<37632; i0+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + 16*i0);
            }
            #pragma omp for simd simdlen(8) 
            for(long i0=602112; i0<602112; i0+=1)
            {
                auto tmp0 = in_out_ptr0[i0];
                auto tmp1 = tmp0 * (tmp0>0);
                in_out_ptr0[i0] = tmp1;
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<3072; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<7; i1+=1)
                {
                    #pragma GCC ivdep
                    for(long i2=0; i2<7; i2+=1)
                    {
                        auto tmp0 = in_out_ptr0[(2*i2) + (28*i1) + (196*i0)];
                        auto tmp2 = in_out_ptr0[1 + (2*i2) + (28*i1) + (196*i0)];
                        auto tmp7 = in_out_ptr0[14 + (2*i2) + (28*i1) + (196*i0)];
                        auto tmp12 = in_out_ptr0[15 + (2*i2) + (28*i1) + (196*i0)];
                        auto tmp1 = static_cast<long>((2*i2) + (28*i1));
                        auto tmp3 = static_cast<long>(1 + (2*i2) + (28*i1));
                        auto tmp4 = tmp2 > tmp0;
                        auto tmp5 = tmp4 ? tmp3 : tmp1;
                        auto tmp6 = (tmp0 != tmp0) ? tmp0 : std::max(tmp2, tmp0);
                        auto tmp8 = static_cast<long>(14 + (2*i2) + (28*i1));
                        auto tmp9 = tmp7 > tmp6;
                        auto tmp10 = tmp9 ? tmp8 : tmp5;
                        auto tmp11 = (tmp6 != tmp6) ? tmp6 : std::max(tmp7, tmp6);
                        auto tmp13 = static_cast<long>(15 + (2*i2) + (28*i1));
                        auto tmp14 = tmp12 > tmp11;
                        auto tmp15 = tmp14 ? tmp13 : tmp10;
                        auto tmp16 = (tmp11 != tmp11) ? tmp11 : std::max(tmp12, tmp11);
                        out_ptr0[i2 + (7*i1) + (49*i0)] = tmp15;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long i0=0; i0<48; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<3136; i1+=1)
                {
                    auto tmp0 = in_out_ptr0[(2*(i1 % 7)) + (28*(i1 / 7)) + (12544*i0)];
                    auto tmp1 = in_out_ptr0[1 + (2*(i1 % 7)) + (28*(i1 / 7)) + (12544*i0)];
                    auto tmp3 = in_out_ptr0[14 + (2*(i1 % 7)) + (28*(i1 / 7)) + (12544*i0)];
                    auto tmp5 = in_out_ptr0[15 + (2*(i1 % 7)) + (28*(i1 / 7)) + (12544*i0)];
                    auto tmp2 = (tmp0 != tmp0) ? tmp0 : std::max(tmp1, tmp0);
                    auto tmp4 = (tmp2 != tmp2) ? tmp2 : std::max(tmp3, tmp2);
                    auto tmp6 = (tmp4 != tmp4) ? tmp4 : std::max(tmp5, tmp4);
                    out_ptr1[i1 + (3136*i0)] = tmp6;
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9 = args
    args.clear()
    buf0 = aten.convolution(primals_9, primals_1, primals_2, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf0, (48, 32, 28, 28), (25088, 784, 28, 1))
    del primals_2
    buf1 = buf0; del buf0  # reuse
    buf2 = empty_strided((48, 32, 14, 14), (6272, 196, 14, 1), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((48, 32, 14, 14), (6272, 196, 14, 1), device='cpu', dtype=torch.int64)
    kernel_cpp_0(c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()))
    buf4 = aten.convolution(buf2, primals_3, primals_4, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf4, (48, 64, 14, 14), (12544, 196, 14, 1))
    del primals_4
    buf5 = buf4; del buf4  # reuse
    buf6 = empty_strided((48, 64, 7, 7), (3136, 49, 7, 1), device='cpu', dtype=torch.int64)
    buf7 = empty_strided((48, 3136), (3136, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_1(c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    buf8 = empty_strided((48, 512), (512, 1), device='cpu', dtype=torch.float32)
    extern_kernels.addmm(primals_6, buf7, as_strided(primals_5, (3136, 512), (1, 3136)), alpha=1, beta=1, out=buf8)
    del primals_6
    buf9 = empty_strided((48, 10), (10, 1), device='cpu', dtype=torch.float32)
    extern_kernels.addmm(primals_8, buf8, as_strided(primals_7, (512, 10), (1, 512)), alpha=1, beta=1, out=buf9)
    del primals_8
    return (buf9, primals_1, primals_3, primals_9, buf1, buf2, buf3, buf5, buf6, buf7, buf8, as_strided(primals_7, (10, 512), (512, 1)), as_strided(primals_5, (512, 3136), (3136, 1)), )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((512, 3136), (3136, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((10, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((10, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((48, 1, 28, 28), (784, 784, 28, 1), device='cpu', dtype=torch.float32)
    print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9]))
