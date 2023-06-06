
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
                       float* __restrict__ out_ptr0)
{
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<7168; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<14; i1+=1)
                {
                    auto tmp0 = in_ptr0[(2*i1) + (56*i0)];
                    auto tmp2 = in_ptr0[1 + (2*i1) + (56*i0)];
                    auto tmp5 = in_ptr0[28 + (2*i1) + (56*i0)];
                    auto tmp8 = in_ptr0[29 + (2*i1) + (56*i0)];
                    auto tmp1 = tmp0 * (tmp0>0);
                    auto tmp3 = tmp2 * (tmp2>0);
                    auto tmp4 = (tmp1 != tmp1) ? tmp1 : std::max(tmp3, tmp1);
                    auto tmp6 = tmp5 * (tmp5>0);
                    auto tmp7 = (tmp4 != tmp4) ? tmp4 : std::max(tmp6, tmp4);
                    auto tmp9 = tmp8 * (tmp8>0);
                    auto tmp10 = (tmp7 != tmp7) ? tmp7 : std::max(tmp9, tmp7);
                    out_ptr0[i1 + (14*i0)] = tmp10;
                }
            }
        }
    }
}
''')


kernel_cpp_1 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long i0=0; i0<7168; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<7; i1+=1)
            {
                auto tmp0 = in_ptr0[(2*i1) + (28*i0)];
                auto tmp2 = in_ptr0[1 + (2*i1) + (28*i0)];
                auto tmp5 = in_ptr0[14 + (2*i1) + (28*i0)];
                auto tmp8 = in_ptr0[15 + (2*i1) + (28*i0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp3 = tmp2 * (tmp2>0);
                auto tmp4 = (tmp1 != tmp1) ? tmp1 : std::max(tmp3, tmp1);
                auto tmp6 = tmp5 * (tmp5>0);
                auto tmp7 = (tmp4 != tmp4) ? tmp4 : std::max(tmp6, tmp4);
                auto tmp9 = tmp8 * (tmp8>0);
                auto tmp10 = (tmp7 != tmp7) ? tmp7 : std::max(tmp9, tmp7);
                out_ptr0[i1 + (7*i0)] = tmp10;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1 = args
    args.clear()
    buf0 = aten.convolution(arg8_1, arg0_1, arg1_1, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf0, (16, 32, 28, 28), (25088, 784, 28, 1))
    del arg0_1
    del arg1_1
    del arg8_1
    buf1 = empty_strided((16, 32, 14, 14), (6272, 196, 14, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_0(c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del buf0
    buf2 = aten.convolution(buf1, arg2_1, arg3_1, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf2, (16, 64, 14, 14), (12544, 196, 14, 1))
    del arg2_1
    del arg3_1
    del buf1
    buf3 = empty_strided((16, 64, 7, 7), (3136, 49, 7, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_1(c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()))
    del buf2
    buf4 = empty_strided((16, 512), (512, 1), device='cpu', dtype=torch.float32)
    extern_kernels.addmm(arg5_1, as_strided(buf3, (16, 3136), (3136, 1)), as_strided(arg4_1, (3136, 512), (1, 3136)), alpha=1, beta=1, out=buf4)
    del arg4_1
    del arg5_1
    del buf3
    buf5 = empty_strided((16, 10), (10, 1), device='cpu', dtype=torch.float32)
    extern_kernels.addmm(arg7_1, buf4, as_strided(arg6_1, (512, 10), (1, 512)), alpha=1, beta=1, out=buf5)
    del arg6_1
    del arg7_1
    return (buf5, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((512, 3136), (3136, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((10, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((10, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((16, 1, 28, 28), (784, 784, 28, 1), device='cpu', dtype=torch.float32)
    print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1]))
