
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
    {
        #pragma GCC ivdep
        for(long i0=0; i0<10; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<64; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (10*i1)];
                    tmp1 += tmp0;
                }
                out_ptr0[i0] = tmp1;
            }
        }
    }
}
''')


kernel_cpp_1 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const long* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long i0=0; i0<512; i0+=1)
        {
            {
                float tmp1 = 0;
                for(long i1=0; i1<64; i1+=1)
                {
                    auto tmp0 = in_ptr0[i0 + (512*i1)];
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
            for(long i0=0; i0<4096; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<14; i1+=1)
                {
                    #pragma GCC ivdep
                    for(long i2=0; i2<14; i2+=1)
                    {
                        auto tmp0 = in_ptr1[i2 + (14*i1) + (196*i0)];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = static_cast<int>(i2 + (14*i1));
                        auto tmp5 = static_cast<int>((i1 / 2));
                        auto tmp6 = static_cast<int>((i2 / 2));
                        auto tmp7 = static_cast<int>(1 + (i1 / 2));
                        auto tmp8 = static_cast<int>(1 + (i2 / 2));
                        auto tmp9 = static_cast<int>(0);
                        auto tmp10 = (tmp9 != tmp9) ? tmp9 : std::max(tmp5, tmp9);
                        auto tmp11 = (tmp9 != tmp9) ? tmp9 : std::max(tmp6, tmp9);
                        auto tmp12 = static_cast<int>(7);
                        auto tmp13 = (tmp12 != tmp12) ? tmp12 : std::min(tmp7, tmp12);
                        auto tmp14 = (tmp12 != tmp12) ? tmp12 : std::min(tmp8, tmp12);
                        auto tmp15 = tmp10 + tmp9;
                        auto tmp16 = tmp11 + tmp9;
                        auto tmp17 = static_cast<int>(1);
                        auto tmp18 = tmp13 - tmp17;
                        auto tmp19 = (tmp18 != tmp18) ? tmp18 : std::min(tmp15, tmp18);
                        auto tmp20 = tmp14 - tmp17;
                        auto tmp21 = (tmp20 != tmp20) ? tmp20 : std::min(tmp16, tmp20);
                        auto tmp22 = in_ptr2[tmp21 + (7*tmp19) + (49*i0)];
                        auto tmp23 = in_ptr3[tmp21 + (7*tmp19) + (49*i0)];
                        auto tmp24 = tmp22 == tmp4;
                        auto tmp25 = tmp24 ? tmp23 : tmp3;
                        auto tmp26 = tmp2 ? tmp3 : tmp25;
                        out_ptr1[i2 + (14*i1) + (196*i0)] = tmp26;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_2 = async_compile.cpp('''
#include "/tmp/torchinductor_xmmw/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const long* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr0)
{
    #pragma omp parallel num_threads(16)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<2048; i0+=1)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<28; i1+=1)
                {
                    #pragma GCC ivdep
                    for(long i2=0; i2<28; i2+=1)
                    {
                        auto tmp0 = in_ptr0[i2 + (28*i1) + (784*i0)];
                        auto tmp1 = static_cast<float>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = static_cast<int>(i2 + (28*i1));
                        auto tmp5 = static_cast<int>((i1 / 2));
                        auto tmp6 = static_cast<int>((i2 / 2));
                        auto tmp7 = static_cast<int>(1 + (i1 / 2));
                        auto tmp8 = static_cast<int>(1 + (i2 / 2));
                        auto tmp9 = static_cast<int>(0);
                        auto tmp10 = (tmp9 != tmp9) ? tmp9 : std::max(tmp5, tmp9);
                        auto tmp11 = (tmp9 != tmp9) ? tmp9 : std::max(tmp6, tmp9);
                        auto tmp12 = static_cast<int>(14);
                        auto tmp13 = (tmp12 != tmp12) ? tmp12 : std::min(tmp7, tmp12);
                        auto tmp14 = (tmp12 != tmp12) ? tmp12 : std::min(tmp8, tmp12);
                        auto tmp15 = tmp10 + tmp9;
                        auto tmp16 = tmp11 + tmp9;
                        auto tmp17 = static_cast<int>(1);
                        auto tmp18 = tmp13 - tmp17;
                        auto tmp19 = (tmp18 != tmp18) ? tmp18 : std::min(tmp15, tmp18);
                        auto tmp20 = tmp14 - tmp17;
                        auto tmp21 = (tmp20 != tmp20) ? tmp20 : std::min(tmp16, tmp20);
                        auto tmp22 = in_ptr1[tmp21 + (14*tmp19) + (196*i0)];
                        auto tmp23 = in_ptr2[tmp21 + (14*tmp19) + (196*i0)];
                        auto tmp24 = tmp22 == tmp4;
                        auto tmp25 = tmp24 ? tmp23 : tmp3;
                        auto tmp26 = tmp2 ? tmp3 : tmp25;
                        out_ptr0[i2 + (28*i1) + (784*i0)] = tmp26;
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
    primals_1, primals_3, primals_9, relu, getitem, getitem_1, relu_1, getitem_3, view, addmm, permute_2, permute_6, tangents_1 = args
    args.clear()
    buf0 = empty_strided((64, 512), (512, 1), device='cpu', dtype=torch.float32)
    extern_kernels.mm(tangents_1, permute_2, out=buf0)
    del permute_2
    buf1 = empty_strided((10, 512), (512, 1), device='cpu', dtype=torch.float32)
    extern_kernels.mm(as_strided(tangents_1, (10, 64), (1, 10)), addmm, out=buf1)
    del addmm
    buf2 = empty_strided((1, 10), (10, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_0(c_void_p(tangents_1.data_ptr()), c_void_p(buf2.data_ptr()))
    del tangents_1
    buf3 = empty_strided((64, 3136), (3136, 1), device='cpu', dtype=torch.float32)
    extern_kernels.mm(buf0, permute_6, out=buf3)
    del permute_6
    buf4 = empty_strided((512, 3136), (3136, 1), device='cpu', dtype=torch.float32)
    extern_kernels.mm(as_strided(buf0, (512, 64), (1, 512)), view, out=buf4)
    del view
    buf5 = empty_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((64, 64, 14, 14), (12544, 196, 14, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_1(c_void_p(buf0.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(getitem_3.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()))
    del buf0
    del buf3
    del getitem_3
    del relu_1
    buf7 = aten.convolution_backward(buf6, getitem, primals_3, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf6
    del getitem
    del primals_3
    buf8 = buf7[0]
    assert_size_stride(buf8, (64, 32, 14, 14), (6272, 196, 14, 1))
    buf9 = buf7[1]
    assert_size_stride(buf9, (64, 32, 3, 3), (288, 9, 3, 1))
    buf10 = buf7[2]
    assert_size_stride(buf10, (64, ), (1, ))
    del buf7
    buf11 = empty_strided((64, 32, 28, 28), (25088, 784, 28, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_2(c_void_p(relu.data_ptr()), c_void_p(getitem_1.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf11.data_ptr()))
    del buf8
    del getitem_1
    del relu
    buf12 = aten.convolution_backward(buf11, primals_9, primals_1, [32], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, True])
    del buf11
    del primals_1
    del primals_9
    buf13 = buf12[1]
    assert_size_stride(buf13, (32, 1, 3, 3), (9, 9, 3, 1))
    buf14 = buf12[2]
    assert_size_stride(buf14, (32, ), (1, ))
    del buf12
    return (buf13, buf14, buf9, buf10, as_strided(buf4, (512, 3136), (3136, 1)), as_strided(buf5, (512, ), (1, )), as_strided(buf1, (10, 512), (512, 1)), as_strided(buf2, (10, ), (1, )), None, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((64, 1, 28, 28), (784, 784, 28, 1), device='cpu', dtype=torch.float32)
    relu = rand_strided((64, 32, 28, 28), (25088, 784, 28, 1), device='cpu', dtype=torch.float32)
    getitem = rand_strided((64, 32, 14, 14), (6272, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_1 = rand_strided((64, 32, 14, 14), (6272, 196, 14, 1), device='cpu', dtype=torch.int64)
    relu_1 = rand_strided((64, 64, 14, 14), (12544, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((64, 64, 7, 7), (3136, 49, 7, 1), device='cpu', dtype=torch.int64)
    view = rand_strided((64, 3136), (3136, 1), device='cpu', dtype=torch.float32)
    addmm = rand_strided((64, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_2 = rand_strided((10, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_6 = rand_strided((512, 3136), (3136, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((64, 10), (10, 1), device='cpu', dtype=torch.float32)
    print_performance(lambda: call([primals_1, primals_3, primals_9, relu, getitem, getitem_1, relu_1, getitem_3, view, addmm, permute_2, permute_6, tangents_1]))
