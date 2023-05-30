
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

import triton
import triton.language as tl
from torch._inductor.triton_ops.autotune import grid
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


triton__0 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 32
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.where(0 != 0, 0, tl.where(0 > tmp2, 0, tmp2))
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp3, xmask)
''')


triton__1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 14
    x1 = (xindex // 14)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (56*x1)), xmask)
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (56*x1)), xmask)
    tmp3 = tl.load(in_ptr0 + (28 + (2*x0) + (56*x1)), xmask)
    tmp5 = tl.load(in_ptr0 + (29 + (2*x0) + (56*x1)), xmask)
    tmp2 = tl.where(tmp1 != tmp1, tmp1, tl.where(tmp1 > tmp0, tmp1, tmp0))
    tmp4 = tl.where(tmp3 != tmp3, tmp3, tl.where(tmp3 > tmp2, tmp3, tmp2))
    tmp6 = tl.where(tmp5 != tmp5, tmp5, tl.where(tmp5 > tmp4, tmp5, tmp4))
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


triton__2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.where(0 != 0, 0, tl.where(0 > tmp2, 0, tmp2))
    tl.store(in_out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp3, xmask)
''')


triton__3 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 7
    x1 = (xindex // 7)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (28*x1)), xmask)
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (28*x1)), xmask)
    tmp3 = tl.load(in_ptr0 + (14 + (2*x0) + (28*x1)), xmask)
    tmp5 = tl.load(in_ptr0 + (15 + (2*x0) + (28*x1)), xmask)
    tmp2 = tl.where(tmp1 != tmp1, tmp1, tl.where(tmp1 > tmp0, tmp1, tmp0))
    tmp4 = tl.where(tmp3 != tmp3, tmp3, tl.where(tmp3 > tmp2, tmp3, tmp2))
    tmp6 = tl.where(tmp5 != tmp5, tmp5, tl.where(tmp5 > tmp4, tmp5, tmp4))
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = aten.convolution(arg8_1, arg0_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf0, (16, 32, 28, 28), (25088, 784, 28, 1))
        del arg0_1
        del arg8_1
        buf1 = buf0; del buf0  # reuse
        stream0 = get_cuda_stream(0)
        triton__0.run(buf1, arg1_1, 401408, grid=grid(401408), stream=stream0)
        del arg1_1
        buf2 = empty_strided((16, 32, 14, 14), (6272, 196, 14, 1), device='cuda', dtype=torch.float32)
        triton__1.run(buf1, buf2, 100352, grid=grid(100352), stream=stream0)
        del buf1
        buf3 = aten.convolution(buf2, arg2_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf3, (16, 64, 14, 14), (12544, 196, 14, 1))
        del arg2_1
        del buf2
        buf4 = buf3; del buf3  # reuse
        triton__2.run(buf4, arg3_1, 200704, grid=grid(200704), stream=stream0)
        del arg3_1
        buf5 = empty_strided((16, 64, 7, 7), (3136, 49, 7, 1), device='cuda', dtype=torch.float32)
        triton__3.run(buf4, buf5, 50176, grid=grid(50176), stream=stream0)
        del buf4
        buf6 = empty_strided((16, 512), (512, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(arg5_1, as_strided(buf5, (16, 3136), (3136, 1)), as_strided(arg4_1, (3136, 512), (1, 3136)), alpha=1, beta=1, out=buf6)
        del arg4_1
        del arg5_1
        del buf5
        buf7 = empty_strided((16, 10), (10, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(arg7_1, buf6, as_strided(arg6_1, (512, 10), (1, 512)), alpha=1, beta=1, out=buf7)
        del arg6_1
        del arg7_1
        return (buf7, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((512, 3136), (3136, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((10, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((16, 1, 28, 28), (784, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1]))
