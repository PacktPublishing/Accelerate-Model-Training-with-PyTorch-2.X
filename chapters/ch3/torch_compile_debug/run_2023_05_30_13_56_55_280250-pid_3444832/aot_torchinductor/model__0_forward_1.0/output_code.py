
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

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
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

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 14
    x1 = (xindex // 14)
    x2 = xindex
    x3 = (xindex // 14) % 14
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (56*x1)), xmask)
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (56*x1)), xmask)
    tmp3 = tl.load(in_ptr0 + (28 + (2*x0) + (56*x1)), xmask)
    tmp5 = tl.load(in_ptr0 + (29 + (2*x0) + (56*x1)), xmask)
    tmp2 = tl.where(tmp1 != tmp1, tmp1, tl.where(tmp1 > tmp0, tmp1, tmp0))
    tmp4 = tl.where(tmp3 != tmp3, tmp3, tl.where(tmp3 > tmp2, tmp3, tmp2))
    tmp6 = tl.where(tmp5 != tmp5, tmp5, tl.where(tmp5 > tmp4, tmp5, tmp4))
    tmp7 = (2*x0) + (56*x3)
    tmp8 = 1 + (2*x0) + (56*x3)
    tmp9 = tmp1 > tmp0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tmp11 = 28 + (2*x0) + (56*x3)
    tmp12 = tmp3 > tmp2
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = 29 + (2*x0) + (56*x3)
    tmp15 = tmp5 > tmp4
    tmp16 = tl.where(tmp15, tmp14, tmp13)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
    tl.store(out_ptr1 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


triton__2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
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

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 7
    x3 = (xindex // 7)
    x1 = (xindex // 7) % 7
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (28*x3)), xmask)
    tmp2 = tl.load(in_ptr0 + (1 + (2*x0) + (28*x3)), xmask)
    tmp7 = tl.load(in_ptr0 + (14 + (2*x0) + (28*x3)), xmask)
    tmp12 = tl.load(in_ptr0 + (15 + (2*x0) + (28*x3)), xmask)
    tmp1 = (2*x0) + (28*x1)
    tmp3 = 1 + (2*x0) + (28*x1)
    tmp4 = tmp2 > tmp0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tmp6 = tl.where(tmp2 != tmp2, tmp2, tl.where(tmp2 > tmp0, tmp2, tmp0))
    tmp8 = 14 + (2*x0) + (28*x1)
    tmp9 = tmp7 > tmp6
    tmp10 = tl.where(tmp9, tmp8, tmp5)
    tmp11 = tl.where(tmp7 != tmp7, tmp7, tl.where(tmp7 > tmp6, tmp7, tmp6))
    tmp13 = 15 + (2*x0) + (28*x1)
    tmp14 = tmp12 > tmp11
    tmp15 = tl.where(tmp14, tmp13, tmp10)
    tmp16 = tl.where(tmp12 != tmp12, tmp12, tl.where(tmp12 > tmp11, tmp12, tmp11))
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
''')


triton__4 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*(x0 % 7)) + (28*(x0 // 7)) + (12544*x1)), xmask)
    tmp1 = tl.load(in_ptr0 + (1 + (2*(x0 % 7)) + (28*(x0 // 7)) + (12544*x1)), xmask)
    tmp3 = tl.load(in_ptr0 + (14 + (2*(x0 % 7)) + (28*(x0 // 7)) + (12544*x1)), xmask)
    tmp5 = tl.load(in_ptr0 + (15 + (2*(x0 % 7)) + (28*(x0 // 7)) + (12544*x1)), xmask)
    tmp2 = tl.where(tmp1 != tmp1, tmp1, tl.where(tmp1 > tmp0, tmp1, tmp0))
    tmp4 = tl.where(tmp3 != tmp3, tmp3, tl.where(tmp3 > tmp2, tmp3, tmp2))
    tmp6 = tl.where(tmp5 != tmp5, tmp5, tl.where(tmp5 > tmp4, tmp5, tmp4))
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = aten.convolution(primals_9, primals_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf0, (64, 32, 28, 28), (25088, 784, 28, 1))
        buf1 = buf0; del buf0  # reuse
        stream0 = get_cuda_stream(0)
        triton__0.run(buf1, primals_2, 1605632, grid=grid(1605632), stream=stream0)
        del primals_2
        buf2 = empty_strided((64, 32, 14, 14), (6272, 196, 14, 1), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((64, 32, 14, 14), (6272, 196, 14, 1), device='cuda', dtype=torch.int64)
        triton__1.run(buf1, buf2, buf3, 401408, grid=grid(401408), stream=stream0)
        buf4 = aten.convolution(buf2, primals_3, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
        assert_size_stride(buf4, (64, 64, 14, 14), (12544, 196, 14, 1))
        buf5 = buf4; del buf4  # reuse
        triton__2.run(buf5, primals_4, 802816, grid=grid(802816), stream=stream0)
        del primals_4
        buf6 = empty_strided((64, 64, 7, 7), (3136, 49, 7, 1), device='cuda', dtype=torch.int64)
        triton__3.run(buf5, buf6, 200704, grid=grid(200704), stream=stream0)
        buf7 = empty_strided((64, 3136), (3136, 1), device='cuda', dtype=torch.float32)
        triton__4.run(buf5, buf7, 200704, grid=grid(200704), stream=stream0)
        buf8 = empty_strided((64, 512), (512, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_6, buf7, as_strided(primals_5, (3136, 512), (1, 3136)), alpha=1, beta=1, out=buf8)
        del primals_6
        buf9 = empty_strided((64, 10), (10, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_8, buf8, as_strided(primals_7, (512, 10), (1, 512)), alpha=1, beta=1, out=buf9)
        del primals_8
        return (buf9, primals_1, primals_3, primals_9, buf1, buf2, buf3, buf5, buf6, buf7, buf8, as_strided(primals_7, (10, 512), (512, 1)), as_strided(primals_5, (512, 3136), (3136, 1)), )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((512, 3136), (3136, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((10, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, 1, 28, 28), (784, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9]))
