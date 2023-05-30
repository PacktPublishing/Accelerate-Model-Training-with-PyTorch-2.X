
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
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[16, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10
    rnumel = 48
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (10*r1)), rmask & xmask, eviction_policy='evict_last', other=0)
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 48
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, eviction_policy='evict_last', other=0)
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = xindex % 196
    x1 = (xindex // 14) % 14
    x0 = xindex % 14
    x2 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp3 = x4
    tmp4 = (x1 // 2)
    tmp5 = (x0 // 2)
    tmp6 = 1 + (x1 // 2)
    tmp7 = 1 + (x0 // 2)
    tmp8 = 0
    tmp9 = tl.where(tmp4 != tmp4, tmp4, tl.where(tmp4 > tmp8, tmp4, tmp8))
    tmp10 = tl.where(tmp5 != tmp5, tmp5, tl.where(tmp5 > tmp8, tmp5, tmp8))
    tmp11 = 7
    tmp12 = tl.where(tmp6 != tmp6, tmp6, tl.where(tmp6 < tmp11, tmp6, tmp11))
    tmp13 = tl.where(tmp7 != tmp7, tmp7, tl.where(tmp7 < tmp11, tmp7, tmp11))
    tmp14 = tmp9 + tmp8
    tmp15 = tmp10 + tmp8
    tmp16 = 1
    tmp17 = tmp12 - tmp16
    tmp18 = tl.where(tmp14 != tmp14, tmp14, tl.where(tmp14 < tmp17, tmp14, tmp17))
    tmp19 = tmp13 - tmp16
    tmp20 = tl.where(tmp15 != tmp15, tmp15, tl.where(tmp15 < tmp19, tmp15, tmp19))
    tmp21 = tl.load(in_ptr1 + (tmp20 + (7*tmp18) + (49*x2)), xmask)
    tmp22 = tl.load(in_ptr2 + (tmp20 + (7*tmp18) + (49*x2)), xmask)
    tmp23 = tmp21 == tmp3
    tmp24 = tl.where(tmp23, tmp22, tmp1)
    tmp25 = tl.where(tmp2, tmp1, tmp24)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp25, xmask)
''')


triton__3 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[128, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((196*x0) + (12544*(r2 // 196)) + (301056*x1) + (r2 % 196)), rmask & xmask, eviction_policy='evict_last', other=0)
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp1, xmask)
''')


triton__4 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[64, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_last', other=0)
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


triton__5 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = xindex % 784
    x1 = (xindex // 28) % 28
    x0 = xindex % 28
    x2 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp3 = x4
    tmp4 = (x1 // 2)
    tmp5 = (x0 // 2)
    tmp6 = 1 + (x1 // 2)
    tmp7 = 1 + (x0 // 2)
    tmp8 = 0
    tmp9 = tl.where(tmp4 != tmp4, tmp4, tl.where(tmp4 > tmp8, tmp4, tmp8))
    tmp10 = tl.where(tmp5 != tmp5, tmp5, tl.where(tmp5 > tmp8, tmp5, tmp8))
    tmp11 = 14
    tmp12 = tl.where(tmp6 != tmp6, tmp6, tl.where(tmp6 < tmp11, tmp6, tmp11))
    tmp13 = tl.where(tmp7 != tmp7, tmp7, tl.where(tmp7 < tmp11, tmp7, tmp11))
    tmp14 = tmp9 + tmp8
    tmp15 = tmp10 + tmp8
    tmp16 = 1
    tmp17 = tmp12 - tmp16
    tmp18 = tl.where(tmp14 != tmp14, tmp14, tl.where(tmp14 < tmp17, tmp14, tmp17))
    tmp19 = tmp13 - tmp16
    tmp20 = tl.where(tmp15 != tmp15, tmp15, tl.where(tmp15 < tmp19, tmp15, tmp19))
    tmp21 = tl.load(in_ptr1 + (tmp20 + (14*tmp18) + (196*x2)), xmask)
    tmp22 = tl.load(in_ptr2 + (tmp20 + (14*tmp18) + (196*x2)), xmask)
    tmp23 = tmp21 == tmp3
    tmp24 = tl.where(tmp23, tmp22, tmp1)
    tmp25 = tl.where(tmp2, tmp1, tmp24)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp25, xmask)
''')


triton__6 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 7527
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 32)
    x0 = xindex % 32
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7527*x1)
        tmp1 = 37632
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((784*x0) + (25088*(((r2 + (7527*x1)) // 784) % 48)) + ((r2 + (7527*x1)) % 784) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp4 = tl.where(tmp2, tmp3, 0)
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp5, xmask)
''')


triton__7 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[32, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 5
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, eviction_policy='evict_last', other=0)
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_9, relu, getitem, getitem_1, relu_1, getitem_3, view, addmm, permute_2, permute_6, tangents_1 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((48, 512), (512, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(tangents_1, permute_2, out=buf0)
        del permute_2
        buf1 = empty_strided((10, 512), (512, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(tangents_1, (10, 48), (1, 10)), addmm, out=buf1)
        del addmm
        buf2 = empty_strided((1, 10), (10, 1), device='cuda', dtype=torch.float32)
        stream0 = get_cuda_stream(0)
        triton__0.run(tangents_1, buf2, 10, 48, grid=grid(10), stream=stream0)
        del tangents_1
        buf3 = empty_strided((48, 3136), (3136, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(buf0, permute_6, out=buf3)
        del permute_6
        buf4 = empty_strided((512, 3136), (3136, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(buf0, (512, 48), (1, 512)), view, out=buf4)
        del view
        buf5 = empty_strided((1, 512), (512, 1), device='cuda', dtype=torch.float32)
        triton__1.run(buf0, buf5, 512, 48, grid=grid(512), stream=stream0)
        del buf0
        buf6 = empty_strided((48, 64, 14, 14), (12544, 196, 14, 1), device='cuda', dtype=torch.float32)
        triton__2.run(relu_1, getitem_3, buf3, buf6, 602112, grid=grid(602112), stream=stream0)
        del buf3
        del getitem_3
        del relu_1
        buf7 = empty_strided((64, 2), (1, 64), device='cuda', dtype=torch.float32)
        triton__3.run(buf6, buf7, 128, 4704, grid=grid(128), stream=stream0)
        buf8 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        triton__4.run(buf7, buf8, 64, 2, grid=grid(64), stream=stream0)
        del buf7
        buf9 = aten.convolution_backward(buf6, getitem, primals_3, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf6
        del getitem
        del primals_3
        buf10 = buf9[0]
        assert_size_stride(buf10, (48, 32, 14, 14), (6272, 196, 14, 1))
        buf11 = buf9[1]
        assert_size_stride(buf11, (64, 32, 3, 3), (288, 9, 3, 1))
        del buf9
        buf12 = empty_strided((48, 32, 28, 28), (25088, 784, 28, 1), device='cuda', dtype=torch.float32)
        triton__5.run(relu, getitem_1, buf10, buf12, 1204224, grid=grid(1204224), stream=stream0)
        del buf10
        del getitem_1
        del relu
        buf13 = empty_strided((32, 5), (1, 32), device='cuda', dtype=torch.float32)
        triton__6.run(buf12, buf13, 160, 7527, grid=grid(160), stream=stream0)
        buf14 = empty_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
        triton__7.run(buf13, buf14, 32, 5, grid=grid(32), stream=stream0)
        del buf13
        buf15 = aten.convolution_backward(buf12, primals_9, primals_1, [32], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf12
        del primals_1
        del primals_9
        buf16 = buf15[1]
        assert_size_stride(buf16, (32, 1, 3, 3), (9, 9, 3, 1))
        del buf15
        return (buf16, buf14, buf11, buf8, as_strided(buf4, (512, 3136), (3136, 1)), as_strided(buf5, (512, ), (1, )), as_strided(buf1, (10, 512), (512, 1)), as_strided(buf2, (10, ), (1, )), None, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((48, 1, 28, 28), (784, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((48, 32, 28, 28), (25088, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    getitem = rand_strided((48, 32, 14, 14), (6272, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((48, 32, 14, 14), (6272, 196, 14, 1), device='cuda:0', dtype=torch.int64)
    relu_1 = rand_strided((48, 64, 14, 14), (12544, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((48, 64, 7, 7), (3136, 49, 7, 1), device='cuda:0', dtype=torch.int64)
    view = rand_strided((48, 3136), (3136, 1), device='cuda:0', dtype=torch.float32)
    addmm = rand_strided((48, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_2 = rand_strided((10, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_6 = rand_strided((512, 3136), (3136, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((48, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    print_performance(lambda: call([primals_1, primals_3, primals_9, relu, getitem, getitem_1, relu_1, getitem_3, view, addmm, permute_2, permute_6, tangents_1]))
