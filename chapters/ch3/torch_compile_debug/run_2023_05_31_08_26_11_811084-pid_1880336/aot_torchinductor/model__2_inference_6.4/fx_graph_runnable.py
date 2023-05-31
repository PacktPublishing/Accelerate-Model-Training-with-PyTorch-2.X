import torch._inductor.overrides

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
torch._dynamo.config.load_config(b'\x80\x02}q\x00(X\x0b\x00\x00\x00output_codeq\x01\x88X\r\x00\x00\x00log_file_nameq\x02NX\x07\x00\x00\x00verboseq\x03\x88X\x11\x00\x00\x00output_graph_codeq\x04\x89X\x12\x00\x00\x00verify_correctnessq\x05\x89X\x12\x00\x00\x00minimum_call_countq\x06K\x01X\x15\x00\x00\x00dead_code_eliminationq\x07\x88X\x10\x00\x00\x00cache_size_limitq\x08K@X\x14\x00\x00\x00specialize_int_floatq\t\x88X\x0e\x00\x00\x00dynamic_shapesq\n\x89X\x10\x00\x00\x00guard_nn_modulesq\x0b\x89X\x1b\x00\x00\x00traceable_tensor_subclassesq\x0cc__builtin__\nset\nq\r]q\x0e\x85q\x0fRq\x10X\x0f\x00\x00\x00suppress_errorsq\x11\x89X\x15\x00\x00\x00replay_record_enabledq\x12\x89X \x00\x00\x00rewrite_assert_with_torch_assertq\x13\x88X\x12\x00\x00\x00print_graph_breaksq\x14\x89X\x07\x00\x00\x00disableq\x15\x89X*\x00\x00\x00allowed_functions_module_string_ignorelistq\x16h\r]q\x17(X\r\x00\x00\x00torch.testingq\x18X\x13\x00\x00\x00torch.distributionsq\x19X\r\x00\x00\x00torch._decompq\x1aX\x0b\x00\x00\x00torch._refsq\x1bX\x0c\x00\x00\x00torch._primsq\x1ce\x85q\x1dRq\x1eX\x12\x00\x00\x00repro_forward_onlyq\x1f\x89X\x0f\x00\x00\x00repro_toleranceq G?PbM\xd2\xf1\xa9\xfcX\x16\x00\x00\x00capture_scalar_outputsq!\x89X\x19\x00\x00\x00enforce_cond_guards_matchq"\x88X\x0c\x00\x00\x00optimize_ddpq#\x88X\x1a\x00\x00\x00raise_on_ctx_manager_usageq$\x88X\x1c\x00\x00\x00raise_on_unsafe_aot_autogradq%\x89X\x17\x00\x00\x00raise_on_backend_changeq&\x89X\x18\x00\x00\x00error_on_nested_fx_traceq\'\x88X\t\x00\x00\x00allow_rnnq(\x89X\x08\x00\x00\x00base_dirq)X\'\x00\x00\x00/opt/conda/lib/python3.10/site-packagesq*X\x0e\x00\x00\x00debug_dir_rootq+XX\x00\x00\x00/u/xmmw/book/Accelerate-Model-Training-with-PyTorch-2.0/chapters/ch3/torch_compile_debugq,X)\x00\x00\x00DO_NOT_USE_legacy_non_fake_example_inputsq-\x89X\x13\x00\x00\x00_save_config_ignoreq.h\r]q/(X!\x00\x00\x00skipfiles_inline_module_allowlistq0X\x0b\x00\x00\x00repro_afterq1X\x0b\x00\x00\x00repro_levelq2X\x12\x00\x00\x00constant_functionsq3e\x85q4Rq5u.')
torch._inductor.config.load_config(b'\x80\x02}q\x00(X\x05\x00\x00\x00debugq\x01\x89X\x10\x00\x00\x00disable_progressq\x02\x88X\x10\x00\x00\x00verbose_progressq\x03\x89X\x0b\x00\x00\x00cpp_wrapperq\x04\x89X\x03\x00\x00\x00dceq\x05\x89X\x14\x00\x00\x00static_weight_shapesq\x06\x88X\x0c\x00\x00\x00size_assertsq\x07\x88X\x10\x00\x00\x00pick_loop_ordersq\x08\x88X\x0f\x00\x00\x00inplace_buffersq\t\x88X\x11\x00\x00\x00benchmark_harnessq\n\x88X\x0f\x00\x00\x00epilogue_fusionq\x0b\x89X\x15\x00\x00\x00epilogue_fusion_firstq\x0c\x89X\x0f\x00\x00\x00pattern_matcherq\r\x88X\n\x00\x00\x00reorderingq\x0e\x89X\x0c\x00\x00\x00max_autotuneq\x0f\x89X\x17\x00\x00\x00realize_reads_thresholdq\x10K\x04X\x17\x00\x00\x00realize_bytes_thresholdq\x11M\xd0\x07X\x1b\x00\x00\x00realize_acc_reads_thresholdq\x12K\x08X\x0f\x00\x00\x00fallback_randomq\x13\x89X\x12\x00\x00\x00implicit_fallbacksq\x14\x88X\x0b\x00\x00\x00tune_layoutq\x15\x89X\x11\x00\x00\x00aggressive_fusionq\x16\x89X\x0f\x00\x00\x00max_fusion_sizeq\x17K@X\x1b\x00\x00\x00unroll_reductions_thresholdq\x18K\x08X\x0e\x00\x00\x00comment_originq\x19\x89X\x12\x00\x00\x00developer_warningsq\x1a\x88X\x0f\x00\x00\x00compile_threadsq\x1bK X\x13\x00\x00\x00kernel_name_max_opsq\x1cK\nX\r\x00\x00\x00shape_paddingq\x1d\x89X\x0e\x00\x00\x00permute_fusionq\x1e\x89X\x1a\x00\x00\x00profiler_mark_wrapper_callq\x1f\x89X\x18\x00\x00\x00_raise_error_for_testingq \x89X\x0b\x00\x00\x00cpp.threadsq!J\xff\xff\xff\xffX\x13\x00\x00\x00cpp.dynamic_threadsq"\x89X\x0b\x00\x00\x00cpp.simdlenq#NX\x12\x00\x00\x00cpp.min_chunk_sizeq$M\x00\x10X\x07\x00\x00\x00cpp.cxxq%NX\x03\x00\x00\x00g++q&\x86q\'X\x19\x00\x00\x00cpp.enable_kernel_profileq(\x89X\x12\x00\x00\x00cpp.weight_prepackq)\x88X\x11\x00\x00\x00triton.cudagraphsq*\x89X\x17\x00\x00\x00triton.debug_sync_graphq+\x89X\x18\x00\x00\x00triton.debug_sync_kernelq,\x89X\x15\x00\x00\x00triton.dense_indexingq-\x89X\x10\x00\x00\x00triton.max_tilesq.K\x02X\x19\x00\x00\x00triton.autotune_pointwiseq/\x88X\'\x00\x00\x00triton.tiling_prevents_pointwise_fusionq0\x88X\'\x00\x00\x00triton.tiling_prevents_reduction_fusionq1\x88X\x1b\x00\x00\x00triton.ordered_kernel_namesq2\x89X\x1f\x00\x00\x00triton.descriptive_kernel_namesq3\x89X\x1c\x00\x00\x00triton.persistent_reductionsq4\x89X\r\x00\x00\x00trace.enabledq5\x88X\x0f\x00\x00\x00trace.debug_logq6\x88X\x0e\x00\x00\x00trace.info_logq7\x89X\x0e\x00\x00\x00trace.fx_graphq8\x88X\x1a\x00\x00\x00trace.fx_graph_transformedq9\x88X\x13\x00\x00\x00trace.ir_pre_fusionq:\x88X\x14\x00\x00\x00trace.ir_post_fusionq;\x88X\x11\x00\x00\x00trace.output_codeq<\x88X\x13\x00\x00\x00trace.graph_diagramq=\x89X\x15\x00\x00\x00trace.compile_profileq>\x89X\x10\x00\x00\x00trace.upload_tarq?Nu.')
torch._functorch.config.load_config(b'\x80\x02}q\x00(X\x11\x00\x00\x00use_functionalizeq\x01\x88X\x0f\x00\x00\x00use_fake_tensorq\x02\x88X\x16\x00\x00\x00fake_tensor_allow_metaq\x03\x88X\x0c\x00\x00\x00debug_assertq\x04\x88X\x14\x00\x00\x00debug_fake_cross_refq\x05\x89X\x12\x00\x00\x00use_dynamic_shapesq\x06\x89X\x14\x00\x00\x00static_weight_shapesq\x07\x88X\x03\x00\x00\x00cseq\x08\x88X\x10\x00\x00\x00max_dist_from_bwq\tK\x03X\x0b\x00\x00\x00debug_jointq\n\x88X\x0c\x00\x00\x00debug_graphsq\x0b\x88X\x11\x00\x00\x00debug_partitionerq\x0c\x88X\t\x00\x00\x00log_levelq\rK\nu.')


# REPLACEABLE COMMENT FOR TESTING PURPOSES


# torch version: 2.0.1+cpu
# torch cuda version: None
# torch git version: e9ebda29d87ce0916ab08c06ab26fd3766a870e5


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1):
        convolution = torch.ops.aten.convolution.default(arg320_1, arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  arg320_1 = arg0_1 = None
        add = torch.ops.aten.add.Tensor(arg163_1, 1)
        empty = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_1 = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub = torch.ops.aten.sub.Tensor(convolution, getitem_1);  convolution = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_1 = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(squeeze, 0.1);  squeeze = None
        mul_2 = torch.ops.aten.mul.Tensor(arg161_1, 0.9)
        add_2 = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2 = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3 = torch.ops.aten.mul.Tensor(squeeze_2, 1.0002442002442002);  squeeze_2 = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5 = torch.ops.aten.mul.Tensor(arg162_1, 0.9)
        add_3 = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(arg1_1, -1);  arg1_1 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_6 = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        add_4 = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        relu = torch.ops.aten.relu.default(add_4);  add_4 = None
        max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu, [3, 3], [2, 2], [1, 1]);  relu = None
        getitem_2 = max_pool2d_with_indices[0]
        getitem_3 = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
        convolution_1 = torch.ops.aten.convolution.default(getitem_2, arg3_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg3_1 = None
        add_5 = torch.ops.aten.add.Tensor(arg166_1, 1)
        empty_1 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
        getitem_4 = var_mean_1[0]
        getitem_5 = var_mean_1[1];  var_mean_1 = None
        add_6 = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1 = torch.ops.aten.sub.Tensor(convolution_1, getitem_5);  convolution_1 = None
        mul_7 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3 = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_4 = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_8 = torch.ops.aten.mul.Tensor(squeeze_3, 0.1);  squeeze_3 = None
        mul_9 = torch.ops.aten.mul.Tensor(arg164_1, 0.9)
        add_7 = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5 = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_10 = torch.ops.aten.mul.Tensor(squeeze_5, 1.0009775171065494);  squeeze_5 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12 = torch.ops.aten.mul.Tensor(arg165_1, 0.9)
        add_8 = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        add_9 = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        relu_1 = torch.ops.aten.relu.default(add_9);  add_9 = None
        convolution_2 = torch.ops.aten.convolution.default(relu_1, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_1 = arg6_1 = None
        add_10 = torch.ops.aten.add.Tensor(arg169_1, 1)
        empty_2 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
        getitem_6 = var_mean_2[0]
        getitem_7 = var_mean_2[1];  var_mean_2 = None
        add_11 = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_2 = torch.ops.aten.sub.Tensor(convolution_2, getitem_7);  convolution_2 = None
        mul_14 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6 = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_7 = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_15 = torch.ops.aten.mul.Tensor(squeeze_6, 0.1);  squeeze_6 = None
        mul_16 = torch.ops.aten.mul.Tensor(arg167_1, 0.9)
        add_12 = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        squeeze_8 = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_17 = torch.ops.aten.mul.Tensor(squeeze_8, 1.0009775171065494);  squeeze_8 = None
        mul_18 = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19 = torch.ops.aten.mul.Tensor(arg168_1, 0.9)
        add_13 = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        add_14 = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
        relu_2 = torch.ops.aten.relu.default(add_14);  add_14 = None
        convolution_3 = torch.ops.aten.convolution.default(relu_2, arg9_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_2 = arg9_1 = None
        add_15 = torch.ops.aten.add.Tensor(arg172_1, 1)
        empty_3 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
        getitem_8 = var_mean_3[0]
        getitem_9 = var_mean_3[1];  var_mean_3 = None
        add_16 = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_3 = torch.ops.aten.sub.Tensor(convolution_3, getitem_9);  convolution_3 = None
        mul_21 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9 = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_10 = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_22 = torch.ops.aten.mul.Tensor(squeeze_9, 0.1);  squeeze_9 = None
        mul_23 = torch.ops.aten.mul.Tensor(arg170_1, 0.9)
        add_17 = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        squeeze_11 = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_24 = torch.ops.aten.mul.Tensor(squeeze_11, 1.0009775171065494);  squeeze_11 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
        mul_26 = torch.ops.aten.mul.Tensor(arg171_1, 0.9)
        add_18 = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        add_19 = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
        convolution_4 = torch.ops.aten.convolution.default(getitem_2, arg12_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_2 = arg12_1 = None
        add_20 = torch.ops.aten.add.Tensor(arg175_1, 1)
        empty_4 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
        getitem_10 = var_mean_4[0]
        getitem_11 = var_mean_4[1];  var_mean_4 = None
        add_21 = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_4 = torch.ops.aten.sub.Tensor(convolution_4, getitem_11);  convolution_4 = None
        mul_28 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12 = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_13 = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_29 = torch.ops.aten.mul.Tensor(squeeze_12, 0.1);  squeeze_12 = None
        mul_30 = torch.ops.aten.mul.Tensor(arg173_1, 0.9)
        add_22 = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
        squeeze_14 = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_31 = torch.ops.aten.mul.Tensor(squeeze_14, 1.0009775171065494);  squeeze_14 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
        mul_33 = torch.ops.aten.mul.Tensor(arg174_1, 0.9)
        add_23 = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        add_24 = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
        add_25 = torch.ops.aten.add.Tensor(add_19, add_24);  add_19 = add_24 = None
        relu_3 = torch.ops.aten.relu.default(add_25);  add_25 = None
        convolution_5 = torch.ops.aten.convolution.default(relu_3, arg15_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg15_1 = None
        add_26 = torch.ops.aten.add.Tensor(arg178_1, 1)
        empty_5 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
        getitem_12 = var_mean_5[0]
        getitem_13 = var_mean_5[1];  var_mean_5 = None
        add_27 = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        sub_5 = torch.ops.aten.sub.Tensor(convolution_5, getitem_13);  convolution_5 = None
        mul_35 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15 = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_16 = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_36 = torch.ops.aten.mul.Tensor(squeeze_15, 0.1);  squeeze_15 = None
        mul_37 = torch.ops.aten.mul.Tensor(arg176_1, 0.9)
        add_28 = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        squeeze_17 = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_38 = torch.ops.aten.mul.Tensor(squeeze_17, 1.0009775171065494);  squeeze_17 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
        mul_40 = torch.ops.aten.mul.Tensor(arg177_1, 0.9)
        add_29 = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        add_30 = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
        relu_4 = torch.ops.aten.relu.default(add_30);  add_30 = None
        convolution_6 = torch.ops.aten.convolution.default(relu_4, arg18_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_4 = arg18_1 = None
        add_31 = torch.ops.aten.add.Tensor(arg181_1, 1)
        empty_6 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
        getitem_14 = var_mean_6[0]
        getitem_15 = var_mean_6[1];  var_mean_6 = None
        add_32 = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_6 = torch.ops.aten.sub.Tensor(convolution_6, getitem_15);  convolution_6 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_18 = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_19 = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_43 = torch.ops.aten.mul.Tensor(squeeze_18, 0.1);  squeeze_18 = None
        mul_44 = torch.ops.aten.mul.Tensor(arg179_1, 0.9)
        add_33 = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
        squeeze_20 = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_45 = torch.ops.aten.mul.Tensor(squeeze_20, 1.0009775171065494);  squeeze_20 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
        mul_47 = torch.ops.aten.mul.Tensor(arg180_1, 0.9)
        add_34 = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
        add_35 = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
        relu_5 = torch.ops.aten.relu.default(add_35);  add_35 = None
        convolution_7 = torch.ops.aten.convolution.default(relu_5, arg21_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_5 = arg21_1 = None
        add_36 = torch.ops.aten.add.Tensor(arg184_1, 1)
        empty_7 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
        getitem_16 = var_mean_7[0]
        getitem_17 = var_mean_7[1];  var_mean_7 = None
        add_37 = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_7 = torch.ops.aten.sub.Tensor(convolution_7, getitem_17);  convolution_7 = None
        mul_49 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_21 = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_22 = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_50 = torch.ops.aten.mul.Tensor(squeeze_21, 0.1);  squeeze_21 = None
        mul_51 = torch.ops.aten.mul.Tensor(arg182_1, 0.9)
        add_38 = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
        squeeze_23 = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_52 = torch.ops.aten.mul.Tensor(squeeze_23, 1.0009775171065494);  squeeze_23 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
        mul_54 = torch.ops.aten.mul.Tensor(arg183_1, 0.9)
        add_39 = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
        add_40 = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
        add_41 = torch.ops.aten.add.Tensor(add_40, relu_3);  add_40 = relu_3 = None
        relu_6 = torch.ops.aten.relu.default(add_41);  add_41 = None
        convolution_8 = torch.ops.aten.convolution.default(relu_6, arg24_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg24_1 = None
        add_42 = torch.ops.aten.add.Tensor(arg187_1, 1)
        empty_8 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
        getitem_18 = var_mean_8[0]
        getitem_19 = var_mean_8[1];  var_mean_8 = None
        add_43 = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_8 = torch.ops.aten.sub.Tensor(convolution_8, getitem_19);  convolution_8 = None
        mul_56 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_24 = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_25 = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_57 = torch.ops.aten.mul.Tensor(squeeze_24, 0.1);  squeeze_24 = None
        mul_58 = torch.ops.aten.mul.Tensor(arg185_1, 0.9)
        add_44 = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
        squeeze_26 = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_59 = torch.ops.aten.mul.Tensor(squeeze_26, 1.0009775171065494);  squeeze_26 = None
        mul_60 = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
        mul_61 = torch.ops.aten.mul.Tensor(arg186_1, 0.9)
        add_45 = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        mul_62 = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
        add_46 = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
        relu_7 = torch.ops.aten.relu.default(add_46);  add_46 = None
        convolution_9 = torch.ops.aten.convolution.default(relu_7, arg27_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_7 = arg27_1 = None
        add_47 = torch.ops.aten.add.Tensor(arg190_1, 1)
        empty_9 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
        getitem_20 = var_mean_9[0]
        getitem_21 = var_mean_9[1];  var_mean_9 = None
        add_48 = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_9 = torch.ops.aten.sub.Tensor(convolution_9, getitem_21);  convolution_9 = None
        mul_63 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_27 = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_28 = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_64 = torch.ops.aten.mul.Tensor(squeeze_27, 0.1);  squeeze_27 = None
        mul_65 = torch.ops.aten.mul.Tensor(arg188_1, 0.9)
        add_49 = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
        squeeze_29 = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_66 = torch.ops.aten.mul.Tensor(squeeze_29, 1.0009775171065494);  squeeze_29 = None
        mul_67 = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
        mul_68 = torch.ops.aten.mul.Tensor(arg189_1, 0.9)
        add_50 = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
        add_51 = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
        relu_8 = torch.ops.aten.relu.default(add_51);  add_51 = None
        convolution_10 = torch.ops.aten.convolution.default(relu_8, arg30_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_8 = arg30_1 = None
        add_52 = torch.ops.aten.add.Tensor(arg193_1, 1)
        empty_10 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
        getitem_22 = var_mean_10[0]
        getitem_23 = var_mean_10[1];  var_mean_10 = None
        add_53 = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_10 = torch.ops.aten.sub.Tensor(convolution_10, getitem_23);  convolution_10 = None
        mul_70 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        squeeze_30 = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_31 = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_71 = torch.ops.aten.mul.Tensor(squeeze_30, 0.1);  squeeze_30 = None
        mul_72 = torch.ops.aten.mul.Tensor(arg191_1, 0.9)
        add_54 = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
        squeeze_32 = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_73 = torch.ops.aten.mul.Tensor(squeeze_32, 1.0009775171065494);  squeeze_32 = None
        mul_74 = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
        mul_75 = torch.ops.aten.mul.Tensor(arg192_1, 0.9)
        add_55 = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        unsqueeze_40 = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
        unsqueeze_41 = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        unsqueeze_42 = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_43 = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        mul_76 = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
        add_56 = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
        add_57 = torch.ops.aten.add.Tensor(add_56, relu_6);  add_56 = relu_6 = None
        relu_9 = torch.ops.aten.relu.default(add_57);  add_57 = None
        convolution_11 = torch.ops.aten.convolution.default(relu_9, arg33_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg33_1 = None
        add_58 = torch.ops.aten.add.Tensor(arg196_1, 1)
        empty_11 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
        getitem_24 = var_mean_11[0]
        getitem_25 = var_mean_11[1];  var_mean_11 = None
        add_59 = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        sub_11 = torch.ops.aten.sub.Tensor(convolution_11, getitem_25);  convolution_11 = None
        mul_77 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        squeeze_33 = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_34 = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_78 = torch.ops.aten.mul.Tensor(squeeze_33, 0.1);  squeeze_33 = None
        mul_79 = torch.ops.aten.mul.Tensor(arg194_1, 0.9)
        add_60 = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
        squeeze_35 = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_80 = torch.ops.aten.mul.Tensor(squeeze_35, 1.0009775171065494);  squeeze_35 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
        mul_82 = torch.ops.aten.mul.Tensor(arg195_1, 0.9)
        add_61 = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
        unsqueeze_44 = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_45 = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        unsqueeze_46 = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_47 = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        mul_83 = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
        add_62 = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
        relu_10 = torch.ops.aten.relu.default(add_62);  add_62 = None
        convolution_12 = torch.ops.aten.convolution.default(relu_10, arg36_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_10 = arg36_1 = None
        add_63 = torch.ops.aten.add.Tensor(arg199_1, 1)
        empty_12 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
        getitem_26 = var_mean_12[0]
        getitem_27 = var_mean_12[1];  var_mean_12 = None
        add_64 = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_12 = torch.ops.aten.sub.Tensor(convolution_12, getitem_27);  convolution_12 = None
        mul_84 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        squeeze_36 = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        squeeze_37 = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_85 = torch.ops.aten.mul.Tensor(squeeze_36, 0.1);  squeeze_36 = None
        mul_86 = torch.ops.aten.mul.Tensor(arg197_1, 0.9)
        add_65 = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
        squeeze_38 = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_87 = torch.ops.aten.mul.Tensor(squeeze_38, 1.003921568627451);  squeeze_38 = None
        mul_88 = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
        mul_89 = torch.ops.aten.mul.Tensor(arg198_1, 0.9)
        add_66 = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
        unsqueeze_48 = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_49 = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        unsqueeze_50 = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
        unsqueeze_51 = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        mul_90 = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
        add_67 = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
        relu_11 = torch.ops.aten.relu.default(add_67);  add_67 = None
        convolution_13 = torch.ops.aten.convolution.default(relu_11, arg39_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_11 = arg39_1 = None
        add_68 = torch.ops.aten.add.Tensor(arg202_1, 1)
        empty_13 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
        getitem_28 = var_mean_13[0]
        getitem_29 = var_mean_13[1];  var_mean_13 = None
        add_69 = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_13 = torch.ops.aten.sub.Tensor(convolution_13, getitem_29);  convolution_13 = None
        mul_91 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        squeeze_39 = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_40 = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_92 = torch.ops.aten.mul.Tensor(squeeze_39, 0.1);  squeeze_39 = None
        mul_93 = torch.ops.aten.mul.Tensor(arg200_1, 0.9)
        add_70 = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
        squeeze_41 = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_94 = torch.ops.aten.mul.Tensor(squeeze_41, 1.003921568627451);  squeeze_41 = None
        mul_95 = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
        mul_96 = torch.ops.aten.mul.Tensor(arg201_1, 0.9)
        add_71 = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
        unsqueeze_52 = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_53 = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        unsqueeze_54 = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
        unsqueeze_55 = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
        add_72 = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
        convolution_14 = torch.ops.aten.convolution.default(relu_9, arg42_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_9 = arg42_1 = None
        add_73 = torch.ops.aten.add.Tensor(arg205_1, 1)
        empty_14 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
        getitem_30 = var_mean_14[0]
        getitem_31 = var_mean_14[1];  var_mean_14 = None
        add_74 = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_14 = torch.ops.aten.sub.Tensor(convolution_14, getitem_31);  convolution_14 = None
        mul_98 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        squeeze_42 = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_43 = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_99 = torch.ops.aten.mul.Tensor(squeeze_42, 0.1);  squeeze_42 = None
        mul_100 = torch.ops.aten.mul.Tensor(arg203_1, 0.9)
        add_75 = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
        squeeze_44 = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_101 = torch.ops.aten.mul.Tensor(squeeze_44, 1.003921568627451);  squeeze_44 = None
        mul_102 = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
        mul_103 = torch.ops.aten.mul.Tensor(arg204_1, 0.9)
        add_76 = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
        unsqueeze_56 = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
        unsqueeze_57 = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        unsqueeze_58 = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_59 = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        mul_104 = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
        add_77 = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
        add_78 = torch.ops.aten.add.Tensor(add_72, add_77);  add_72 = add_77 = None
        relu_12 = torch.ops.aten.relu.default(add_78);  add_78 = None
        convolution_15 = torch.ops.aten.convolution.default(relu_12, arg45_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg45_1 = None
        add_79 = torch.ops.aten.add.Tensor(arg208_1, 1)
        empty_15 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
        getitem_32 = var_mean_15[0]
        getitem_33 = var_mean_15[1];  var_mean_15 = None
        add_80 = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_15 = torch.ops.aten.sub.Tensor(convolution_15, getitem_33);  convolution_15 = None
        mul_105 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        squeeze_45 = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_46 = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_106 = torch.ops.aten.mul.Tensor(squeeze_45, 0.1);  squeeze_45 = None
        mul_107 = torch.ops.aten.mul.Tensor(arg206_1, 0.9)
        add_81 = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
        squeeze_47 = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_108 = torch.ops.aten.mul.Tensor(squeeze_47, 1.003921568627451);  squeeze_47 = None
        mul_109 = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
        mul_110 = torch.ops.aten.mul.Tensor(arg207_1, 0.9)
        add_82 = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
        unsqueeze_60 = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
        unsqueeze_61 = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        unsqueeze_62 = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_63 = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        mul_111 = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
        add_83 = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
        relu_13 = torch.ops.aten.relu.default(add_83);  add_83 = None
        convolution_16 = torch.ops.aten.convolution.default(relu_13, arg48_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_13 = arg48_1 = None
        add_84 = torch.ops.aten.add.Tensor(arg211_1, 1)
        empty_16 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
        getitem_34 = var_mean_16[0]
        getitem_35 = var_mean_16[1];  var_mean_16 = None
        add_85 = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_16 = torch.ops.aten.sub.Tensor(convolution_16, getitem_35);  convolution_16 = None
        mul_112 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        squeeze_48 = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_49 = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_113 = torch.ops.aten.mul.Tensor(squeeze_48, 0.1);  squeeze_48 = None
        mul_114 = torch.ops.aten.mul.Tensor(arg209_1, 0.9)
        add_86 = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
        squeeze_50 = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_115 = torch.ops.aten.mul.Tensor(squeeze_50, 1.003921568627451);  squeeze_50 = None
        mul_116 = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
        mul_117 = torch.ops.aten.mul.Tensor(arg210_1, 0.9)
        add_87 = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        unsqueeze_64 = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_65 = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        unsqueeze_66 = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_67 = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        mul_118 = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
        add_88 = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
        relu_14 = torch.ops.aten.relu.default(add_88);  add_88 = None
        convolution_17 = torch.ops.aten.convolution.default(relu_14, arg51_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_14 = arg51_1 = None
        add_89 = torch.ops.aten.add.Tensor(arg214_1, 1)
        empty_17 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_17 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
        getitem_36 = var_mean_17[0]
        getitem_37 = var_mean_17[1];  var_mean_17 = None
        add_90 = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_17 = torch.ops.aten.sub.Tensor(convolution_17, getitem_37);  convolution_17 = None
        mul_119 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        squeeze_51 = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_52 = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_120 = torch.ops.aten.mul.Tensor(squeeze_51, 0.1);  squeeze_51 = None
        mul_121 = torch.ops.aten.mul.Tensor(arg212_1, 0.9)
        add_91 = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
        squeeze_53 = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_122 = torch.ops.aten.mul.Tensor(squeeze_53, 1.003921568627451);  squeeze_53 = None
        mul_123 = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
        mul_124 = torch.ops.aten.mul.Tensor(arg213_1, 0.9)
        add_92 = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
        unsqueeze_68 = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_69 = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        unsqueeze_70 = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
        unsqueeze_71 = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        mul_125 = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
        add_93 = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
        add_94 = torch.ops.aten.add.Tensor(add_93, relu_12);  add_93 = relu_12 = None
        relu_15 = torch.ops.aten.relu.default(add_94);  add_94 = None
        convolution_18 = torch.ops.aten.convolution.default(relu_15, arg54_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg54_1 = None
        add_95 = torch.ops.aten.add.Tensor(arg217_1, 1)
        empty_18 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_18 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
        getitem_38 = var_mean_18[0]
        getitem_39 = var_mean_18[1];  var_mean_18 = None
        add_96 = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_18 = torch.ops.aten.sub.Tensor(convolution_18, getitem_39);  convolution_18 = None
        mul_126 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        squeeze_54 = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        squeeze_55 = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_127 = torch.ops.aten.mul.Tensor(squeeze_54, 0.1);  squeeze_54 = None
        mul_128 = torch.ops.aten.mul.Tensor(arg215_1, 0.9)
        add_97 = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
        squeeze_56 = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_129 = torch.ops.aten.mul.Tensor(squeeze_56, 1.003921568627451);  squeeze_56 = None
        mul_130 = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
        mul_131 = torch.ops.aten.mul.Tensor(arg216_1, 0.9)
        add_98 = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
        unsqueeze_72 = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_73 = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        unsqueeze_74 = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
        unsqueeze_75 = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        mul_132 = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
        add_99 = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
        relu_16 = torch.ops.aten.relu.default(add_99);  add_99 = None
        convolution_19 = torch.ops.aten.convolution.default(relu_16, arg57_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_16 = arg57_1 = None
        add_100 = torch.ops.aten.add.Tensor(arg220_1, 1)
        empty_19 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_19 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
        getitem_40 = var_mean_19[0]
        getitem_41 = var_mean_19[1];  var_mean_19 = None
        add_101 = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        sub_19 = torch.ops.aten.sub.Tensor(convolution_19, getitem_41);  convolution_19 = None
        mul_133 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        squeeze_57 = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_58 = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_134 = torch.ops.aten.mul.Tensor(squeeze_57, 0.1);  squeeze_57 = None
        mul_135 = torch.ops.aten.mul.Tensor(arg218_1, 0.9)
        add_102 = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
        squeeze_59 = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_136 = torch.ops.aten.mul.Tensor(squeeze_59, 1.003921568627451);  squeeze_59 = None
        mul_137 = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
        mul_138 = torch.ops.aten.mul.Tensor(arg219_1, 0.9)
        add_103 = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
        unsqueeze_76 = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
        unsqueeze_77 = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        unsqueeze_78 = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_79 = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        mul_139 = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
        add_104 = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
        relu_17 = torch.ops.aten.relu.default(add_104);  add_104 = None
        convolution_20 = torch.ops.aten.convolution.default(relu_17, arg60_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_17 = arg60_1 = None
        add_105 = torch.ops.aten.add.Tensor(arg223_1, 1)
        empty_20 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_20 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
        getitem_42 = var_mean_20[0]
        getitem_43 = var_mean_20[1];  var_mean_20 = None
        add_106 = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_20 = torch.ops.aten.sub.Tensor(convolution_20, getitem_43);  convolution_20 = None
        mul_140 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        squeeze_60 = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        squeeze_61 = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_141 = torch.ops.aten.mul.Tensor(squeeze_60, 0.1);  squeeze_60 = None
        mul_142 = torch.ops.aten.mul.Tensor(arg221_1, 0.9)
        add_107 = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
        squeeze_62 = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_143 = torch.ops.aten.mul.Tensor(squeeze_62, 1.003921568627451);  squeeze_62 = None
        mul_144 = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
        mul_145 = torch.ops.aten.mul.Tensor(arg222_1, 0.9)
        add_108 = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
        unsqueeze_80 = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
        unsqueeze_81 = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        unsqueeze_82 = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_83 = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        mul_146 = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_81);  mul_140 = unsqueeze_81 = None
        add_109 = torch.ops.aten.add.Tensor(mul_146, unsqueeze_83);  mul_146 = unsqueeze_83 = None
        add_110 = torch.ops.aten.add.Tensor(add_109, relu_15);  add_109 = relu_15 = None
        relu_18 = torch.ops.aten.relu.default(add_110);  add_110 = None
        convolution_21 = torch.ops.aten.convolution.default(relu_18, arg63_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg63_1 = None
        add_111 = torch.ops.aten.add.Tensor(arg226_1, 1)
        empty_21 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_21 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
        getitem_44 = var_mean_21[0]
        getitem_45 = var_mean_21[1];  var_mean_21 = None
        add_112 = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        sub_21 = torch.ops.aten.sub.Tensor(convolution_21, getitem_45);  convolution_21 = None
        mul_147 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        squeeze_63 = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        squeeze_64 = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_148 = torch.ops.aten.mul.Tensor(squeeze_63, 0.1);  squeeze_63 = None
        mul_149 = torch.ops.aten.mul.Tensor(arg224_1, 0.9)
        add_113 = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
        squeeze_65 = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        mul_150 = torch.ops.aten.mul.Tensor(squeeze_65, 1.003921568627451);  squeeze_65 = None
        mul_151 = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
        mul_152 = torch.ops.aten.mul.Tensor(arg225_1, 0.9)
        add_114 = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
        unsqueeze_84 = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_85 = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        unsqueeze_86 = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_87 = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        mul_153 = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
        add_115 = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None
        relu_19 = torch.ops.aten.relu.default(add_115);  add_115 = None
        convolution_22 = torch.ops.aten.convolution.default(relu_19, arg66_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_19 = arg66_1 = None
        add_116 = torch.ops.aten.add.Tensor(arg229_1, 1)
        empty_22 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_22 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
        getitem_46 = var_mean_22[0]
        getitem_47 = var_mean_22[1];  var_mean_22 = None
        add_117 = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
        sub_22 = torch.ops.aten.sub.Tensor(convolution_22, getitem_47);  convolution_22 = None
        mul_154 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        squeeze_66 = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        squeeze_67 = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_155 = torch.ops.aten.mul.Tensor(squeeze_66, 0.1);  squeeze_66 = None
        mul_156 = torch.ops.aten.mul.Tensor(arg227_1, 0.9)
        add_118 = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
        squeeze_68 = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_157 = torch.ops.aten.mul.Tensor(squeeze_68, 1.003921568627451);  squeeze_68 = None
        mul_158 = torch.ops.aten.mul.Tensor(mul_157, 0.1);  mul_157 = None
        mul_159 = torch.ops.aten.mul.Tensor(arg228_1, 0.9)
        add_119 = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
        unsqueeze_88 = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_89 = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        unsqueeze_90 = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
        unsqueeze_91 = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        mul_160 = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
        add_120 = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None
        relu_20 = torch.ops.aten.relu.default(add_120);  add_120 = None
        convolution_23 = torch.ops.aten.convolution.default(relu_20, arg69_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_20 = arg69_1 = None
        add_121 = torch.ops.aten.add.Tensor(arg232_1, 1)
        empty_23 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_23 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
        getitem_48 = var_mean_23[0]
        getitem_49 = var_mean_23[1];  var_mean_23 = None
        add_122 = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        sub_23 = torch.ops.aten.sub.Tensor(convolution_23, getitem_49);  convolution_23 = None
        mul_161 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        squeeze_69 = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        squeeze_70 = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_162 = torch.ops.aten.mul.Tensor(squeeze_69, 0.1);  squeeze_69 = None
        mul_163 = torch.ops.aten.mul.Tensor(arg230_1, 0.9)
        add_123 = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
        squeeze_71 = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        mul_164 = torch.ops.aten.mul.Tensor(squeeze_71, 1.003921568627451);  squeeze_71 = None
        mul_165 = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
        mul_166 = torch.ops.aten.mul.Tensor(arg231_1, 0.9)
        add_124 = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
        unsqueeze_92 = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_93 = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        unsqueeze_94 = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
        unsqueeze_95 = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        mul_167 = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_93);  mul_161 = unsqueeze_93 = None
        add_125 = torch.ops.aten.add.Tensor(mul_167, unsqueeze_95);  mul_167 = unsqueeze_95 = None
        add_126 = torch.ops.aten.add.Tensor(add_125, relu_18);  add_125 = relu_18 = None
        relu_21 = torch.ops.aten.relu.default(add_126);  add_126 = None
        convolution_24 = torch.ops.aten.convolution.default(relu_21, arg72_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg72_1 = None
        add_127 = torch.ops.aten.add.Tensor(arg235_1, 1)
        empty_24 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_24 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
        getitem_50 = var_mean_24[0]
        getitem_51 = var_mean_24[1];  var_mean_24 = None
        add_128 = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        sub_24 = torch.ops.aten.sub.Tensor(convolution_24, getitem_51);  convolution_24 = None
        mul_168 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        squeeze_72 = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        squeeze_73 = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_169 = torch.ops.aten.mul.Tensor(squeeze_72, 0.1);  squeeze_72 = None
        mul_170 = torch.ops.aten.mul.Tensor(arg233_1, 0.9)
        add_129 = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
        squeeze_74 = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_171 = torch.ops.aten.mul.Tensor(squeeze_74, 1.003921568627451);  squeeze_74 = None
        mul_172 = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
        mul_173 = torch.ops.aten.mul.Tensor(arg234_1, 0.9)
        add_130 = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
        unsqueeze_96 = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
        unsqueeze_97 = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        unsqueeze_98 = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_99 = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        mul_174 = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_97);  mul_168 = unsqueeze_97 = None
        add_131 = torch.ops.aten.add.Tensor(mul_174, unsqueeze_99);  mul_174 = unsqueeze_99 = None
        relu_22 = torch.ops.aten.relu.default(add_131);  add_131 = None
        convolution_25 = torch.ops.aten.convolution.default(relu_22, arg75_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_22 = arg75_1 = None
        add_132 = torch.ops.aten.add.Tensor(arg238_1, 1)
        empty_25 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_25 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
        getitem_52 = var_mean_25[0]
        getitem_53 = var_mean_25[1];  var_mean_25 = None
        add_133 = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        sub_25 = torch.ops.aten.sub.Tensor(convolution_25, getitem_53);  convolution_25 = None
        mul_175 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        squeeze_75 = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        squeeze_76 = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_176 = torch.ops.aten.mul.Tensor(squeeze_75, 0.1);  squeeze_75 = None
        mul_177 = torch.ops.aten.mul.Tensor(arg236_1, 0.9)
        add_134 = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
        squeeze_77 = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
        mul_178 = torch.ops.aten.mul.Tensor(squeeze_77, 1.0158730158730158);  squeeze_77 = None
        mul_179 = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
        mul_180 = torch.ops.aten.mul.Tensor(arg237_1, 0.9)
        add_135 = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
        unsqueeze_100 = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
        unsqueeze_101 = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        unsqueeze_102 = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_103 = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        mul_181 = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
        add_136 = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None
        relu_23 = torch.ops.aten.relu.default(add_136);  add_136 = None
        convolution_26 = torch.ops.aten.convolution.default(relu_23, arg78_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_23 = arg78_1 = None
        add_137 = torch.ops.aten.add.Tensor(arg241_1, 1)
        empty_26 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_26 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
        getitem_54 = var_mean_26[0]
        getitem_55 = var_mean_26[1];  var_mean_26 = None
        add_138 = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        sub_26 = torch.ops.aten.sub.Tensor(convolution_26, getitem_55);  convolution_26 = None
        mul_182 = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        squeeze_78 = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        squeeze_79 = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_183 = torch.ops.aten.mul.Tensor(squeeze_78, 0.1);  squeeze_78 = None
        mul_184 = torch.ops.aten.mul.Tensor(arg239_1, 0.9)
        add_139 = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
        squeeze_80 = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
        mul_185 = torch.ops.aten.mul.Tensor(squeeze_80, 1.0158730158730158);  squeeze_80 = None
        mul_186 = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
        mul_187 = torch.ops.aten.mul.Tensor(arg240_1, 0.9)
        add_140 = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
        unsqueeze_104 = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_105 = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        unsqueeze_106 = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_107 = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        mul_188 = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_105);  mul_182 = unsqueeze_105 = None
        add_141 = torch.ops.aten.add.Tensor(mul_188, unsqueeze_107);  mul_188 = unsqueeze_107 = None
        convolution_27 = torch.ops.aten.convolution.default(relu_21, arg81_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_21 = arg81_1 = None
        add_142 = torch.ops.aten.add.Tensor(arg244_1, 1)
        empty_27 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_27 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
        getitem_56 = var_mean_27[0]
        getitem_57 = var_mean_27[1];  var_mean_27 = None
        add_143 = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
        sub_27 = torch.ops.aten.sub.Tensor(convolution_27, getitem_57);  convolution_27 = None
        mul_189 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        squeeze_81 = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        squeeze_82 = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_190 = torch.ops.aten.mul.Tensor(squeeze_81, 0.1);  squeeze_81 = None
        mul_191 = torch.ops.aten.mul.Tensor(arg242_1, 0.9)
        add_144 = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
        squeeze_83 = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        mul_192 = torch.ops.aten.mul.Tensor(squeeze_83, 1.0158730158730158);  squeeze_83 = None
        mul_193 = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
        mul_194 = torch.ops.aten.mul.Tensor(arg243_1, 0.9)
        add_145 = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
        unsqueeze_108 = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_109 = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        unsqueeze_110 = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
        unsqueeze_111 = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        mul_195 = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_109);  mul_189 = unsqueeze_109 = None
        add_146 = torch.ops.aten.add.Tensor(mul_195, unsqueeze_111);  mul_195 = unsqueeze_111 = None
        add_147 = torch.ops.aten.add.Tensor(add_141, add_146);  add_141 = add_146 = None
        relu_24 = torch.ops.aten.relu.default(add_147);  add_147 = None
        convolution_28 = torch.ops.aten.convolution.default(relu_24, arg84_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg84_1 = None
        add_148 = torch.ops.aten.add.Tensor(arg247_1, 1)
        empty_28 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_28 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
        getitem_58 = var_mean_28[0]
        getitem_59 = var_mean_28[1];  var_mean_28 = None
        add_149 = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        sub_28 = torch.ops.aten.sub.Tensor(convolution_28, getitem_59);  convolution_28 = None
        mul_196 = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        squeeze_84 = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        squeeze_85 = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_197 = torch.ops.aten.mul.Tensor(squeeze_84, 0.1);  squeeze_84 = None
        mul_198 = torch.ops.aten.mul.Tensor(arg245_1, 0.9)
        add_150 = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
        squeeze_86 = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        mul_199 = torch.ops.aten.mul.Tensor(squeeze_86, 1.0158730158730158);  squeeze_86 = None
        mul_200 = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
        mul_201 = torch.ops.aten.mul.Tensor(arg246_1, 0.9)
        add_151 = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
        unsqueeze_112 = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_113 = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        unsqueeze_114 = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
        unsqueeze_115 = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        mul_202 = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
        add_152 = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None
        relu_25 = torch.ops.aten.relu.default(add_152);  add_152 = None
        convolution_29 = torch.ops.aten.convolution.default(relu_25, arg87_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_25 = arg87_1 = None
        add_153 = torch.ops.aten.add.Tensor(arg250_1, 1)
        empty_29 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_29 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
        getitem_60 = var_mean_29[0]
        getitem_61 = var_mean_29[1];  var_mean_29 = None
        add_154 = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        sub_29 = torch.ops.aten.sub.Tensor(convolution_29, getitem_61);  convolution_29 = None
        mul_203 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        squeeze_87 = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        squeeze_88 = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
        mul_204 = torch.ops.aten.mul.Tensor(squeeze_87, 0.1);  squeeze_87 = None
        mul_205 = torch.ops.aten.mul.Tensor(arg248_1, 0.9)
        add_155 = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
        squeeze_89 = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_206 = torch.ops.aten.mul.Tensor(squeeze_89, 1.0158730158730158);  squeeze_89 = None
        mul_207 = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
        mul_208 = torch.ops.aten.mul.Tensor(arg249_1, 0.9)
        add_156 = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
        unsqueeze_116 = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
        unsqueeze_117 = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        unsqueeze_118 = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_119 = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        mul_209 = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_117);  mul_203 = unsqueeze_117 = None
        add_157 = torch.ops.aten.add.Tensor(mul_209, unsqueeze_119);  mul_209 = unsqueeze_119 = None
        relu_26 = torch.ops.aten.relu.default(add_157);  add_157 = None
        convolution_30 = torch.ops.aten.convolution.default(relu_26, arg90_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_26 = arg90_1 = None
        add_158 = torch.ops.aten.add.Tensor(arg253_1, 1)
        empty_30 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_30 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
        getitem_62 = var_mean_30[0]
        getitem_63 = var_mean_30[1];  var_mean_30 = None
        add_159 = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        sub_30 = torch.ops.aten.sub.Tensor(convolution_30, getitem_63);  convolution_30 = None
        mul_210 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        squeeze_90 = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        squeeze_91 = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_211 = torch.ops.aten.mul.Tensor(squeeze_90, 0.1);  squeeze_90 = None
        mul_212 = torch.ops.aten.mul.Tensor(arg251_1, 0.9)
        add_160 = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
        squeeze_92 = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_213 = torch.ops.aten.mul.Tensor(squeeze_92, 1.0158730158730158);  squeeze_92 = None
        mul_214 = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
        mul_215 = torch.ops.aten.mul.Tensor(arg252_1, 0.9)
        add_161 = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
        unsqueeze_120 = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
        unsqueeze_121 = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        unsqueeze_122 = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_123 = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        mul_216 = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_121);  mul_210 = unsqueeze_121 = None
        add_162 = torch.ops.aten.add.Tensor(mul_216, unsqueeze_123);  mul_216 = unsqueeze_123 = None
        add_163 = torch.ops.aten.add.Tensor(add_162, relu_24);  add_162 = relu_24 = None
        relu_27 = torch.ops.aten.relu.default(add_163);  add_163 = None
        convolution_31 = torch.ops.aten.convolution.default(relu_27, arg93_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg93_1 = None
        add_164 = torch.ops.aten.add.Tensor(arg256_1, 1)
        empty_31 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_31 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
        getitem_64 = var_mean_31[0]
        getitem_65 = var_mean_31[1];  var_mean_31 = None
        add_165 = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        sub_31 = torch.ops.aten.sub.Tensor(convolution_31, getitem_65);  convolution_31 = None
        mul_217 = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        squeeze_93 = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        squeeze_94 = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_218 = torch.ops.aten.mul.Tensor(squeeze_93, 0.1);  squeeze_93 = None
        mul_219 = torch.ops.aten.mul.Tensor(arg254_1, 0.9)
        add_166 = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
        squeeze_95 = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        mul_220 = torch.ops.aten.mul.Tensor(squeeze_95, 1.0158730158730158);  squeeze_95 = None
        mul_221 = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
        mul_222 = torch.ops.aten.mul.Tensor(arg255_1, 0.9)
        add_167 = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
        unsqueeze_124 = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_125 = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        unsqueeze_126 = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_127 = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        mul_223 = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
        add_168 = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None
        relu_28 = torch.ops.aten.relu.default(add_168);  add_168 = None
        convolution_32 = torch.ops.aten.convolution.default(relu_28, arg96_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_28 = arg96_1 = None
        add_169 = torch.ops.aten.add.Tensor(arg259_1, 1)
        empty_32 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_32 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
        getitem_66 = var_mean_32[0]
        getitem_67 = var_mean_32[1];  var_mean_32 = None
        add_170 = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        sub_32 = torch.ops.aten.sub.Tensor(convolution_32, getitem_67);  convolution_32 = None
        mul_224 = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
        squeeze_96 = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        squeeze_97 = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
        mul_225 = torch.ops.aten.mul.Tensor(squeeze_96, 0.1);  squeeze_96 = None
        mul_226 = torch.ops.aten.mul.Tensor(arg257_1, 0.9)
        add_171 = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
        squeeze_98 = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_227 = torch.ops.aten.mul.Tensor(squeeze_98, 1.0158730158730158);  squeeze_98 = None
        mul_228 = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
        mul_229 = torch.ops.aten.mul.Tensor(arg258_1, 0.9)
        add_172 = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
        unsqueeze_128 = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
        unsqueeze_129 = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        unsqueeze_130 = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
        unsqueeze_131 = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        mul_230 = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_129);  mul_224 = unsqueeze_129 = None
        add_173 = torch.ops.aten.add.Tensor(mul_230, unsqueeze_131);  mul_230 = unsqueeze_131 = None
        relu_29 = torch.ops.aten.relu.default(add_173);  add_173 = None
        convolution_33 = torch.ops.aten.convolution.default(relu_29, arg99_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_29 = arg99_1 = None
        add_174 = torch.ops.aten.add.Tensor(arg262_1, 1)
        empty_33 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_33 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
        getitem_68 = var_mean_33[0]
        getitem_69 = var_mean_33[1];  var_mean_33 = None
        add_175 = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
        rsqrt_33 = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
        sub_33 = torch.ops.aten.sub.Tensor(convolution_33, getitem_69);  convolution_33 = None
        mul_231 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        squeeze_99 = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        squeeze_100 = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_232 = torch.ops.aten.mul.Tensor(squeeze_99, 0.1);  squeeze_99 = None
        mul_233 = torch.ops.aten.mul.Tensor(arg260_1, 0.9)
        add_176 = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
        squeeze_101 = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_234 = torch.ops.aten.mul.Tensor(squeeze_101, 1.0158730158730158);  squeeze_101 = None
        mul_235 = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
        mul_236 = torch.ops.aten.mul.Tensor(arg261_1, 0.9)
        add_177 = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
        unsqueeze_132 = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_133 = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        unsqueeze_134 = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
        unsqueeze_135 = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        mul_237 = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_133);  mul_231 = unsqueeze_133 = None
        add_178 = torch.ops.aten.add.Tensor(mul_237, unsqueeze_135);  mul_237 = unsqueeze_135 = None
        add_179 = torch.ops.aten.add.Tensor(add_178, relu_27);  add_178 = relu_27 = None
        relu_30 = torch.ops.aten.relu.default(add_179);  add_179 = None
        convolution_34 = torch.ops.aten.convolution.default(relu_30, arg102_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg102_1 = None
        add_180 = torch.ops.aten.add.Tensor(arg265_1, 1)
        empty_34 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_34 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
        getitem_70 = var_mean_34[0]
        getitem_71 = var_mean_34[1];  var_mean_34 = None
        add_181 = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
        rsqrt_34 = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_34 = torch.ops.aten.sub.Tensor(convolution_34, getitem_71);  convolution_34 = None
        mul_238 = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        squeeze_102 = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        squeeze_103 = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_239 = torch.ops.aten.mul.Tensor(squeeze_102, 0.1);  squeeze_102 = None
        mul_240 = torch.ops.aten.mul.Tensor(arg263_1, 0.9)
        add_182 = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
        squeeze_104 = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        mul_241 = torch.ops.aten.mul.Tensor(squeeze_104, 1.0158730158730158);  squeeze_104 = None
        mul_242 = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
        mul_243 = torch.ops.aten.mul.Tensor(arg264_1, 0.9)
        add_183 = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
        unsqueeze_136 = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
        unsqueeze_137 = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        unsqueeze_138 = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_139 = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        mul_244 = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
        add_184 = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None
        relu_31 = torch.ops.aten.relu.default(add_184);  add_184 = None
        convolution_35 = torch.ops.aten.convolution.default(relu_31, arg105_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_31 = arg105_1 = None
        add_185 = torch.ops.aten.add.Tensor(arg268_1, 1)
        empty_35 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_35 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
        getitem_72 = var_mean_35[0]
        getitem_73 = var_mean_35[1];  var_mean_35 = None
        add_186 = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
        rsqrt_35 = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        sub_35 = torch.ops.aten.sub.Tensor(convolution_35, getitem_73);  convolution_35 = None
        mul_245 = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
        squeeze_105 = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        squeeze_106 = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
        mul_246 = torch.ops.aten.mul.Tensor(squeeze_105, 0.1);  squeeze_105 = None
        mul_247 = torch.ops.aten.mul.Tensor(arg266_1, 0.9)
        add_187 = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
        squeeze_107 = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        mul_248 = torch.ops.aten.mul.Tensor(squeeze_107, 1.0158730158730158);  squeeze_107 = None
        mul_249 = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
        mul_250 = torch.ops.aten.mul.Tensor(arg267_1, 0.9)
        add_188 = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
        unsqueeze_140 = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
        unsqueeze_141 = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        unsqueeze_142 = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_143 = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        mul_251 = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_141);  mul_245 = unsqueeze_141 = None
        add_189 = torch.ops.aten.add.Tensor(mul_251, unsqueeze_143);  mul_251 = unsqueeze_143 = None
        relu_32 = torch.ops.aten.relu.default(add_189);  add_189 = None
        convolution_36 = torch.ops.aten.convolution.default(relu_32, arg108_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_32 = arg108_1 = None
        add_190 = torch.ops.aten.add.Tensor(arg271_1, 1)
        empty_36 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_36 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
        getitem_74 = var_mean_36[0]
        getitem_75 = var_mean_36[1];  var_mean_36 = None
        add_191 = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
        rsqrt_36 = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
        sub_36 = torch.ops.aten.sub.Tensor(convolution_36, getitem_75);  convolution_36 = None
        mul_252 = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        squeeze_108 = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        squeeze_109 = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
        mul_253 = torch.ops.aten.mul.Tensor(squeeze_108, 0.1);  squeeze_108 = None
        mul_254 = torch.ops.aten.mul.Tensor(arg269_1, 0.9)
        add_192 = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
        squeeze_110 = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
        mul_255 = torch.ops.aten.mul.Tensor(squeeze_110, 1.0158730158730158);  squeeze_110 = None
        mul_256 = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
        mul_257 = torch.ops.aten.mul.Tensor(arg270_1, 0.9)
        add_193 = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
        unsqueeze_144 = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_145 = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        unsqueeze_146 = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_147 = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        mul_258 = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_145);  mul_252 = unsqueeze_145 = None
        add_194 = torch.ops.aten.add.Tensor(mul_258, unsqueeze_147);  mul_258 = unsqueeze_147 = None
        add_195 = torch.ops.aten.add.Tensor(add_194, relu_30);  add_194 = relu_30 = None
        relu_33 = torch.ops.aten.relu.default(add_195);  add_195 = None
        convolution_37 = torch.ops.aten.convolution.default(relu_33, arg111_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg111_1 = None
        add_196 = torch.ops.aten.add.Tensor(arg274_1, 1)
        empty_37 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_37 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
        getitem_76 = var_mean_37[0]
        getitem_77 = var_mean_37[1];  var_mean_37 = None
        add_197 = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
        rsqrt_37 = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
        sub_37 = torch.ops.aten.sub.Tensor(convolution_37, getitem_77);  convolution_37 = None
        mul_259 = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        squeeze_111 = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_112 = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_260 = torch.ops.aten.mul.Tensor(squeeze_111, 0.1);  squeeze_111 = None
        mul_261 = torch.ops.aten.mul.Tensor(arg272_1, 0.9)
        add_198 = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
        squeeze_113 = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_262 = torch.ops.aten.mul.Tensor(squeeze_113, 1.0158730158730158);  squeeze_113 = None
        mul_263 = torch.ops.aten.mul.Tensor(mul_262, 0.1);  mul_262 = None
        mul_264 = torch.ops.aten.mul.Tensor(arg273_1, 0.9)
        add_199 = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
        unsqueeze_148 = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_149 = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        unsqueeze_150 = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
        unsqueeze_151 = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        mul_265 = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
        add_200 = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None
        relu_34 = torch.ops.aten.relu.default(add_200);  add_200 = None
        convolution_38 = torch.ops.aten.convolution.default(relu_34, arg114_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_34 = arg114_1 = None
        add_201 = torch.ops.aten.add.Tensor(arg277_1, 1)
        empty_38 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_38 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
        getitem_78 = var_mean_38[0]
        getitem_79 = var_mean_38[1];  var_mean_38 = None
        add_202 = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
        rsqrt_38 = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
        sub_38 = torch.ops.aten.sub.Tensor(convolution_38, getitem_79);  convolution_38 = None
        mul_266 = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        squeeze_114 = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        squeeze_115 = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_267 = torch.ops.aten.mul.Tensor(squeeze_114, 0.1);  squeeze_114 = None
        mul_268 = torch.ops.aten.mul.Tensor(arg275_1, 0.9)
        add_203 = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
        squeeze_116 = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_269 = torch.ops.aten.mul.Tensor(squeeze_116, 1.0158730158730158);  squeeze_116 = None
        mul_270 = torch.ops.aten.mul.Tensor(mul_269, 0.1);  mul_269 = None
        mul_271 = torch.ops.aten.mul.Tensor(arg276_1, 0.9)
        add_204 = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
        unsqueeze_152 = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_153 = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        unsqueeze_154 = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_155 = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        mul_272 = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_153);  mul_266 = unsqueeze_153 = None
        add_205 = torch.ops.aten.add.Tensor(mul_272, unsqueeze_155);  mul_272 = unsqueeze_155 = None
        relu_35 = torch.ops.aten.relu.default(add_205);  add_205 = None
        convolution_39 = torch.ops.aten.convolution.default(relu_35, arg117_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_35 = arg117_1 = None
        add_206 = torch.ops.aten.add.Tensor(arg280_1, 1)
        empty_39 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_39 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
        getitem_80 = var_mean_39[0]
        getitem_81 = var_mean_39[1];  var_mean_39 = None
        add_207 = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
        rsqrt_39 = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
        sub_39 = torch.ops.aten.sub.Tensor(convolution_39, getitem_81);  convolution_39 = None
        mul_273 = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        squeeze_117 = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        squeeze_118 = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
        mul_274 = torch.ops.aten.mul.Tensor(squeeze_117, 0.1);  squeeze_117 = None
        mul_275 = torch.ops.aten.mul.Tensor(arg278_1, 0.9)
        add_208 = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
        squeeze_119 = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
        mul_276 = torch.ops.aten.mul.Tensor(squeeze_119, 1.0158730158730158);  squeeze_119 = None
        mul_277 = torch.ops.aten.mul.Tensor(mul_276, 0.1);  mul_276 = None
        mul_278 = torch.ops.aten.mul.Tensor(arg279_1, 0.9)
        add_209 = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
        unsqueeze_156 = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
        unsqueeze_157 = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        unsqueeze_158 = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_159 = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        mul_279 = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_157);  mul_273 = unsqueeze_157 = None
        add_210 = torch.ops.aten.add.Tensor(mul_279, unsqueeze_159);  mul_279 = unsqueeze_159 = None
        add_211 = torch.ops.aten.add.Tensor(add_210, relu_33);  add_210 = relu_33 = None
        relu_36 = torch.ops.aten.relu.default(add_211);  add_211 = None
        convolution_40 = torch.ops.aten.convolution.default(relu_36, arg120_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg120_1 = None
        add_212 = torch.ops.aten.add.Tensor(arg283_1, 1)
        empty_40 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_40 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
        getitem_82 = var_mean_40[0]
        getitem_83 = var_mean_40[1];  var_mean_40 = None
        add_213 = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
        rsqrt_40 = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
        sub_40 = torch.ops.aten.sub.Tensor(convolution_40, getitem_83);  convolution_40 = None
        mul_280 = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        squeeze_120 = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
        squeeze_121 = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_281 = torch.ops.aten.mul.Tensor(squeeze_120, 0.1);  squeeze_120 = None
        mul_282 = torch.ops.aten.mul.Tensor(arg281_1, 0.9)
        add_214 = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
        squeeze_122 = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
        mul_283 = torch.ops.aten.mul.Tensor(squeeze_122, 1.0158730158730158);  squeeze_122 = None
        mul_284 = torch.ops.aten.mul.Tensor(mul_283, 0.1);  mul_283 = None
        mul_285 = torch.ops.aten.mul.Tensor(arg282_1, 0.9)
        add_215 = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
        unsqueeze_160 = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_161 = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        unsqueeze_162 = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_163 = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        mul_286 = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
        add_216 = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None
        relu_37 = torch.ops.aten.relu.default(add_216);  add_216 = None
        convolution_41 = torch.ops.aten.convolution.default(relu_37, arg123_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_37 = arg123_1 = None
        add_217 = torch.ops.aten.add.Tensor(arg286_1, 1)
        empty_41 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_41 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
        getitem_84 = var_mean_41[0]
        getitem_85 = var_mean_41[1];  var_mean_41 = None
        add_218 = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
        sub_41 = torch.ops.aten.sub.Tensor(convolution_41, getitem_85);  convolution_41 = None
        mul_287 = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
        squeeze_123 = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
        squeeze_124 = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_288 = torch.ops.aten.mul.Tensor(squeeze_123, 0.1);  squeeze_123 = None
        mul_289 = torch.ops.aten.mul.Tensor(arg284_1, 0.9)
        add_219 = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
        squeeze_125 = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
        mul_290 = torch.ops.aten.mul.Tensor(squeeze_125, 1.0158730158730158);  squeeze_125 = None
        mul_291 = torch.ops.aten.mul.Tensor(mul_290, 0.1);  mul_290 = None
        mul_292 = torch.ops.aten.mul.Tensor(arg285_1, 0.9)
        add_220 = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
        unsqueeze_164 = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_165 = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        unsqueeze_166 = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_167 = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        mul_293 = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_165);  mul_287 = unsqueeze_165 = None
        add_221 = torch.ops.aten.add.Tensor(mul_293, unsqueeze_167);  mul_293 = unsqueeze_167 = None
        relu_38 = torch.ops.aten.relu.default(add_221);  add_221 = None
        convolution_42 = torch.ops.aten.convolution.default(relu_38, arg126_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_38 = arg126_1 = None
        add_222 = torch.ops.aten.add.Tensor(arg289_1, 1)
        empty_42 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_42 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
        getitem_86 = var_mean_42[0]
        getitem_87 = var_mean_42[1];  var_mean_42 = None
        add_223 = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
        rsqrt_42 = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
        sub_42 = torch.ops.aten.sub.Tensor(convolution_42, getitem_87);  convolution_42 = None
        mul_294 = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        squeeze_126 = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        squeeze_127 = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_295 = torch.ops.aten.mul.Tensor(squeeze_126, 0.1);  squeeze_126 = None
        mul_296 = torch.ops.aten.mul.Tensor(arg287_1, 0.9)
        add_224 = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
        squeeze_128 = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
        mul_297 = torch.ops.aten.mul.Tensor(squeeze_128, 1.0158730158730158);  squeeze_128 = None
        mul_298 = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
        mul_299 = torch.ops.aten.mul.Tensor(arg288_1, 0.9)
        add_225 = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
        unsqueeze_168 = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_169 = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        unsqueeze_170 = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
        unsqueeze_171 = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        mul_300 = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_169);  mul_294 = unsqueeze_169 = None
        add_226 = torch.ops.aten.add.Tensor(mul_300, unsqueeze_171);  mul_300 = unsqueeze_171 = None
        add_227 = torch.ops.aten.add.Tensor(add_226, relu_36);  add_226 = relu_36 = None
        relu_39 = torch.ops.aten.relu.default(add_227);  add_227 = None
        convolution_43 = torch.ops.aten.convolution.default(relu_39, arg129_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg129_1 = None
        add_228 = torch.ops.aten.add.Tensor(arg292_1, 1)
        empty_43 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_43 = torch.ops.aten.var_mean.correction(convolution_43, [0, 2, 3], correction = 0, keepdim = True)
        getitem_88 = var_mean_43[0]
        getitem_89 = var_mean_43[1];  var_mean_43 = None
        add_229 = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
        rsqrt_43 = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        sub_43 = torch.ops.aten.sub.Tensor(convolution_43, getitem_89);  convolution_43 = None
        mul_301 = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        squeeze_129 = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
        squeeze_130 = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
        mul_302 = torch.ops.aten.mul.Tensor(squeeze_129, 0.1);  squeeze_129 = None
        mul_303 = torch.ops.aten.mul.Tensor(arg290_1, 0.9)
        add_230 = torch.ops.aten.add.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
        squeeze_131 = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
        mul_304 = torch.ops.aten.mul.Tensor(squeeze_131, 1.0158730158730158);  squeeze_131 = None
        mul_305 = torch.ops.aten.mul.Tensor(mul_304, 0.1);  mul_304 = None
        mul_306 = torch.ops.aten.mul.Tensor(arg291_1, 0.9)
        add_231 = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
        unsqueeze_172 = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_173 = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        unsqueeze_174 = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
        unsqueeze_175 = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        mul_307 = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_173);  mul_301 = unsqueeze_173 = None
        add_232 = torch.ops.aten.add.Tensor(mul_307, unsqueeze_175);  mul_307 = unsqueeze_175 = None
        relu_40 = torch.ops.aten.relu.default(add_232);  add_232 = None
        convolution_44 = torch.ops.aten.convolution.default(relu_40, arg132_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_40 = arg132_1 = None
        add_233 = torch.ops.aten.add.Tensor(arg295_1, 1)
        empty_44 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_44 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
        getitem_90 = var_mean_44[0]
        getitem_91 = var_mean_44[1];  var_mean_44 = None
        add_234 = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
        rsqrt_44 = torch.ops.aten.rsqrt.default(add_234);  add_234 = None
        sub_44 = torch.ops.aten.sub.Tensor(convolution_44, getitem_91);  convolution_44 = None
        mul_308 = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
        squeeze_132 = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
        squeeze_133 = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
        mul_309 = torch.ops.aten.mul.Tensor(squeeze_132, 0.1);  squeeze_132 = None
        mul_310 = torch.ops.aten.mul.Tensor(arg293_1, 0.9)
        add_235 = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
        squeeze_134 = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
        mul_311 = torch.ops.aten.mul.Tensor(squeeze_134, 1.0666666666666667);  squeeze_134 = None
        mul_312 = torch.ops.aten.mul.Tensor(mul_311, 0.1);  mul_311 = None
        mul_313 = torch.ops.aten.mul.Tensor(arg294_1, 0.9)
        add_236 = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
        unsqueeze_176 = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_177 = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        unsqueeze_178 = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_179 = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
        mul_314 = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_177);  mul_308 = unsqueeze_177 = None
        add_237 = torch.ops.aten.add.Tensor(mul_314, unsqueeze_179);  mul_314 = unsqueeze_179 = None
        relu_41 = torch.ops.aten.relu.default(add_237);  add_237 = None
        convolution_45 = torch.ops.aten.convolution.default(relu_41, arg135_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_41 = arg135_1 = None
        add_238 = torch.ops.aten.add.Tensor(arg298_1, 1)
        empty_45 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_45 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
        getitem_92 = var_mean_45[0]
        getitem_93 = var_mean_45[1];  var_mean_45 = None
        add_239 = torch.ops.aten.add.Tensor(getitem_92, 1e-05)
        rsqrt_45 = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
        sub_45 = torch.ops.aten.sub.Tensor(convolution_45, getitem_93);  convolution_45 = None
        mul_315 = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        squeeze_135 = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
        squeeze_136 = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
        mul_316 = torch.ops.aten.mul.Tensor(squeeze_135, 0.1);  squeeze_135 = None
        mul_317 = torch.ops.aten.mul.Tensor(arg296_1, 0.9)
        add_240 = torch.ops.aten.add.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
        squeeze_137 = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
        mul_318 = torch.ops.aten.mul.Tensor(squeeze_137, 1.0666666666666667);  squeeze_137 = None
        mul_319 = torch.ops.aten.mul.Tensor(mul_318, 0.1);  mul_318 = None
        mul_320 = torch.ops.aten.mul.Tensor(arg297_1, 0.9)
        add_241 = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
        unsqueeze_180 = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
        unsqueeze_181 = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        unsqueeze_182 = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_183 = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        mul_321 = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_181);  mul_315 = unsqueeze_181 = None
        add_242 = torch.ops.aten.add.Tensor(mul_321, unsqueeze_183);  mul_321 = unsqueeze_183 = None
        convolution_46 = torch.ops.aten.convolution.default(relu_39, arg138_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_39 = arg138_1 = None
        add_243 = torch.ops.aten.add.Tensor(arg301_1, 1)
        empty_46 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_46 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
        getitem_94 = var_mean_46[0]
        getitem_95 = var_mean_46[1];  var_mean_46 = None
        add_244 = torch.ops.aten.add.Tensor(getitem_94, 1e-05)
        rsqrt_46 = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
        sub_46 = torch.ops.aten.sub.Tensor(convolution_46, getitem_95);  convolution_46 = None
        mul_322 = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
        squeeze_138 = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
        squeeze_139 = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
        mul_323 = torch.ops.aten.mul.Tensor(squeeze_138, 0.1);  squeeze_138 = None
        mul_324 = torch.ops.aten.mul.Tensor(arg299_1, 0.9)
        add_245 = torch.ops.aten.add.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
        squeeze_140 = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
        mul_325 = torch.ops.aten.mul.Tensor(squeeze_140, 1.0666666666666667);  squeeze_140 = None
        mul_326 = torch.ops.aten.mul.Tensor(mul_325, 0.1);  mul_325 = None
        mul_327 = torch.ops.aten.mul.Tensor(arg300_1, 0.9)
        add_246 = torch.ops.aten.add.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
        unsqueeze_184 = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_185 = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        unsqueeze_186 = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_187 = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        mul_328 = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_185);  mul_322 = unsqueeze_185 = None
        add_247 = torch.ops.aten.add.Tensor(mul_328, unsqueeze_187);  mul_328 = unsqueeze_187 = None
        add_248 = torch.ops.aten.add.Tensor(add_242, add_247);  add_242 = add_247 = None
        relu_42 = torch.ops.aten.relu.default(add_248);  add_248 = None
        convolution_47 = torch.ops.aten.convolution.default(relu_42, arg141_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg141_1 = None
        add_249 = torch.ops.aten.add.Tensor(arg304_1, 1)
        empty_47 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_47 = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
        getitem_96 = var_mean_47[0]
        getitem_97 = var_mean_47[1];  var_mean_47 = None
        add_250 = torch.ops.aten.add.Tensor(getitem_96, 1e-05)
        rsqrt_47 = torch.ops.aten.rsqrt.default(add_250);  add_250 = None
        sub_47 = torch.ops.aten.sub.Tensor(convolution_47, getitem_97);  convolution_47 = None
        mul_329 = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
        squeeze_141 = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
        squeeze_142 = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
        mul_330 = torch.ops.aten.mul.Tensor(squeeze_141, 0.1);  squeeze_141 = None
        mul_331 = torch.ops.aten.mul.Tensor(arg302_1, 0.9)
        add_251 = torch.ops.aten.add.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
        squeeze_143 = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
        mul_332 = torch.ops.aten.mul.Tensor(squeeze_143, 1.0666666666666667);  squeeze_143 = None
        mul_333 = torch.ops.aten.mul.Tensor(mul_332, 0.1);  mul_332 = None
        mul_334 = torch.ops.aten.mul.Tensor(arg303_1, 0.9)
        add_252 = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
        unsqueeze_188 = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_189 = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        unsqueeze_190 = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
        unsqueeze_191 = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        mul_335 = torch.ops.aten.mul.Tensor(mul_329, unsqueeze_189);  mul_329 = unsqueeze_189 = None
        add_253 = torch.ops.aten.add.Tensor(mul_335, unsqueeze_191);  mul_335 = unsqueeze_191 = None
        relu_43 = torch.ops.aten.relu.default(add_253);  add_253 = None
        convolution_48 = torch.ops.aten.convolution.default(relu_43, arg144_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_43 = arg144_1 = None
        add_254 = torch.ops.aten.add.Tensor(arg307_1, 1)
        empty_48 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_48 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
        getitem_98 = var_mean_48[0]
        getitem_99 = var_mean_48[1];  var_mean_48 = None
        add_255 = torch.ops.aten.add.Tensor(getitem_98, 1e-05)
        rsqrt_48 = torch.ops.aten.rsqrt.default(add_255);  add_255 = None
        sub_48 = torch.ops.aten.sub.Tensor(convolution_48, getitem_99);  convolution_48 = None
        mul_336 = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
        squeeze_144 = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
        squeeze_145 = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
        mul_337 = torch.ops.aten.mul.Tensor(squeeze_144, 0.1);  squeeze_144 = None
        mul_338 = torch.ops.aten.mul.Tensor(arg305_1, 0.9)
        add_256 = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
        squeeze_146 = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
        mul_339 = torch.ops.aten.mul.Tensor(squeeze_146, 1.0666666666666667);  squeeze_146 = None
        mul_340 = torch.ops.aten.mul.Tensor(mul_339, 0.1);  mul_339 = None
        mul_341 = torch.ops.aten.mul.Tensor(arg306_1, 0.9)
        add_257 = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
        unsqueeze_192 = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
        unsqueeze_193 = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        unsqueeze_194 = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
        unsqueeze_195 = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        mul_342 = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_193);  mul_336 = unsqueeze_193 = None
        add_258 = torch.ops.aten.add.Tensor(mul_342, unsqueeze_195);  mul_342 = unsqueeze_195 = None
        relu_44 = torch.ops.aten.relu.default(add_258);  add_258 = None
        convolution_49 = torch.ops.aten.convolution.default(relu_44, arg147_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_44 = arg147_1 = None
        add_259 = torch.ops.aten.add.Tensor(arg310_1, 1)
        empty_49 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_49 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
        getitem_100 = var_mean_49[0]
        getitem_101 = var_mean_49[1];  var_mean_49 = None
        add_260 = torch.ops.aten.add.Tensor(getitem_100, 1e-05)
        rsqrt_49 = torch.ops.aten.rsqrt.default(add_260);  add_260 = None
        sub_49 = torch.ops.aten.sub.Tensor(convolution_49, getitem_101);  convolution_49 = None
        mul_343 = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
        squeeze_147 = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
        squeeze_148 = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
        mul_344 = torch.ops.aten.mul.Tensor(squeeze_147, 0.1);  squeeze_147 = None
        mul_345 = torch.ops.aten.mul.Tensor(arg308_1, 0.9)
        add_261 = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
        squeeze_149 = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
        mul_346 = torch.ops.aten.mul.Tensor(squeeze_149, 1.0666666666666667);  squeeze_149 = None
        mul_347 = torch.ops.aten.mul.Tensor(mul_346, 0.1);  mul_346 = None
        mul_348 = torch.ops.aten.mul.Tensor(arg309_1, 0.9)
        add_262 = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
        unsqueeze_196 = torch.ops.aten.unsqueeze.default(arg148_1, -1);  arg148_1 = None
        unsqueeze_197 = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        unsqueeze_198 = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_199 = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        mul_349 = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_197);  mul_343 = unsqueeze_197 = None
        add_263 = torch.ops.aten.add.Tensor(mul_349, unsqueeze_199);  mul_349 = unsqueeze_199 = None
        add_264 = torch.ops.aten.add.Tensor(add_263, relu_42);  add_263 = relu_42 = None
        relu_45 = torch.ops.aten.relu.default(add_264);  add_264 = None
        convolution_50 = torch.ops.aten.convolution.default(relu_45, arg150_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg150_1 = None
        add_265 = torch.ops.aten.add.Tensor(arg313_1, 1)
        empty_50 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_50 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
        getitem_102 = var_mean_50[0]
        getitem_103 = var_mean_50[1];  var_mean_50 = None
        add_266 = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
        rsqrt_50 = torch.ops.aten.rsqrt.default(add_266);  add_266 = None
        sub_50 = torch.ops.aten.sub.Tensor(convolution_50, getitem_103);  convolution_50 = None
        mul_350 = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
        squeeze_150 = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
        squeeze_151 = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
        mul_351 = torch.ops.aten.mul.Tensor(squeeze_150, 0.1);  squeeze_150 = None
        mul_352 = torch.ops.aten.mul.Tensor(arg311_1, 0.9)
        add_267 = torch.ops.aten.add.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
        squeeze_152 = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
        mul_353 = torch.ops.aten.mul.Tensor(squeeze_152, 1.0666666666666667);  squeeze_152 = None
        mul_354 = torch.ops.aten.mul.Tensor(mul_353, 0.1);  mul_353 = None
        mul_355 = torch.ops.aten.mul.Tensor(arg312_1, 0.9)
        add_268 = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
        unsqueeze_200 = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_201 = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        unsqueeze_202 = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_203 = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        mul_356 = torch.ops.aten.mul.Tensor(mul_350, unsqueeze_201);  mul_350 = unsqueeze_201 = None
        add_269 = torch.ops.aten.add.Tensor(mul_356, unsqueeze_203);  mul_356 = unsqueeze_203 = None
        relu_46 = torch.ops.aten.relu.default(add_269);  add_269 = None
        convolution_51 = torch.ops.aten.convolution.default(relu_46, arg153_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_46 = arg153_1 = None
        add_270 = torch.ops.aten.add.Tensor(arg316_1, 1)
        empty_51 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_51 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
        getitem_104 = var_mean_51[0]
        getitem_105 = var_mean_51[1];  var_mean_51 = None
        add_271 = torch.ops.aten.add.Tensor(getitem_104, 1e-05)
        rsqrt_51 = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
        sub_51 = torch.ops.aten.sub.Tensor(convolution_51, getitem_105);  convolution_51 = None
        mul_357 = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
        squeeze_153 = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
        squeeze_154 = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
        mul_358 = torch.ops.aten.mul.Tensor(squeeze_153, 0.1);  squeeze_153 = None
        mul_359 = torch.ops.aten.mul.Tensor(arg314_1, 0.9)
        add_272 = torch.ops.aten.add.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
        squeeze_155 = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
        mul_360 = torch.ops.aten.mul.Tensor(squeeze_155, 1.0666666666666667);  squeeze_155 = None
        mul_361 = torch.ops.aten.mul.Tensor(mul_360, 0.1);  mul_360 = None
        mul_362 = torch.ops.aten.mul.Tensor(arg315_1, 0.9)
        add_273 = torch.ops.aten.add.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
        unsqueeze_204 = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
        unsqueeze_205 = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        unsqueeze_206 = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
        unsqueeze_207 = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        mul_363 = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
        add_274 = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None
        relu_47 = torch.ops.aten.relu.default(add_274);  add_274 = None
        convolution_52 = torch.ops.aten.convolution.default(relu_47, arg156_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_47 = arg156_1 = None
        add_275 = torch.ops.aten.add.Tensor(arg319_1, 1)
        empty_52 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
        var_mean_52 = torch.ops.aten.var_mean.correction(convolution_52, [0, 2, 3], correction = 0, keepdim = True)
        getitem_106 = var_mean_52[0]
        getitem_107 = var_mean_52[1];  var_mean_52 = None
        add_276 = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
        rsqrt_52 = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
        sub_52 = torch.ops.aten.sub.Tensor(convolution_52, getitem_107);  convolution_52 = None
        mul_364 = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
        squeeze_156 = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
        squeeze_157 = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
        mul_365 = torch.ops.aten.mul.Tensor(squeeze_156, 0.1);  squeeze_156 = None
        mul_366 = torch.ops.aten.mul.Tensor(arg317_1, 0.9)
        add_277 = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
        squeeze_158 = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
        mul_367 = torch.ops.aten.mul.Tensor(squeeze_158, 1.0666666666666667);  squeeze_158 = None
        mul_368 = torch.ops.aten.mul.Tensor(mul_367, 0.1);  mul_367 = None
        mul_369 = torch.ops.aten.mul.Tensor(arg318_1, 0.9)
        add_278 = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
        unsqueeze_208 = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_209 = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
        unsqueeze_210 = torch.ops.aten.unsqueeze.default(arg158_1, -1);  arg158_1 = None
        unsqueeze_211 = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
        mul_370 = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_209);  mul_364 = unsqueeze_209 = None
        add_279 = torch.ops.aten.add.Tensor(mul_370, unsqueeze_211);  mul_370 = unsqueeze_211 = None
        add_280 = torch.ops.aten.add.Tensor(add_279, relu_45);  add_279 = relu_45 = None
        relu_48 = torch.ops.aten.relu.default(add_280);  add_280 = None
        mean = torch.ops.aten.mean.dim(relu_48, [-1, -2], True);  relu_48 = None
        view = torch.ops.aten.view.default(mean, [16, 2048]);  mean = None
        permute = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm = torch.ops.aten.addmm.default(arg160_1, view, permute);  arg160_1 = view = permute = None
        copy_ = torch.ops.aten.copy_.default(arg161_1, add_2);  arg161_1 = add_2 = None
        copy__1 = torch.ops.aten.copy_.default(arg162_1, add_3);  arg162_1 = add_3 = None
        copy__2 = torch.ops.aten.copy_.default(arg163_1, add);  arg163_1 = add = None
        copy__3 = torch.ops.aten.copy_.default(arg164_1, add_7);  arg164_1 = add_7 = None
        copy__4 = torch.ops.aten.copy_.default(arg165_1, add_8);  arg165_1 = add_8 = None
        copy__5 = torch.ops.aten.copy_.default(arg166_1, add_5);  arg166_1 = add_5 = None
        copy__6 = torch.ops.aten.copy_.default(arg167_1, add_12);  arg167_1 = add_12 = None
        copy__7 = torch.ops.aten.copy_.default(arg168_1, add_13);  arg168_1 = add_13 = None
        copy__8 = torch.ops.aten.copy_.default(arg169_1, add_10);  arg169_1 = add_10 = None
        copy__9 = torch.ops.aten.copy_.default(arg170_1, add_17);  arg170_1 = add_17 = None
        copy__10 = torch.ops.aten.copy_.default(arg171_1, add_18);  arg171_1 = add_18 = None
        copy__11 = torch.ops.aten.copy_.default(arg172_1, add_15);  arg172_1 = add_15 = None
        copy__12 = torch.ops.aten.copy_.default(arg173_1, add_22);  arg173_1 = add_22 = None
        copy__13 = torch.ops.aten.copy_.default(arg174_1, add_23);  arg174_1 = add_23 = None
        copy__14 = torch.ops.aten.copy_.default(arg175_1, add_20);  arg175_1 = add_20 = None
        copy__15 = torch.ops.aten.copy_.default(arg176_1, add_28);  arg176_1 = add_28 = None
        copy__16 = torch.ops.aten.copy_.default(arg177_1, add_29);  arg177_1 = add_29 = None
        copy__17 = torch.ops.aten.copy_.default(arg178_1, add_26);  arg178_1 = add_26 = None
        copy__18 = torch.ops.aten.copy_.default(arg179_1, add_33);  arg179_1 = add_33 = None
        copy__19 = torch.ops.aten.copy_.default(arg180_1, add_34);  arg180_1 = add_34 = None
        copy__20 = torch.ops.aten.copy_.default(arg181_1, add_31);  arg181_1 = add_31 = None
        copy__21 = torch.ops.aten.copy_.default(arg182_1, add_38);  arg182_1 = add_38 = None
        copy__22 = torch.ops.aten.copy_.default(arg183_1, add_39);  arg183_1 = add_39 = None
        copy__23 = torch.ops.aten.copy_.default(arg184_1, add_36);  arg184_1 = add_36 = None
        copy__24 = torch.ops.aten.copy_.default(arg185_1, add_44);  arg185_1 = add_44 = None
        copy__25 = torch.ops.aten.copy_.default(arg186_1, add_45);  arg186_1 = add_45 = None
        copy__26 = torch.ops.aten.copy_.default(arg187_1, add_42);  arg187_1 = add_42 = None
        copy__27 = torch.ops.aten.copy_.default(arg188_1, add_49);  arg188_1 = add_49 = None
        copy__28 = torch.ops.aten.copy_.default(arg189_1, add_50);  arg189_1 = add_50 = None
        copy__29 = torch.ops.aten.copy_.default(arg190_1, add_47);  arg190_1 = add_47 = None
        copy__30 = torch.ops.aten.copy_.default(arg191_1, add_54);  arg191_1 = add_54 = None
        copy__31 = torch.ops.aten.copy_.default(arg192_1, add_55);  arg192_1 = add_55 = None
        copy__32 = torch.ops.aten.copy_.default(arg193_1, add_52);  arg193_1 = add_52 = None
        copy__33 = torch.ops.aten.copy_.default(arg194_1, add_60);  arg194_1 = add_60 = None
        copy__34 = torch.ops.aten.copy_.default(arg195_1, add_61);  arg195_1 = add_61 = None
        copy__35 = torch.ops.aten.copy_.default(arg196_1, add_58);  arg196_1 = add_58 = None
        copy__36 = torch.ops.aten.copy_.default(arg197_1, add_65);  arg197_1 = add_65 = None
        copy__37 = torch.ops.aten.copy_.default(arg198_1, add_66);  arg198_1 = add_66 = None
        copy__38 = torch.ops.aten.copy_.default(arg199_1, add_63);  arg199_1 = add_63 = None
        copy__39 = torch.ops.aten.copy_.default(arg200_1, add_70);  arg200_1 = add_70 = None
        copy__40 = torch.ops.aten.copy_.default(arg201_1, add_71);  arg201_1 = add_71 = None
        copy__41 = torch.ops.aten.copy_.default(arg202_1, add_68);  arg202_1 = add_68 = None
        copy__42 = torch.ops.aten.copy_.default(arg203_1, add_75);  arg203_1 = add_75 = None
        copy__43 = torch.ops.aten.copy_.default(arg204_1, add_76);  arg204_1 = add_76 = None
        copy__44 = torch.ops.aten.copy_.default(arg205_1, add_73);  arg205_1 = add_73 = None
        copy__45 = torch.ops.aten.copy_.default(arg206_1, add_81);  arg206_1 = add_81 = None
        copy__46 = torch.ops.aten.copy_.default(arg207_1, add_82);  arg207_1 = add_82 = None
        copy__47 = torch.ops.aten.copy_.default(arg208_1, add_79);  arg208_1 = add_79 = None
        copy__48 = torch.ops.aten.copy_.default(arg209_1, add_86);  arg209_1 = add_86 = None
        copy__49 = torch.ops.aten.copy_.default(arg210_1, add_87);  arg210_1 = add_87 = None
        copy__50 = torch.ops.aten.copy_.default(arg211_1, add_84);  arg211_1 = add_84 = None
        copy__51 = torch.ops.aten.copy_.default(arg212_1, add_91);  arg212_1 = add_91 = None
        copy__52 = torch.ops.aten.copy_.default(arg213_1, add_92);  arg213_1 = add_92 = None
        copy__53 = torch.ops.aten.copy_.default(arg214_1, add_89);  arg214_1 = add_89 = None
        copy__54 = torch.ops.aten.copy_.default(arg215_1, add_97);  arg215_1 = add_97 = None
        copy__55 = torch.ops.aten.copy_.default(arg216_1, add_98);  arg216_1 = add_98 = None
        copy__56 = torch.ops.aten.copy_.default(arg217_1, add_95);  arg217_1 = add_95 = None
        copy__57 = torch.ops.aten.copy_.default(arg218_1, add_102);  arg218_1 = add_102 = None
        copy__58 = torch.ops.aten.copy_.default(arg219_1, add_103);  arg219_1 = add_103 = None
        copy__59 = torch.ops.aten.copy_.default(arg220_1, add_100);  arg220_1 = add_100 = None
        copy__60 = torch.ops.aten.copy_.default(arg221_1, add_107);  arg221_1 = add_107 = None
        copy__61 = torch.ops.aten.copy_.default(arg222_1, add_108);  arg222_1 = add_108 = None
        copy__62 = torch.ops.aten.copy_.default(arg223_1, add_105);  arg223_1 = add_105 = None
        copy__63 = torch.ops.aten.copy_.default(arg224_1, add_113);  arg224_1 = add_113 = None
        copy__64 = torch.ops.aten.copy_.default(arg225_1, add_114);  arg225_1 = add_114 = None
        copy__65 = torch.ops.aten.copy_.default(arg226_1, add_111);  arg226_1 = add_111 = None
        copy__66 = torch.ops.aten.copy_.default(arg227_1, add_118);  arg227_1 = add_118 = None
        copy__67 = torch.ops.aten.copy_.default(arg228_1, add_119);  arg228_1 = add_119 = None
        copy__68 = torch.ops.aten.copy_.default(arg229_1, add_116);  arg229_1 = add_116 = None
        copy__69 = torch.ops.aten.copy_.default(arg230_1, add_123);  arg230_1 = add_123 = None
        copy__70 = torch.ops.aten.copy_.default(arg231_1, add_124);  arg231_1 = add_124 = None
        copy__71 = torch.ops.aten.copy_.default(arg232_1, add_121);  arg232_1 = add_121 = None
        copy__72 = torch.ops.aten.copy_.default(arg233_1, add_129);  arg233_1 = add_129 = None
        copy__73 = torch.ops.aten.copy_.default(arg234_1, add_130);  arg234_1 = add_130 = None
        copy__74 = torch.ops.aten.copy_.default(arg235_1, add_127);  arg235_1 = add_127 = None
        copy__75 = torch.ops.aten.copy_.default(arg236_1, add_134);  arg236_1 = add_134 = None
        copy__76 = torch.ops.aten.copy_.default(arg237_1, add_135);  arg237_1 = add_135 = None
        copy__77 = torch.ops.aten.copy_.default(arg238_1, add_132);  arg238_1 = add_132 = None
        copy__78 = torch.ops.aten.copy_.default(arg239_1, add_139);  arg239_1 = add_139 = None
        copy__79 = torch.ops.aten.copy_.default(arg240_1, add_140);  arg240_1 = add_140 = None
        copy__80 = torch.ops.aten.copy_.default(arg241_1, add_137);  arg241_1 = add_137 = None
        copy__81 = torch.ops.aten.copy_.default(arg242_1, add_144);  arg242_1 = add_144 = None
        copy__82 = torch.ops.aten.copy_.default(arg243_1, add_145);  arg243_1 = add_145 = None
        copy__83 = torch.ops.aten.copy_.default(arg244_1, add_142);  arg244_1 = add_142 = None
        copy__84 = torch.ops.aten.copy_.default(arg245_1, add_150);  arg245_1 = add_150 = None
        copy__85 = torch.ops.aten.copy_.default(arg246_1, add_151);  arg246_1 = add_151 = None
        copy__86 = torch.ops.aten.copy_.default(arg247_1, add_148);  arg247_1 = add_148 = None
        copy__87 = torch.ops.aten.copy_.default(arg248_1, add_155);  arg248_1 = add_155 = None
        copy__88 = torch.ops.aten.copy_.default(arg249_1, add_156);  arg249_1 = add_156 = None
        copy__89 = torch.ops.aten.copy_.default(arg250_1, add_153);  arg250_1 = add_153 = None
        copy__90 = torch.ops.aten.copy_.default(arg251_1, add_160);  arg251_1 = add_160 = None
        copy__91 = torch.ops.aten.copy_.default(arg252_1, add_161);  arg252_1 = add_161 = None
        copy__92 = torch.ops.aten.copy_.default(arg253_1, add_158);  arg253_1 = add_158 = None
        copy__93 = torch.ops.aten.copy_.default(arg254_1, add_166);  arg254_1 = add_166 = None
        copy__94 = torch.ops.aten.copy_.default(arg255_1, add_167);  arg255_1 = add_167 = None
        copy__95 = torch.ops.aten.copy_.default(arg256_1, add_164);  arg256_1 = add_164 = None
        copy__96 = torch.ops.aten.copy_.default(arg257_1, add_171);  arg257_1 = add_171 = None
        copy__97 = torch.ops.aten.copy_.default(arg258_1, add_172);  arg258_1 = add_172 = None
        copy__98 = torch.ops.aten.copy_.default(arg259_1, add_169);  arg259_1 = add_169 = None
        copy__99 = torch.ops.aten.copy_.default(arg260_1, add_176);  arg260_1 = add_176 = None
        copy__100 = torch.ops.aten.copy_.default(arg261_1, add_177);  arg261_1 = add_177 = None
        copy__101 = torch.ops.aten.copy_.default(arg262_1, add_174);  arg262_1 = add_174 = None
        copy__102 = torch.ops.aten.copy_.default(arg263_1, add_182);  arg263_1 = add_182 = None
        copy__103 = torch.ops.aten.copy_.default(arg264_1, add_183);  arg264_1 = add_183 = None
        copy__104 = torch.ops.aten.copy_.default(arg265_1, add_180);  arg265_1 = add_180 = None
        copy__105 = torch.ops.aten.copy_.default(arg266_1, add_187);  arg266_1 = add_187 = None
        copy__106 = torch.ops.aten.copy_.default(arg267_1, add_188);  arg267_1 = add_188 = None
        copy__107 = torch.ops.aten.copy_.default(arg268_1, add_185);  arg268_1 = add_185 = None
        copy__108 = torch.ops.aten.copy_.default(arg269_1, add_192);  arg269_1 = add_192 = None
        copy__109 = torch.ops.aten.copy_.default(arg270_1, add_193);  arg270_1 = add_193 = None
        copy__110 = torch.ops.aten.copy_.default(arg271_1, add_190);  arg271_1 = add_190 = None
        copy__111 = torch.ops.aten.copy_.default(arg272_1, add_198);  arg272_1 = add_198 = None
        copy__112 = torch.ops.aten.copy_.default(arg273_1, add_199);  arg273_1 = add_199 = None
        copy__113 = torch.ops.aten.copy_.default(arg274_1, add_196);  arg274_1 = add_196 = None
        copy__114 = torch.ops.aten.copy_.default(arg275_1, add_203);  arg275_1 = add_203 = None
        copy__115 = torch.ops.aten.copy_.default(arg276_1, add_204);  arg276_1 = add_204 = None
        copy__116 = torch.ops.aten.copy_.default(arg277_1, add_201);  arg277_1 = add_201 = None
        copy__117 = torch.ops.aten.copy_.default(arg278_1, add_208);  arg278_1 = add_208 = None
        copy__118 = torch.ops.aten.copy_.default(arg279_1, add_209);  arg279_1 = add_209 = None
        copy__119 = torch.ops.aten.copy_.default(arg280_1, add_206);  arg280_1 = add_206 = None
        copy__120 = torch.ops.aten.copy_.default(arg281_1, add_214);  arg281_1 = add_214 = None
        copy__121 = torch.ops.aten.copy_.default(arg282_1, add_215);  arg282_1 = add_215 = None
        copy__122 = torch.ops.aten.copy_.default(arg283_1, add_212);  arg283_1 = add_212 = None
        copy__123 = torch.ops.aten.copy_.default(arg284_1, add_219);  arg284_1 = add_219 = None
        copy__124 = torch.ops.aten.copy_.default(arg285_1, add_220);  arg285_1 = add_220 = None
        copy__125 = torch.ops.aten.copy_.default(arg286_1, add_217);  arg286_1 = add_217 = None
        copy__126 = torch.ops.aten.copy_.default(arg287_1, add_224);  arg287_1 = add_224 = None
        copy__127 = torch.ops.aten.copy_.default(arg288_1, add_225);  arg288_1 = add_225 = None
        copy__128 = torch.ops.aten.copy_.default(arg289_1, add_222);  arg289_1 = add_222 = None
        copy__129 = torch.ops.aten.copy_.default(arg290_1, add_230);  arg290_1 = add_230 = None
        copy__130 = torch.ops.aten.copy_.default(arg291_1, add_231);  arg291_1 = add_231 = None
        copy__131 = torch.ops.aten.copy_.default(arg292_1, add_228);  arg292_1 = add_228 = None
        copy__132 = torch.ops.aten.copy_.default(arg293_1, add_235);  arg293_1 = add_235 = None
        copy__133 = torch.ops.aten.copy_.default(arg294_1, add_236);  arg294_1 = add_236 = None
        copy__134 = torch.ops.aten.copy_.default(arg295_1, add_233);  arg295_1 = add_233 = None
        copy__135 = torch.ops.aten.copy_.default(arg296_1, add_240);  arg296_1 = add_240 = None
        copy__136 = torch.ops.aten.copy_.default(arg297_1, add_241);  arg297_1 = add_241 = None
        copy__137 = torch.ops.aten.copy_.default(arg298_1, add_238);  arg298_1 = add_238 = None
        copy__138 = torch.ops.aten.copy_.default(arg299_1, add_245);  arg299_1 = add_245 = None
        copy__139 = torch.ops.aten.copy_.default(arg300_1, add_246);  arg300_1 = add_246 = None
        copy__140 = torch.ops.aten.copy_.default(arg301_1, add_243);  arg301_1 = add_243 = None
        copy__141 = torch.ops.aten.copy_.default(arg302_1, add_251);  arg302_1 = add_251 = None
        copy__142 = torch.ops.aten.copy_.default(arg303_1, add_252);  arg303_1 = add_252 = None
        copy__143 = torch.ops.aten.copy_.default(arg304_1, add_249);  arg304_1 = add_249 = None
        copy__144 = torch.ops.aten.copy_.default(arg305_1, add_256);  arg305_1 = add_256 = None
        copy__145 = torch.ops.aten.copy_.default(arg306_1, add_257);  arg306_1 = add_257 = None
        copy__146 = torch.ops.aten.copy_.default(arg307_1, add_254);  arg307_1 = add_254 = None
        copy__147 = torch.ops.aten.copy_.default(arg308_1, add_261);  arg308_1 = add_261 = None
        copy__148 = torch.ops.aten.copy_.default(arg309_1, add_262);  arg309_1 = add_262 = None
        copy__149 = torch.ops.aten.copy_.default(arg310_1, add_259);  arg310_1 = add_259 = None
        copy__150 = torch.ops.aten.copy_.default(arg311_1, add_267);  arg311_1 = add_267 = None
        copy__151 = torch.ops.aten.copy_.default(arg312_1, add_268);  arg312_1 = add_268 = None
        copy__152 = torch.ops.aten.copy_.default(arg313_1, add_265);  arg313_1 = add_265 = None
        copy__153 = torch.ops.aten.copy_.default(arg314_1, add_272);  arg314_1 = add_272 = None
        copy__154 = torch.ops.aten.copy_.default(arg315_1, add_273);  arg315_1 = add_273 = None
        copy__155 = torch.ops.aten.copy_.default(arg316_1, add_270);  arg316_1 = add_270 = None
        copy__156 = torch.ops.aten.copy_.default(arg317_1, add_277);  arg317_1 = add_277 = None
        copy__157 = torch.ops.aten.copy_.default(arg318_1, add_278);  arg318_1 = add_278 = None
        copy__158 = torch.ops.aten.copy_.default(arg319_1, add_275);  arg319_1 = add_275 = None
        return (addmm,)
        
args = [((64, 3, 7, 7), (147, 49, 7, 1), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64, 64, 3, 3), (576, 9, 3, 1), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((256, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((64, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64, 64, 3, 3), (576, 9, 3, 1), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((256, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((64, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64, 64, 3, 3), (576, 9, 3, 1), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((256, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((128, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((512, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((128, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((512, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((128, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((512, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((128, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((512, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((256, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((512, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512, 3, 3), (4608, 9, 3, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((2048, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((2048, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((512, 2048, 1, 1), (2048, 1, 1, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512, 3, 3), (4608, 9, 3, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((2048, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((512, 2048, 1, 1), (2048, 1, 1, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512, 3, 3), (4608, 9, 3, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((2048, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((1000, 2048), (2048, 1), torch.float32, 'cpu'), ((1000,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((16, 3, 32, 32), (3072, 1024, 32, 1), torch.float32, 'cpu')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro(), tracing_mode='real')(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
ref = compiled(args)

