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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321):
        clone = torch.ops.aten.clone.default(primals_162);  primals_162 = None
        clone_1 = torch.ops.aten.clone.default(primals_163);  primals_163 = None
        clone_2 = torch.ops.aten.clone.default(primals_164);  primals_164 = None
        clone_3 = torch.ops.aten.clone.default(primals_165);  primals_165 = None
        clone_4 = torch.ops.aten.clone.default(primals_166);  primals_166 = None
        clone_5 = torch.ops.aten.clone.default(primals_167);  primals_167 = None
        clone_6 = torch.ops.aten.clone.default(primals_168);  primals_168 = None
        clone_7 = torch.ops.aten.clone.default(primals_169);  primals_169 = None
        clone_8 = torch.ops.aten.clone.default(primals_170);  primals_170 = None
        clone_9 = torch.ops.aten.clone.default(primals_171);  primals_171 = None
        clone_10 = torch.ops.aten.clone.default(primals_172);  primals_172 = None
        clone_11 = torch.ops.aten.clone.default(primals_173);  primals_173 = None
        clone_12 = torch.ops.aten.clone.default(primals_174);  primals_174 = None
        clone_13 = torch.ops.aten.clone.default(primals_175);  primals_175 = None
        clone_14 = torch.ops.aten.clone.default(primals_176);  primals_176 = None
        clone_15 = torch.ops.aten.clone.default(primals_177);  primals_177 = None
        clone_16 = torch.ops.aten.clone.default(primals_178);  primals_178 = None
        clone_17 = torch.ops.aten.clone.default(primals_179);  primals_179 = None
        clone_18 = torch.ops.aten.clone.default(primals_180);  primals_180 = None
        clone_19 = torch.ops.aten.clone.default(primals_181);  primals_181 = None
        clone_20 = torch.ops.aten.clone.default(primals_182);  primals_182 = None
        clone_21 = torch.ops.aten.clone.default(primals_183);  primals_183 = None
        clone_22 = torch.ops.aten.clone.default(primals_184);  primals_184 = None
        clone_23 = torch.ops.aten.clone.default(primals_185);  primals_185 = None
        clone_24 = torch.ops.aten.clone.default(primals_186);  primals_186 = None
        clone_25 = torch.ops.aten.clone.default(primals_187);  primals_187 = None
        clone_26 = torch.ops.aten.clone.default(primals_188);  primals_188 = None
        clone_27 = torch.ops.aten.clone.default(primals_189);  primals_189 = None
        clone_28 = torch.ops.aten.clone.default(primals_190);  primals_190 = None
        clone_29 = torch.ops.aten.clone.default(primals_191);  primals_191 = None
        clone_30 = torch.ops.aten.clone.default(primals_192);  primals_192 = None
        clone_31 = torch.ops.aten.clone.default(primals_193);  primals_193 = None
        clone_32 = torch.ops.aten.clone.default(primals_194);  primals_194 = None
        clone_33 = torch.ops.aten.clone.default(primals_195);  primals_195 = None
        clone_34 = torch.ops.aten.clone.default(primals_196);  primals_196 = None
        clone_35 = torch.ops.aten.clone.default(primals_197);  primals_197 = None
        clone_36 = torch.ops.aten.clone.default(primals_198);  primals_198 = None
        clone_37 = torch.ops.aten.clone.default(primals_199);  primals_199 = None
        clone_38 = torch.ops.aten.clone.default(primals_200);  primals_200 = None
        clone_39 = torch.ops.aten.clone.default(primals_201);  primals_201 = None
        clone_40 = torch.ops.aten.clone.default(primals_202);  primals_202 = None
        clone_41 = torch.ops.aten.clone.default(primals_203);  primals_203 = None
        clone_42 = torch.ops.aten.clone.default(primals_204);  primals_204 = None
        clone_43 = torch.ops.aten.clone.default(primals_205);  primals_205 = None
        clone_44 = torch.ops.aten.clone.default(primals_206);  primals_206 = None
        clone_45 = torch.ops.aten.clone.default(primals_207);  primals_207 = None
        clone_46 = torch.ops.aten.clone.default(primals_208);  primals_208 = None
        clone_47 = torch.ops.aten.clone.default(primals_209);  primals_209 = None
        clone_48 = torch.ops.aten.clone.default(primals_210);  primals_210 = None
        clone_49 = torch.ops.aten.clone.default(primals_211);  primals_211 = None
        clone_50 = torch.ops.aten.clone.default(primals_212);  primals_212 = None
        clone_51 = torch.ops.aten.clone.default(primals_213);  primals_213 = None
        clone_52 = torch.ops.aten.clone.default(primals_214);  primals_214 = None
        clone_53 = torch.ops.aten.clone.default(primals_215);  primals_215 = None
        clone_54 = torch.ops.aten.clone.default(primals_216);  primals_216 = None
        clone_55 = torch.ops.aten.clone.default(primals_217);  primals_217 = None
        clone_56 = torch.ops.aten.clone.default(primals_218);  primals_218 = None
        clone_57 = torch.ops.aten.clone.default(primals_219);  primals_219 = None
        clone_58 = torch.ops.aten.clone.default(primals_220);  primals_220 = None
        clone_59 = torch.ops.aten.clone.default(primals_221);  primals_221 = None
        clone_60 = torch.ops.aten.clone.default(primals_222);  primals_222 = None
        clone_61 = torch.ops.aten.clone.default(primals_223);  primals_223 = None
        clone_62 = torch.ops.aten.clone.default(primals_224);  primals_224 = None
        clone_63 = torch.ops.aten.clone.default(primals_225);  primals_225 = None
        clone_64 = torch.ops.aten.clone.default(primals_226);  primals_226 = None
        clone_65 = torch.ops.aten.clone.default(primals_227);  primals_227 = None
        clone_66 = torch.ops.aten.clone.default(primals_228);  primals_228 = None
        clone_67 = torch.ops.aten.clone.default(primals_229);  primals_229 = None
        clone_68 = torch.ops.aten.clone.default(primals_230);  primals_230 = None
        clone_69 = torch.ops.aten.clone.default(primals_231);  primals_231 = None
        clone_70 = torch.ops.aten.clone.default(primals_232);  primals_232 = None
        clone_71 = torch.ops.aten.clone.default(primals_233);  primals_233 = None
        clone_72 = torch.ops.aten.clone.default(primals_234);  primals_234 = None
        clone_73 = torch.ops.aten.clone.default(primals_235);  primals_235 = None
        clone_74 = torch.ops.aten.clone.default(primals_236);  primals_236 = None
        clone_75 = torch.ops.aten.clone.default(primals_237);  primals_237 = None
        clone_76 = torch.ops.aten.clone.default(primals_238);  primals_238 = None
        clone_77 = torch.ops.aten.clone.default(primals_239);  primals_239 = None
        clone_78 = torch.ops.aten.clone.default(primals_240);  primals_240 = None
        clone_79 = torch.ops.aten.clone.default(primals_241);  primals_241 = None
        clone_80 = torch.ops.aten.clone.default(primals_242);  primals_242 = None
        clone_81 = torch.ops.aten.clone.default(primals_243);  primals_243 = None
        clone_82 = torch.ops.aten.clone.default(primals_244);  primals_244 = None
        clone_83 = torch.ops.aten.clone.default(primals_245);  primals_245 = None
        clone_84 = torch.ops.aten.clone.default(primals_246);  primals_246 = None
        clone_85 = torch.ops.aten.clone.default(primals_247);  primals_247 = None
        clone_86 = torch.ops.aten.clone.default(primals_248);  primals_248 = None
        clone_87 = torch.ops.aten.clone.default(primals_249);  primals_249 = None
        clone_88 = torch.ops.aten.clone.default(primals_250);  primals_250 = None
        clone_89 = torch.ops.aten.clone.default(primals_251);  primals_251 = None
        clone_90 = torch.ops.aten.clone.default(primals_252);  primals_252 = None
        clone_91 = torch.ops.aten.clone.default(primals_253);  primals_253 = None
        clone_92 = torch.ops.aten.clone.default(primals_254);  primals_254 = None
        clone_93 = torch.ops.aten.clone.default(primals_255);  primals_255 = None
        clone_94 = torch.ops.aten.clone.default(primals_256);  primals_256 = None
        clone_95 = torch.ops.aten.clone.default(primals_257);  primals_257 = None
        clone_96 = torch.ops.aten.clone.default(primals_258);  primals_258 = None
        clone_97 = torch.ops.aten.clone.default(primals_259);  primals_259 = None
        clone_98 = torch.ops.aten.clone.default(primals_260);  primals_260 = None
        clone_99 = torch.ops.aten.clone.default(primals_261);  primals_261 = None
        clone_100 = torch.ops.aten.clone.default(primals_262);  primals_262 = None
        clone_101 = torch.ops.aten.clone.default(primals_263);  primals_263 = None
        clone_102 = torch.ops.aten.clone.default(primals_264);  primals_264 = None
        clone_103 = torch.ops.aten.clone.default(primals_265);  primals_265 = None
        clone_104 = torch.ops.aten.clone.default(primals_266);  primals_266 = None
        clone_105 = torch.ops.aten.clone.default(primals_267);  primals_267 = None
        clone_106 = torch.ops.aten.clone.default(primals_268);  primals_268 = None
        clone_107 = torch.ops.aten.clone.default(primals_269);  primals_269 = None
        clone_108 = torch.ops.aten.clone.default(primals_270);  primals_270 = None
        clone_109 = torch.ops.aten.clone.default(primals_271);  primals_271 = None
        clone_110 = torch.ops.aten.clone.default(primals_272);  primals_272 = None
        clone_111 = torch.ops.aten.clone.default(primals_273);  primals_273 = None
        clone_112 = torch.ops.aten.clone.default(primals_274);  primals_274 = None
        clone_113 = torch.ops.aten.clone.default(primals_275);  primals_275 = None
        clone_114 = torch.ops.aten.clone.default(primals_276);  primals_276 = None
        clone_115 = torch.ops.aten.clone.default(primals_277);  primals_277 = None
        clone_116 = torch.ops.aten.clone.default(primals_278);  primals_278 = None
        clone_117 = torch.ops.aten.clone.default(primals_279);  primals_279 = None
        clone_118 = torch.ops.aten.clone.default(primals_280);  primals_280 = None
        clone_119 = torch.ops.aten.clone.default(primals_281);  primals_281 = None
        clone_120 = torch.ops.aten.clone.default(primals_282);  primals_282 = None
        clone_121 = torch.ops.aten.clone.default(primals_283);  primals_283 = None
        clone_122 = torch.ops.aten.clone.default(primals_284);  primals_284 = None
        clone_123 = torch.ops.aten.clone.default(primals_285);  primals_285 = None
        clone_124 = torch.ops.aten.clone.default(primals_286);  primals_286 = None
        clone_125 = torch.ops.aten.clone.default(primals_287);  primals_287 = None
        clone_126 = torch.ops.aten.clone.default(primals_288);  primals_288 = None
        clone_127 = torch.ops.aten.clone.default(primals_289);  primals_289 = None
        clone_128 = torch.ops.aten.clone.default(primals_290);  primals_290 = None
        clone_129 = torch.ops.aten.clone.default(primals_291);  primals_291 = None
        clone_130 = torch.ops.aten.clone.default(primals_292);  primals_292 = None
        clone_131 = torch.ops.aten.clone.default(primals_293);  primals_293 = None
        clone_132 = torch.ops.aten.clone.default(primals_294);  primals_294 = None
        clone_133 = torch.ops.aten.clone.default(primals_295);  primals_295 = None
        clone_134 = torch.ops.aten.clone.default(primals_296);  primals_296 = None
        clone_135 = torch.ops.aten.clone.default(primals_297);  primals_297 = None
        clone_136 = torch.ops.aten.clone.default(primals_298);  primals_298 = None
        clone_137 = torch.ops.aten.clone.default(primals_299);  primals_299 = None
        clone_138 = torch.ops.aten.clone.default(primals_300);  primals_300 = None
        clone_139 = torch.ops.aten.clone.default(primals_301);  primals_301 = None
        clone_140 = torch.ops.aten.clone.default(primals_302);  primals_302 = None
        clone_141 = torch.ops.aten.clone.default(primals_303);  primals_303 = None
        clone_142 = torch.ops.aten.clone.default(primals_304);  primals_304 = None
        clone_143 = torch.ops.aten.clone.default(primals_305);  primals_305 = None
        clone_144 = torch.ops.aten.clone.default(primals_306);  primals_306 = None
        clone_145 = torch.ops.aten.clone.default(primals_307);  primals_307 = None
        clone_146 = torch.ops.aten.clone.default(primals_308);  primals_308 = None
        clone_147 = torch.ops.aten.clone.default(primals_309);  primals_309 = None
        clone_148 = torch.ops.aten.clone.default(primals_310);  primals_310 = None
        clone_149 = torch.ops.aten.clone.default(primals_311);  primals_311 = None
        clone_150 = torch.ops.aten.clone.default(primals_312);  primals_312 = None
        clone_151 = torch.ops.aten.clone.default(primals_313);  primals_313 = None
        clone_152 = torch.ops.aten.clone.default(primals_314);  primals_314 = None
        clone_153 = torch.ops.aten.clone.default(primals_315);  primals_315 = None
        clone_154 = torch.ops.aten.clone.default(primals_316);  primals_316 = None
        clone_155 = torch.ops.aten.clone.default(primals_317);  primals_317 = None
        clone_156 = torch.ops.aten.clone.default(primals_318);  primals_318 = None
        clone_157 = torch.ops.aten.clone.default(primals_319);  primals_319 = None
        clone_158 = torch.ops.aten.clone.default(primals_320);  primals_320 = None
        convolution = torch.ops.aten.convolution.default(primals_321, primals_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)
        add = torch.ops.aten.add.Tensor(clone_2, 1);  clone_2 = None
        var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_1 = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_1 = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(squeeze, 0.1)
        mul_2 = torch.ops.aten.mul.Tensor(clone, 0.9);  clone = None
        add_2 = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2 = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3 = torch.ops.aten.mul.Tensor(squeeze_2, 1.0004885197850513);  squeeze_2 = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5 = torch.ops.aten.mul.Tensor(clone_1, 0.9);  clone_1 = None
        add_3 = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_2, -1)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(primals_3, -1);  primals_3 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_6 = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        add_4 = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        relu = torch.ops.aten.relu.default(add_4);  add_4 = None
        max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu, [3, 3], [2, 2], [1, 1])
        getitem_2 = max_pool2d_with_indices[0]
        getitem_3 = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
        convolution_1 = torch.ops.aten.convolution.default(getitem_2, primals_4, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_5 = torch.ops.aten.add.Tensor(clone_5, 1);  clone_5 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
        getitem_4 = var_mean_1[0]
        getitem_5 = var_mean_1[1];  var_mean_1 = None
        add_6 = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1 = torch.ops.aten.sub.Tensor(convolution_1, getitem_5)
        mul_7 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3 = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_4 = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_8 = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
        mul_9 = torch.ops.aten.mul.Tensor(clone_3, 0.9);  clone_3 = None
        add_7 = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5 = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_10 = torch.ops.aten.mul.Tensor(squeeze_5, 1.0019569471624266);  squeeze_5 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12 = torch.ops.aten.mul.Tensor(clone_4, 0.9);  clone_4 = None
        add_8 = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(primals_5, -1)
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        add_9 = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        relu_1 = torch.ops.aten.relu.default(add_9);  add_9 = None
        convolution_2 = torch.ops.aten.convolution.default(relu_1, primals_7, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_10 = torch.ops.aten.add.Tensor(clone_8, 1);  clone_8 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
        getitem_6 = var_mean_2[0]
        getitem_7 = var_mean_2[1];  var_mean_2 = None
        add_11 = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_2 = torch.ops.aten.sub.Tensor(convolution_2, getitem_7)
        mul_14 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6 = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_7 = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_15 = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
        mul_16 = torch.ops.aten.mul.Tensor(clone_6, 0.9);  clone_6 = None
        add_12 = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        squeeze_8 = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_17 = torch.ops.aten.mul.Tensor(squeeze_8, 1.0019569471624266);  squeeze_8 = None
        mul_18 = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19 = torch.ops.aten.mul.Tensor(clone_7, 0.9);  clone_7 = None
        add_13 = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(primals_8, -1)
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(primals_9, -1);  primals_9 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        add_14 = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
        relu_2 = torch.ops.aten.relu.default(add_14);  add_14 = None
        convolution_3 = torch.ops.aten.convolution.default(relu_2, primals_10, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_15 = torch.ops.aten.add.Tensor(clone_11, 1);  clone_11 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
        getitem_8 = var_mean_3[0]
        getitem_9 = var_mean_3[1];  var_mean_3 = None
        add_16 = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_3 = torch.ops.aten.sub.Tensor(convolution_3, getitem_9)
        mul_21 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9 = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_10 = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_22 = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
        mul_23 = torch.ops.aten.mul.Tensor(clone_9, 0.9);  clone_9 = None
        add_17 = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        squeeze_11 = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_24 = torch.ops.aten.mul.Tensor(squeeze_11, 1.0019569471624266);  squeeze_11 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
        mul_26 = torch.ops.aten.mul.Tensor(clone_10, 0.9);  clone_10 = None
        add_18 = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(primals_11, -1)
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        add_19 = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
        convolution_4 = torch.ops.aten.convolution.default(getitem_2, primals_13, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_20 = torch.ops.aten.add.Tensor(clone_14, 1);  clone_14 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
        getitem_10 = var_mean_4[0]
        getitem_11 = var_mean_4[1];  var_mean_4 = None
        add_21 = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_4 = torch.ops.aten.sub.Tensor(convolution_4, getitem_11)
        mul_28 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12 = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_13 = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_29 = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
        mul_30 = torch.ops.aten.mul.Tensor(clone_12, 0.9);  clone_12 = None
        add_22 = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
        squeeze_14 = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_31 = torch.ops.aten.mul.Tensor(squeeze_14, 1.0019569471624266);  squeeze_14 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
        mul_33 = torch.ops.aten.mul.Tensor(clone_13, 0.9);  clone_13 = None
        add_23 = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(primals_14, -1)
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(primals_15, -1);  primals_15 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        add_24 = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
        add_25 = torch.ops.aten.add.Tensor(add_19, add_24);  add_19 = add_24 = None
        relu_3 = torch.ops.aten.relu.default(add_25);  add_25 = None
        convolution_5 = torch.ops.aten.convolution.default(relu_3, primals_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_26 = torch.ops.aten.add.Tensor(clone_17, 1);  clone_17 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
        getitem_12 = var_mean_5[0]
        getitem_13 = var_mean_5[1];  var_mean_5 = None
        add_27 = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        sub_5 = torch.ops.aten.sub.Tensor(convolution_5, getitem_13)
        mul_35 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15 = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_16 = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_36 = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
        mul_37 = torch.ops.aten.mul.Tensor(clone_15, 0.9);  clone_15 = None
        add_28 = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        squeeze_17 = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_38 = torch.ops.aten.mul.Tensor(squeeze_17, 1.0019569471624266);  squeeze_17 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
        mul_40 = torch.ops.aten.mul.Tensor(clone_16, 0.9);  clone_16 = None
        add_29 = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(primals_17, -1)
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        add_30 = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
        relu_4 = torch.ops.aten.relu.default(add_30);  add_30 = None
        convolution_6 = torch.ops.aten.convolution.default(relu_4, primals_19, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_31 = torch.ops.aten.add.Tensor(clone_20, 1);  clone_20 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
        getitem_14 = var_mean_6[0]
        getitem_15 = var_mean_6[1];  var_mean_6 = None
        add_32 = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_6 = torch.ops.aten.sub.Tensor(convolution_6, getitem_15)
        mul_42 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_18 = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_19 = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_43 = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
        mul_44 = torch.ops.aten.mul.Tensor(clone_18, 0.9);  clone_18 = None
        add_33 = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
        squeeze_20 = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_45 = torch.ops.aten.mul.Tensor(squeeze_20, 1.0019569471624266);  squeeze_20 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
        mul_47 = torch.ops.aten.mul.Tensor(clone_19, 0.9);  clone_19 = None
        add_34 = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(primals_20, -1)
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(primals_21, -1);  primals_21 = None
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
        add_35 = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
        relu_5 = torch.ops.aten.relu.default(add_35);  add_35 = None
        convolution_7 = torch.ops.aten.convolution.default(relu_5, primals_22, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_36 = torch.ops.aten.add.Tensor(clone_23, 1);  clone_23 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
        getitem_16 = var_mean_7[0]
        getitem_17 = var_mean_7[1];  var_mean_7 = None
        add_37 = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_7 = torch.ops.aten.sub.Tensor(convolution_7, getitem_17)
        mul_49 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_21 = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_22 = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_50 = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
        mul_51 = torch.ops.aten.mul.Tensor(clone_21, 0.9);  clone_21 = None
        add_38 = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
        squeeze_23 = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_52 = torch.ops.aten.mul.Tensor(squeeze_23, 1.0019569471624266);  squeeze_23 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
        mul_54 = torch.ops.aten.mul.Tensor(clone_22, 0.9);  clone_22 = None
        add_39 = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(primals_23, -1)
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
        add_40 = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
        add_41 = torch.ops.aten.add.Tensor(add_40, relu_3);  add_40 = None
        relu_6 = torch.ops.aten.relu.default(add_41);  add_41 = None
        convolution_8 = torch.ops.aten.convolution.default(relu_6, primals_25, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_42 = torch.ops.aten.add.Tensor(clone_26, 1);  clone_26 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
        getitem_18 = var_mean_8[0]
        getitem_19 = var_mean_8[1];  var_mean_8 = None
        add_43 = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_8 = torch.ops.aten.sub.Tensor(convolution_8, getitem_19)
        mul_56 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_24 = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_25 = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_57 = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
        mul_58 = torch.ops.aten.mul.Tensor(clone_24, 0.9);  clone_24 = None
        add_44 = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
        squeeze_26 = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_59 = torch.ops.aten.mul.Tensor(squeeze_26, 1.0019569471624266);  squeeze_26 = None
        mul_60 = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
        mul_61 = torch.ops.aten.mul.Tensor(clone_25, 0.9);  clone_25 = None
        add_45 = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(primals_26, -1)
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(primals_27, -1);  primals_27 = None
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        mul_62 = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
        add_46 = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
        relu_7 = torch.ops.aten.relu.default(add_46);  add_46 = None
        convolution_9 = torch.ops.aten.convolution.default(relu_7, primals_28, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_47 = torch.ops.aten.add.Tensor(clone_29, 1);  clone_29 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
        getitem_20 = var_mean_9[0]
        getitem_21 = var_mean_9[1];  var_mean_9 = None
        add_48 = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_9 = torch.ops.aten.sub.Tensor(convolution_9, getitem_21)
        mul_63 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_27 = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_28 = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_64 = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
        mul_65 = torch.ops.aten.mul.Tensor(clone_27, 0.9);  clone_27 = None
        add_49 = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
        squeeze_29 = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_66 = torch.ops.aten.mul.Tensor(squeeze_29, 1.0019569471624266);  squeeze_29 = None
        mul_67 = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
        mul_68 = torch.ops.aten.mul.Tensor(clone_28, 0.9);  clone_28 = None
        add_50 = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(primals_29, -1)
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
        add_51 = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
        relu_8 = torch.ops.aten.relu.default(add_51);  add_51 = None
        convolution_10 = torch.ops.aten.convolution.default(relu_8, primals_31, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_52 = torch.ops.aten.add.Tensor(clone_32, 1);  clone_32 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
        getitem_22 = var_mean_10[0]
        getitem_23 = var_mean_10[1];  var_mean_10 = None
        add_53 = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_10 = torch.ops.aten.sub.Tensor(convolution_10, getitem_23)
        mul_70 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        squeeze_30 = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_31 = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_71 = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
        mul_72 = torch.ops.aten.mul.Tensor(clone_30, 0.9);  clone_30 = None
        add_54 = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
        squeeze_32 = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_73 = torch.ops.aten.mul.Tensor(squeeze_32, 1.0019569471624266);  squeeze_32 = None
        mul_74 = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
        mul_75 = torch.ops.aten.mul.Tensor(clone_31, 0.9);  clone_31 = None
        add_55 = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        unsqueeze_40 = torch.ops.aten.unsqueeze.default(primals_32, -1)
        unsqueeze_41 = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        unsqueeze_42 = torch.ops.aten.unsqueeze.default(primals_33, -1);  primals_33 = None
        unsqueeze_43 = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        mul_76 = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
        add_56 = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
        add_57 = torch.ops.aten.add.Tensor(add_56, relu_6);  add_56 = None
        relu_9 = torch.ops.aten.relu.default(add_57);  add_57 = None
        convolution_11 = torch.ops.aten.convolution.default(relu_9, primals_34, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_58 = torch.ops.aten.add.Tensor(clone_35, 1);  clone_35 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
        getitem_24 = var_mean_11[0]
        getitem_25 = var_mean_11[1];  var_mean_11 = None
        add_59 = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        sub_11 = torch.ops.aten.sub.Tensor(convolution_11, getitem_25)
        mul_77 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        squeeze_33 = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_34 = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_78 = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
        mul_79 = torch.ops.aten.mul.Tensor(clone_33, 0.9);  clone_33 = None
        add_60 = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
        squeeze_35 = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_80 = torch.ops.aten.mul.Tensor(squeeze_35, 1.0019569471624266);  squeeze_35 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
        mul_82 = torch.ops.aten.mul.Tensor(clone_34, 0.9);  clone_34 = None
        add_61 = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
        unsqueeze_44 = torch.ops.aten.unsqueeze.default(primals_35, -1)
        unsqueeze_45 = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        unsqueeze_46 = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
        unsqueeze_47 = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        mul_83 = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
        add_62 = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
        relu_10 = torch.ops.aten.relu.default(add_62);  add_62 = None
        convolution_12 = torch.ops.aten.convolution.default(relu_10, primals_37, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add_63 = torch.ops.aten.add.Tensor(clone_38, 1);  clone_38 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
        getitem_26 = var_mean_12[0]
        getitem_27 = var_mean_12[1];  var_mean_12 = None
        add_64 = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_12 = torch.ops.aten.sub.Tensor(convolution_12, getitem_27)
        mul_84 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        squeeze_36 = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        squeeze_37 = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_85 = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
        mul_86 = torch.ops.aten.mul.Tensor(clone_36, 0.9);  clone_36 = None
        add_65 = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
        squeeze_38 = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_87 = torch.ops.aten.mul.Tensor(squeeze_38, 1.0078740157480315);  squeeze_38 = None
        mul_88 = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
        mul_89 = torch.ops.aten.mul.Tensor(clone_37, 0.9);  clone_37 = None
        add_66 = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
        unsqueeze_48 = torch.ops.aten.unsqueeze.default(primals_38, -1)
        unsqueeze_49 = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        unsqueeze_50 = torch.ops.aten.unsqueeze.default(primals_39, -1);  primals_39 = None
        unsqueeze_51 = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        mul_90 = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
        add_67 = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
        relu_11 = torch.ops.aten.relu.default(add_67);  add_67 = None
        convolution_13 = torch.ops.aten.convolution.default(relu_11, primals_40, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_68 = torch.ops.aten.add.Tensor(clone_41, 1);  clone_41 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
        getitem_28 = var_mean_13[0]
        getitem_29 = var_mean_13[1];  var_mean_13 = None
        add_69 = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_13 = torch.ops.aten.sub.Tensor(convolution_13, getitem_29)
        mul_91 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        squeeze_39 = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_40 = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_92 = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
        mul_93 = torch.ops.aten.mul.Tensor(clone_39, 0.9);  clone_39 = None
        add_70 = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
        squeeze_41 = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_94 = torch.ops.aten.mul.Tensor(squeeze_41, 1.0078740157480315);  squeeze_41 = None
        mul_95 = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
        mul_96 = torch.ops.aten.mul.Tensor(clone_40, 0.9);  clone_40 = None
        add_71 = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
        unsqueeze_52 = torch.ops.aten.unsqueeze.default(primals_41, -1)
        unsqueeze_53 = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        unsqueeze_54 = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
        unsqueeze_55 = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
        add_72 = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
        convolution_14 = torch.ops.aten.convolution.default(relu_9, primals_43, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_73 = torch.ops.aten.add.Tensor(clone_44, 1);  clone_44 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
        getitem_30 = var_mean_14[0]
        getitem_31 = var_mean_14[1];  var_mean_14 = None
        add_74 = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_14 = torch.ops.aten.sub.Tensor(convolution_14, getitem_31)
        mul_98 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        squeeze_42 = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_43 = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_99 = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
        mul_100 = torch.ops.aten.mul.Tensor(clone_42, 0.9);  clone_42 = None
        add_75 = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
        squeeze_44 = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_101 = torch.ops.aten.mul.Tensor(squeeze_44, 1.0078740157480315);  squeeze_44 = None
        mul_102 = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
        mul_103 = torch.ops.aten.mul.Tensor(clone_43, 0.9);  clone_43 = None
        add_76 = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
        unsqueeze_56 = torch.ops.aten.unsqueeze.default(primals_44, -1)
        unsqueeze_57 = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        unsqueeze_58 = torch.ops.aten.unsqueeze.default(primals_45, -1);  primals_45 = None
        unsqueeze_59 = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        mul_104 = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
        add_77 = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
        add_78 = torch.ops.aten.add.Tensor(add_72, add_77);  add_72 = add_77 = None
        relu_12 = torch.ops.aten.relu.default(add_78);  add_78 = None
        convolution_15 = torch.ops.aten.convolution.default(relu_12, primals_46, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_79 = torch.ops.aten.add.Tensor(clone_47, 1);  clone_47 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
        getitem_32 = var_mean_15[0]
        getitem_33 = var_mean_15[1];  var_mean_15 = None
        add_80 = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_15 = torch.ops.aten.sub.Tensor(convolution_15, getitem_33)
        mul_105 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        squeeze_45 = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_46 = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_106 = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
        mul_107 = torch.ops.aten.mul.Tensor(clone_45, 0.9);  clone_45 = None
        add_81 = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
        squeeze_47 = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_108 = torch.ops.aten.mul.Tensor(squeeze_47, 1.0078740157480315);  squeeze_47 = None
        mul_109 = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
        mul_110 = torch.ops.aten.mul.Tensor(clone_46, 0.9);  clone_46 = None
        add_82 = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
        unsqueeze_60 = torch.ops.aten.unsqueeze.default(primals_47, -1)
        unsqueeze_61 = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        unsqueeze_62 = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
        unsqueeze_63 = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        mul_111 = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
        add_83 = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
        relu_13 = torch.ops.aten.relu.default(add_83);  add_83 = None
        convolution_16 = torch.ops.aten.convolution.default(relu_13, primals_49, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_84 = torch.ops.aten.add.Tensor(clone_50, 1);  clone_50 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
        getitem_34 = var_mean_16[0]
        getitem_35 = var_mean_16[1];  var_mean_16 = None
        add_85 = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_16 = torch.ops.aten.sub.Tensor(convolution_16, getitem_35)
        mul_112 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        squeeze_48 = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_49 = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_113 = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
        mul_114 = torch.ops.aten.mul.Tensor(clone_48, 0.9);  clone_48 = None
        add_86 = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
        squeeze_50 = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_115 = torch.ops.aten.mul.Tensor(squeeze_50, 1.0078740157480315);  squeeze_50 = None
        mul_116 = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
        mul_117 = torch.ops.aten.mul.Tensor(clone_49, 0.9);  clone_49 = None
        add_87 = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        unsqueeze_64 = torch.ops.aten.unsqueeze.default(primals_50, -1)
        unsqueeze_65 = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        unsqueeze_66 = torch.ops.aten.unsqueeze.default(primals_51, -1);  primals_51 = None
        unsqueeze_67 = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        mul_118 = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
        add_88 = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
        relu_14 = torch.ops.aten.relu.default(add_88);  add_88 = None
        convolution_17 = torch.ops.aten.convolution.default(relu_14, primals_52, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_89 = torch.ops.aten.add.Tensor(clone_53, 1);  clone_53 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
        getitem_36 = var_mean_17[0]
        getitem_37 = var_mean_17[1];  var_mean_17 = None
        add_90 = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_17 = torch.ops.aten.sub.Tensor(convolution_17, getitem_37)
        mul_119 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        squeeze_51 = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_52 = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_120 = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
        mul_121 = torch.ops.aten.mul.Tensor(clone_51, 0.9);  clone_51 = None
        add_91 = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
        squeeze_53 = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_122 = torch.ops.aten.mul.Tensor(squeeze_53, 1.0078740157480315);  squeeze_53 = None
        mul_123 = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
        mul_124 = torch.ops.aten.mul.Tensor(clone_52, 0.9);  clone_52 = None
        add_92 = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
        unsqueeze_68 = torch.ops.aten.unsqueeze.default(primals_53, -1)
        unsqueeze_69 = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        unsqueeze_70 = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
        unsqueeze_71 = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        mul_125 = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
        add_93 = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
        add_94 = torch.ops.aten.add.Tensor(add_93, relu_12);  add_93 = None
        relu_15 = torch.ops.aten.relu.default(add_94);  add_94 = None
        convolution_18 = torch.ops.aten.convolution.default(relu_15, primals_55, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_95 = torch.ops.aten.add.Tensor(clone_56, 1);  clone_56 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
        getitem_38 = var_mean_18[0]
        getitem_39 = var_mean_18[1];  var_mean_18 = None
        add_96 = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_18 = torch.ops.aten.sub.Tensor(convolution_18, getitem_39)
        mul_126 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        squeeze_54 = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        squeeze_55 = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_127 = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
        mul_128 = torch.ops.aten.mul.Tensor(clone_54, 0.9);  clone_54 = None
        add_97 = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
        squeeze_56 = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_129 = torch.ops.aten.mul.Tensor(squeeze_56, 1.0078740157480315);  squeeze_56 = None
        mul_130 = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
        mul_131 = torch.ops.aten.mul.Tensor(clone_55, 0.9);  clone_55 = None
        add_98 = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
        unsqueeze_72 = torch.ops.aten.unsqueeze.default(primals_56, -1)
        unsqueeze_73 = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        unsqueeze_74 = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
        unsqueeze_75 = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        mul_132 = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
        add_99 = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
        relu_16 = torch.ops.aten.relu.default(add_99);  add_99 = None
        convolution_19 = torch.ops.aten.convolution.default(relu_16, primals_58, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_100 = torch.ops.aten.add.Tensor(clone_59, 1);  clone_59 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
        getitem_40 = var_mean_19[0]
        getitem_41 = var_mean_19[1];  var_mean_19 = None
        add_101 = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        sub_19 = torch.ops.aten.sub.Tensor(convolution_19, getitem_41)
        mul_133 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        squeeze_57 = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_58 = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_134 = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
        mul_135 = torch.ops.aten.mul.Tensor(clone_57, 0.9);  clone_57 = None
        add_102 = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
        squeeze_59 = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_136 = torch.ops.aten.mul.Tensor(squeeze_59, 1.0078740157480315);  squeeze_59 = None
        mul_137 = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
        mul_138 = torch.ops.aten.mul.Tensor(clone_58, 0.9);  clone_58 = None
        add_103 = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
        unsqueeze_76 = torch.ops.aten.unsqueeze.default(primals_59, -1)
        unsqueeze_77 = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        unsqueeze_78 = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
        unsqueeze_79 = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        mul_139 = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
        add_104 = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
        relu_17 = torch.ops.aten.relu.default(add_104);  add_104 = None
        convolution_20 = torch.ops.aten.convolution.default(relu_17, primals_61, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_105 = torch.ops.aten.add.Tensor(clone_62, 1);  clone_62 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
        getitem_42 = var_mean_20[0]
        getitem_43 = var_mean_20[1];  var_mean_20 = None
        add_106 = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_20 = torch.ops.aten.sub.Tensor(convolution_20, getitem_43)
        mul_140 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        squeeze_60 = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        squeeze_61 = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_141 = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
        mul_142 = torch.ops.aten.mul.Tensor(clone_60, 0.9);  clone_60 = None
        add_107 = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
        squeeze_62 = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_143 = torch.ops.aten.mul.Tensor(squeeze_62, 1.0078740157480315);  squeeze_62 = None
        mul_144 = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
        mul_145 = torch.ops.aten.mul.Tensor(clone_61, 0.9);  clone_61 = None
        add_108 = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
        unsqueeze_80 = torch.ops.aten.unsqueeze.default(primals_62, -1)
        unsqueeze_81 = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        unsqueeze_82 = torch.ops.aten.unsqueeze.default(primals_63, -1);  primals_63 = None
        unsqueeze_83 = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        mul_146 = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_81);  mul_140 = unsqueeze_81 = None
        add_109 = torch.ops.aten.add.Tensor(mul_146, unsqueeze_83);  mul_146 = unsqueeze_83 = None
        add_110 = torch.ops.aten.add.Tensor(add_109, relu_15);  add_109 = None
        relu_18 = torch.ops.aten.relu.default(add_110);  add_110 = None
        convolution_21 = torch.ops.aten.convolution.default(relu_18, primals_64, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_111 = torch.ops.aten.add.Tensor(clone_65, 1);  clone_65 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
        getitem_44 = var_mean_21[0]
        getitem_45 = var_mean_21[1];  var_mean_21 = None
        add_112 = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        sub_21 = torch.ops.aten.sub.Tensor(convolution_21, getitem_45)
        mul_147 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        squeeze_63 = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        squeeze_64 = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_148 = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
        mul_149 = torch.ops.aten.mul.Tensor(clone_63, 0.9);  clone_63 = None
        add_113 = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
        squeeze_65 = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        mul_150 = torch.ops.aten.mul.Tensor(squeeze_65, 1.0078740157480315);  squeeze_65 = None
        mul_151 = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
        mul_152 = torch.ops.aten.mul.Tensor(clone_64, 0.9);  clone_64 = None
        add_114 = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
        unsqueeze_84 = torch.ops.aten.unsqueeze.default(primals_65, -1)
        unsqueeze_85 = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        unsqueeze_86 = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
        unsqueeze_87 = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        mul_153 = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
        add_115 = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None
        relu_19 = torch.ops.aten.relu.default(add_115);  add_115 = None
        convolution_22 = torch.ops.aten.convolution.default(relu_19, primals_67, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_116 = torch.ops.aten.add.Tensor(clone_68, 1);  clone_68 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
        getitem_46 = var_mean_22[0]
        getitem_47 = var_mean_22[1];  var_mean_22 = None
        add_117 = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
        sub_22 = torch.ops.aten.sub.Tensor(convolution_22, getitem_47)
        mul_154 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        squeeze_66 = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        squeeze_67 = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_155 = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
        mul_156 = torch.ops.aten.mul.Tensor(clone_66, 0.9);  clone_66 = None
        add_118 = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
        squeeze_68 = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_157 = torch.ops.aten.mul.Tensor(squeeze_68, 1.0078740157480315);  squeeze_68 = None
        mul_158 = torch.ops.aten.mul.Tensor(mul_157, 0.1);  mul_157 = None
        mul_159 = torch.ops.aten.mul.Tensor(clone_67, 0.9);  clone_67 = None
        add_119 = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
        unsqueeze_88 = torch.ops.aten.unsqueeze.default(primals_68, -1)
        unsqueeze_89 = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        unsqueeze_90 = torch.ops.aten.unsqueeze.default(primals_69, -1);  primals_69 = None
        unsqueeze_91 = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        mul_160 = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
        add_120 = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None
        relu_20 = torch.ops.aten.relu.default(add_120);  add_120 = None
        convolution_23 = torch.ops.aten.convolution.default(relu_20, primals_70, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_121 = torch.ops.aten.add.Tensor(clone_71, 1);  clone_71 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
        getitem_48 = var_mean_23[0]
        getitem_49 = var_mean_23[1];  var_mean_23 = None
        add_122 = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        sub_23 = torch.ops.aten.sub.Tensor(convolution_23, getitem_49)
        mul_161 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        squeeze_69 = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        squeeze_70 = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_162 = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
        mul_163 = torch.ops.aten.mul.Tensor(clone_69, 0.9);  clone_69 = None
        add_123 = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
        squeeze_71 = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        mul_164 = torch.ops.aten.mul.Tensor(squeeze_71, 1.0078740157480315);  squeeze_71 = None
        mul_165 = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
        mul_166 = torch.ops.aten.mul.Tensor(clone_70, 0.9);  clone_70 = None
        add_124 = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
        unsqueeze_92 = torch.ops.aten.unsqueeze.default(primals_71, -1)
        unsqueeze_93 = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        unsqueeze_94 = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
        unsqueeze_95 = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        mul_167 = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_93);  mul_161 = unsqueeze_93 = None
        add_125 = torch.ops.aten.add.Tensor(mul_167, unsqueeze_95);  mul_167 = unsqueeze_95 = None
        add_126 = torch.ops.aten.add.Tensor(add_125, relu_18);  add_125 = None
        relu_21 = torch.ops.aten.relu.default(add_126);  add_126 = None
        convolution_24 = torch.ops.aten.convolution.default(relu_21, primals_73, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_127 = torch.ops.aten.add.Tensor(clone_74, 1);  clone_74 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
        getitem_50 = var_mean_24[0]
        getitem_51 = var_mean_24[1];  var_mean_24 = None
        add_128 = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        sub_24 = torch.ops.aten.sub.Tensor(convolution_24, getitem_51)
        mul_168 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        squeeze_72 = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        squeeze_73 = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_169 = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
        mul_170 = torch.ops.aten.mul.Tensor(clone_72, 0.9);  clone_72 = None
        add_129 = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
        squeeze_74 = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_171 = torch.ops.aten.mul.Tensor(squeeze_74, 1.0078740157480315);  squeeze_74 = None
        mul_172 = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
        mul_173 = torch.ops.aten.mul.Tensor(clone_73, 0.9);  clone_73 = None
        add_130 = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
        unsqueeze_96 = torch.ops.aten.unsqueeze.default(primals_74, -1)
        unsqueeze_97 = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        unsqueeze_98 = torch.ops.aten.unsqueeze.default(primals_75, -1);  primals_75 = None
        unsqueeze_99 = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        mul_174 = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_97);  mul_168 = unsqueeze_97 = None
        add_131 = torch.ops.aten.add.Tensor(mul_174, unsqueeze_99);  mul_174 = unsqueeze_99 = None
        relu_22 = torch.ops.aten.relu.default(add_131);  add_131 = None
        convolution_25 = torch.ops.aten.convolution.default(relu_22, primals_76, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add_132 = torch.ops.aten.add.Tensor(clone_77, 1);  clone_77 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
        getitem_52 = var_mean_25[0]
        getitem_53 = var_mean_25[1];  var_mean_25 = None
        add_133 = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        sub_25 = torch.ops.aten.sub.Tensor(convolution_25, getitem_53)
        mul_175 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        squeeze_75 = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        squeeze_76 = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_176 = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
        mul_177 = torch.ops.aten.mul.Tensor(clone_75, 0.9);  clone_75 = None
        add_134 = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
        squeeze_77 = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
        mul_178 = torch.ops.aten.mul.Tensor(squeeze_77, 1.032258064516129);  squeeze_77 = None
        mul_179 = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
        mul_180 = torch.ops.aten.mul.Tensor(clone_76, 0.9);  clone_76 = None
        add_135 = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
        unsqueeze_100 = torch.ops.aten.unsqueeze.default(primals_77, -1)
        unsqueeze_101 = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        unsqueeze_102 = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
        unsqueeze_103 = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        mul_181 = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
        add_136 = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None
        relu_23 = torch.ops.aten.relu.default(add_136);  add_136 = None
        convolution_26 = torch.ops.aten.convolution.default(relu_23, primals_79, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_137 = torch.ops.aten.add.Tensor(clone_80, 1);  clone_80 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
        getitem_54 = var_mean_26[0]
        getitem_55 = var_mean_26[1];  var_mean_26 = None
        add_138 = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        sub_26 = torch.ops.aten.sub.Tensor(convolution_26, getitem_55)
        mul_182 = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        squeeze_78 = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        squeeze_79 = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_183 = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
        mul_184 = torch.ops.aten.mul.Tensor(clone_78, 0.9);  clone_78 = None
        add_139 = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
        squeeze_80 = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
        mul_185 = torch.ops.aten.mul.Tensor(squeeze_80, 1.032258064516129);  squeeze_80 = None
        mul_186 = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
        mul_187 = torch.ops.aten.mul.Tensor(clone_79, 0.9);  clone_79 = None
        add_140 = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
        unsqueeze_104 = torch.ops.aten.unsqueeze.default(primals_80, -1)
        unsqueeze_105 = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        unsqueeze_106 = torch.ops.aten.unsqueeze.default(primals_81, -1);  primals_81 = None
        unsqueeze_107 = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        mul_188 = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_105);  mul_182 = unsqueeze_105 = None
        add_141 = torch.ops.aten.add.Tensor(mul_188, unsqueeze_107);  mul_188 = unsqueeze_107 = None
        convolution_27 = torch.ops.aten.convolution.default(relu_21, primals_82, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_142 = torch.ops.aten.add.Tensor(clone_83, 1);  clone_83 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
        getitem_56 = var_mean_27[0]
        getitem_57 = var_mean_27[1];  var_mean_27 = None
        add_143 = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
        sub_27 = torch.ops.aten.sub.Tensor(convolution_27, getitem_57)
        mul_189 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        squeeze_81 = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        squeeze_82 = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_190 = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
        mul_191 = torch.ops.aten.mul.Tensor(clone_81, 0.9);  clone_81 = None
        add_144 = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
        squeeze_83 = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        mul_192 = torch.ops.aten.mul.Tensor(squeeze_83, 1.032258064516129);  squeeze_83 = None
        mul_193 = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
        mul_194 = torch.ops.aten.mul.Tensor(clone_82, 0.9);  clone_82 = None
        add_145 = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
        unsqueeze_108 = torch.ops.aten.unsqueeze.default(primals_83, -1)
        unsqueeze_109 = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        unsqueeze_110 = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
        unsqueeze_111 = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        mul_195 = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_109);  mul_189 = unsqueeze_109 = None
        add_146 = torch.ops.aten.add.Tensor(mul_195, unsqueeze_111);  mul_195 = unsqueeze_111 = None
        add_147 = torch.ops.aten.add.Tensor(add_141, add_146);  add_141 = add_146 = None
        relu_24 = torch.ops.aten.relu.default(add_147);  add_147 = None
        convolution_28 = torch.ops.aten.convolution.default(relu_24, primals_85, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_148 = torch.ops.aten.add.Tensor(clone_86, 1);  clone_86 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
        getitem_58 = var_mean_28[0]
        getitem_59 = var_mean_28[1];  var_mean_28 = None
        add_149 = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        sub_28 = torch.ops.aten.sub.Tensor(convolution_28, getitem_59)
        mul_196 = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        squeeze_84 = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        squeeze_85 = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_197 = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
        mul_198 = torch.ops.aten.mul.Tensor(clone_84, 0.9);  clone_84 = None
        add_150 = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
        squeeze_86 = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        mul_199 = torch.ops.aten.mul.Tensor(squeeze_86, 1.032258064516129);  squeeze_86 = None
        mul_200 = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
        mul_201 = torch.ops.aten.mul.Tensor(clone_85, 0.9);  clone_85 = None
        add_151 = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
        unsqueeze_112 = torch.ops.aten.unsqueeze.default(primals_86, -1)
        unsqueeze_113 = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        unsqueeze_114 = torch.ops.aten.unsqueeze.default(primals_87, -1);  primals_87 = None
        unsqueeze_115 = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        mul_202 = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
        add_152 = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None
        relu_25 = torch.ops.aten.relu.default(add_152);  add_152 = None
        convolution_29 = torch.ops.aten.convolution.default(relu_25, primals_88, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_153 = torch.ops.aten.add.Tensor(clone_89, 1);  clone_89 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
        getitem_60 = var_mean_29[0]
        getitem_61 = var_mean_29[1];  var_mean_29 = None
        add_154 = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        sub_29 = torch.ops.aten.sub.Tensor(convolution_29, getitem_61)
        mul_203 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        squeeze_87 = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        squeeze_88 = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
        mul_204 = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
        mul_205 = torch.ops.aten.mul.Tensor(clone_87, 0.9);  clone_87 = None
        add_155 = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
        squeeze_89 = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_206 = torch.ops.aten.mul.Tensor(squeeze_89, 1.032258064516129);  squeeze_89 = None
        mul_207 = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
        mul_208 = torch.ops.aten.mul.Tensor(clone_88, 0.9);  clone_88 = None
        add_156 = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
        unsqueeze_116 = torch.ops.aten.unsqueeze.default(primals_89, -1)
        unsqueeze_117 = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        unsqueeze_118 = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
        unsqueeze_119 = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        mul_209 = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_117);  mul_203 = unsqueeze_117 = None
        add_157 = torch.ops.aten.add.Tensor(mul_209, unsqueeze_119);  mul_209 = unsqueeze_119 = None
        relu_26 = torch.ops.aten.relu.default(add_157);  add_157 = None
        convolution_30 = torch.ops.aten.convolution.default(relu_26, primals_91, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_158 = torch.ops.aten.add.Tensor(clone_92, 1);  clone_92 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
        getitem_62 = var_mean_30[0]
        getitem_63 = var_mean_30[1];  var_mean_30 = None
        add_159 = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        sub_30 = torch.ops.aten.sub.Tensor(convolution_30, getitem_63)
        mul_210 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        squeeze_90 = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        squeeze_91 = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_211 = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
        mul_212 = torch.ops.aten.mul.Tensor(clone_90, 0.9);  clone_90 = None
        add_160 = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
        squeeze_92 = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_213 = torch.ops.aten.mul.Tensor(squeeze_92, 1.032258064516129);  squeeze_92 = None
        mul_214 = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
        mul_215 = torch.ops.aten.mul.Tensor(clone_91, 0.9);  clone_91 = None
        add_161 = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
        unsqueeze_120 = torch.ops.aten.unsqueeze.default(primals_92, -1)
        unsqueeze_121 = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        unsqueeze_122 = torch.ops.aten.unsqueeze.default(primals_93, -1);  primals_93 = None
        unsqueeze_123 = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        mul_216 = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_121);  mul_210 = unsqueeze_121 = None
        add_162 = torch.ops.aten.add.Tensor(mul_216, unsqueeze_123);  mul_216 = unsqueeze_123 = None
        add_163 = torch.ops.aten.add.Tensor(add_162, relu_24);  add_162 = None
        relu_27 = torch.ops.aten.relu.default(add_163);  add_163 = None
        convolution_31 = torch.ops.aten.convolution.default(relu_27, primals_94, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_164 = torch.ops.aten.add.Tensor(clone_95, 1);  clone_95 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
        getitem_64 = var_mean_31[0]
        getitem_65 = var_mean_31[1];  var_mean_31 = None
        add_165 = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        sub_31 = torch.ops.aten.sub.Tensor(convolution_31, getitem_65)
        mul_217 = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        squeeze_93 = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        squeeze_94 = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_218 = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
        mul_219 = torch.ops.aten.mul.Tensor(clone_93, 0.9);  clone_93 = None
        add_166 = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
        squeeze_95 = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        mul_220 = torch.ops.aten.mul.Tensor(squeeze_95, 1.032258064516129);  squeeze_95 = None
        mul_221 = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
        mul_222 = torch.ops.aten.mul.Tensor(clone_94, 0.9);  clone_94 = None
        add_167 = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
        unsqueeze_124 = torch.ops.aten.unsqueeze.default(primals_95, -1)
        unsqueeze_125 = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        unsqueeze_126 = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
        unsqueeze_127 = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        mul_223 = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
        add_168 = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None
        relu_28 = torch.ops.aten.relu.default(add_168);  add_168 = None
        convolution_32 = torch.ops.aten.convolution.default(relu_28, primals_97, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_169 = torch.ops.aten.add.Tensor(clone_98, 1);  clone_98 = None
        var_mean_32 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
        getitem_66 = var_mean_32[0]
        getitem_67 = var_mean_32[1];  var_mean_32 = None
        add_170 = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        sub_32 = torch.ops.aten.sub.Tensor(convolution_32, getitem_67)
        mul_224 = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
        squeeze_96 = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        squeeze_97 = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
        mul_225 = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
        mul_226 = torch.ops.aten.mul.Tensor(clone_96, 0.9);  clone_96 = None
        add_171 = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
        squeeze_98 = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_227 = torch.ops.aten.mul.Tensor(squeeze_98, 1.032258064516129);  squeeze_98 = None
        mul_228 = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
        mul_229 = torch.ops.aten.mul.Tensor(clone_97, 0.9);  clone_97 = None
        add_172 = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
        unsqueeze_128 = torch.ops.aten.unsqueeze.default(primals_98, -1)
        unsqueeze_129 = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        unsqueeze_130 = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
        unsqueeze_131 = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        mul_230 = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_129);  mul_224 = unsqueeze_129 = None
        add_173 = torch.ops.aten.add.Tensor(mul_230, unsqueeze_131);  mul_230 = unsqueeze_131 = None
        relu_29 = torch.ops.aten.relu.default(add_173);  add_173 = None
        convolution_33 = torch.ops.aten.convolution.default(relu_29, primals_100, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_174 = torch.ops.aten.add.Tensor(clone_101, 1);  clone_101 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
        getitem_68 = var_mean_33[0]
        getitem_69 = var_mean_33[1];  var_mean_33 = None
        add_175 = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
        rsqrt_33 = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
        sub_33 = torch.ops.aten.sub.Tensor(convolution_33, getitem_69)
        mul_231 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        squeeze_99 = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        squeeze_100 = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_232 = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
        mul_233 = torch.ops.aten.mul.Tensor(clone_99, 0.9);  clone_99 = None
        add_176 = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
        squeeze_101 = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_234 = torch.ops.aten.mul.Tensor(squeeze_101, 1.032258064516129);  squeeze_101 = None
        mul_235 = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
        mul_236 = torch.ops.aten.mul.Tensor(clone_100, 0.9);  clone_100 = None
        add_177 = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
        unsqueeze_132 = torch.ops.aten.unsqueeze.default(primals_101, -1)
        unsqueeze_133 = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        unsqueeze_134 = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
        unsqueeze_135 = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        mul_237 = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_133);  mul_231 = unsqueeze_133 = None
        add_178 = torch.ops.aten.add.Tensor(mul_237, unsqueeze_135);  mul_237 = unsqueeze_135 = None
        add_179 = torch.ops.aten.add.Tensor(add_178, relu_27);  add_178 = None
        relu_30 = torch.ops.aten.relu.default(add_179);  add_179 = None
        convolution_34 = torch.ops.aten.convolution.default(relu_30, primals_103, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_180 = torch.ops.aten.add.Tensor(clone_104, 1);  clone_104 = None
        var_mean_34 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
        getitem_70 = var_mean_34[0]
        getitem_71 = var_mean_34[1];  var_mean_34 = None
        add_181 = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
        rsqrt_34 = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_34 = torch.ops.aten.sub.Tensor(convolution_34, getitem_71)
        mul_238 = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        squeeze_102 = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        squeeze_103 = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_239 = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
        mul_240 = torch.ops.aten.mul.Tensor(clone_102, 0.9);  clone_102 = None
        add_182 = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
        squeeze_104 = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        mul_241 = torch.ops.aten.mul.Tensor(squeeze_104, 1.032258064516129);  squeeze_104 = None
        mul_242 = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
        mul_243 = torch.ops.aten.mul.Tensor(clone_103, 0.9);  clone_103 = None
        add_183 = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
        unsqueeze_136 = torch.ops.aten.unsqueeze.default(primals_104, -1)
        unsqueeze_137 = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        unsqueeze_138 = torch.ops.aten.unsqueeze.default(primals_105, -1);  primals_105 = None
        unsqueeze_139 = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        mul_244 = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
        add_184 = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None
        relu_31 = torch.ops.aten.relu.default(add_184);  add_184 = None
        convolution_35 = torch.ops.aten.convolution.default(relu_31, primals_106, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_185 = torch.ops.aten.add.Tensor(clone_107, 1);  clone_107 = None
        var_mean_35 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
        getitem_72 = var_mean_35[0]
        getitem_73 = var_mean_35[1];  var_mean_35 = None
        add_186 = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
        rsqrt_35 = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        sub_35 = torch.ops.aten.sub.Tensor(convolution_35, getitem_73)
        mul_245 = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
        squeeze_105 = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        squeeze_106 = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
        mul_246 = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
        mul_247 = torch.ops.aten.mul.Tensor(clone_105, 0.9);  clone_105 = None
        add_187 = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
        squeeze_107 = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        mul_248 = torch.ops.aten.mul.Tensor(squeeze_107, 1.032258064516129);  squeeze_107 = None
        mul_249 = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
        mul_250 = torch.ops.aten.mul.Tensor(clone_106, 0.9);  clone_106 = None
        add_188 = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
        unsqueeze_140 = torch.ops.aten.unsqueeze.default(primals_107, -1)
        unsqueeze_141 = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        unsqueeze_142 = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
        unsqueeze_143 = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        mul_251 = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_141);  mul_245 = unsqueeze_141 = None
        add_189 = torch.ops.aten.add.Tensor(mul_251, unsqueeze_143);  mul_251 = unsqueeze_143 = None
        relu_32 = torch.ops.aten.relu.default(add_189);  add_189 = None
        convolution_36 = torch.ops.aten.convolution.default(relu_32, primals_109, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_190 = torch.ops.aten.add.Tensor(clone_110, 1);  clone_110 = None
        var_mean_36 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
        getitem_74 = var_mean_36[0]
        getitem_75 = var_mean_36[1];  var_mean_36 = None
        add_191 = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
        rsqrt_36 = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
        sub_36 = torch.ops.aten.sub.Tensor(convolution_36, getitem_75)
        mul_252 = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        squeeze_108 = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        squeeze_109 = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
        mul_253 = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
        mul_254 = torch.ops.aten.mul.Tensor(clone_108, 0.9);  clone_108 = None
        add_192 = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
        squeeze_110 = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
        mul_255 = torch.ops.aten.mul.Tensor(squeeze_110, 1.032258064516129);  squeeze_110 = None
        mul_256 = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
        mul_257 = torch.ops.aten.mul.Tensor(clone_109, 0.9);  clone_109 = None
        add_193 = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
        unsqueeze_144 = torch.ops.aten.unsqueeze.default(primals_110, -1)
        unsqueeze_145 = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        unsqueeze_146 = torch.ops.aten.unsqueeze.default(primals_111, -1);  primals_111 = None
        unsqueeze_147 = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        mul_258 = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_145);  mul_252 = unsqueeze_145 = None
        add_194 = torch.ops.aten.add.Tensor(mul_258, unsqueeze_147);  mul_258 = unsqueeze_147 = None
        add_195 = torch.ops.aten.add.Tensor(add_194, relu_30);  add_194 = None
        relu_33 = torch.ops.aten.relu.default(add_195);  add_195 = None
        convolution_37 = torch.ops.aten.convolution.default(relu_33, primals_112, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_196 = torch.ops.aten.add.Tensor(clone_113, 1);  clone_113 = None
        var_mean_37 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
        getitem_76 = var_mean_37[0]
        getitem_77 = var_mean_37[1];  var_mean_37 = None
        add_197 = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
        rsqrt_37 = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
        sub_37 = torch.ops.aten.sub.Tensor(convolution_37, getitem_77)
        mul_259 = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        squeeze_111 = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_112 = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_260 = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
        mul_261 = torch.ops.aten.mul.Tensor(clone_111, 0.9);  clone_111 = None
        add_198 = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
        squeeze_113 = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_262 = torch.ops.aten.mul.Tensor(squeeze_113, 1.032258064516129);  squeeze_113 = None
        mul_263 = torch.ops.aten.mul.Tensor(mul_262, 0.1);  mul_262 = None
        mul_264 = torch.ops.aten.mul.Tensor(clone_112, 0.9);  clone_112 = None
        add_199 = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
        unsqueeze_148 = torch.ops.aten.unsqueeze.default(primals_113, -1)
        unsqueeze_149 = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        unsqueeze_150 = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
        unsqueeze_151 = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        mul_265 = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
        add_200 = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None
        relu_34 = torch.ops.aten.relu.default(add_200);  add_200 = None
        convolution_38 = torch.ops.aten.convolution.default(relu_34, primals_115, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_201 = torch.ops.aten.add.Tensor(clone_116, 1);  clone_116 = None
        var_mean_38 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
        getitem_78 = var_mean_38[0]
        getitem_79 = var_mean_38[1];  var_mean_38 = None
        add_202 = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
        rsqrt_38 = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
        sub_38 = torch.ops.aten.sub.Tensor(convolution_38, getitem_79)
        mul_266 = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        squeeze_114 = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        squeeze_115 = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_267 = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
        mul_268 = torch.ops.aten.mul.Tensor(clone_114, 0.9);  clone_114 = None
        add_203 = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
        squeeze_116 = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_269 = torch.ops.aten.mul.Tensor(squeeze_116, 1.032258064516129);  squeeze_116 = None
        mul_270 = torch.ops.aten.mul.Tensor(mul_269, 0.1);  mul_269 = None
        mul_271 = torch.ops.aten.mul.Tensor(clone_115, 0.9);  clone_115 = None
        add_204 = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
        unsqueeze_152 = torch.ops.aten.unsqueeze.default(primals_116, -1)
        unsqueeze_153 = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        unsqueeze_154 = torch.ops.aten.unsqueeze.default(primals_117, -1);  primals_117 = None
        unsqueeze_155 = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        mul_272 = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_153);  mul_266 = unsqueeze_153 = None
        add_205 = torch.ops.aten.add.Tensor(mul_272, unsqueeze_155);  mul_272 = unsqueeze_155 = None
        relu_35 = torch.ops.aten.relu.default(add_205);  add_205 = None
        convolution_39 = torch.ops.aten.convolution.default(relu_35, primals_118, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_206 = torch.ops.aten.add.Tensor(clone_119, 1);  clone_119 = None
        var_mean_39 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
        getitem_80 = var_mean_39[0]
        getitem_81 = var_mean_39[1];  var_mean_39 = None
        add_207 = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
        rsqrt_39 = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
        sub_39 = torch.ops.aten.sub.Tensor(convolution_39, getitem_81)
        mul_273 = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        squeeze_117 = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        squeeze_118 = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
        mul_274 = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
        mul_275 = torch.ops.aten.mul.Tensor(clone_117, 0.9);  clone_117 = None
        add_208 = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
        squeeze_119 = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
        mul_276 = torch.ops.aten.mul.Tensor(squeeze_119, 1.032258064516129);  squeeze_119 = None
        mul_277 = torch.ops.aten.mul.Tensor(mul_276, 0.1);  mul_276 = None
        mul_278 = torch.ops.aten.mul.Tensor(clone_118, 0.9);  clone_118 = None
        add_209 = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
        unsqueeze_156 = torch.ops.aten.unsqueeze.default(primals_119, -1)
        unsqueeze_157 = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        unsqueeze_158 = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
        unsqueeze_159 = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        mul_279 = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_157);  mul_273 = unsqueeze_157 = None
        add_210 = torch.ops.aten.add.Tensor(mul_279, unsqueeze_159);  mul_279 = unsqueeze_159 = None
        add_211 = torch.ops.aten.add.Tensor(add_210, relu_33);  add_210 = None
        relu_36 = torch.ops.aten.relu.default(add_211);  add_211 = None
        convolution_40 = torch.ops.aten.convolution.default(relu_36, primals_121, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_212 = torch.ops.aten.add.Tensor(clone_122, 1);  clone_122 = None
        var_mean_40 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
        getitem_82 = var_mean_40[0]
        getitem_83 = var_mean_40[1];  var_mean_40 = None
        add_213 = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
        rsqrt_40 = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
        sub_40 = torch.ops.aten.sub.Tensor(convolution_40, getitem_83)
        mul_280 = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        squeeze_120 = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
        squeeze_121 = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_281 = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
        mul_282 = torch.ops.aten.mul.Tensor(clone_120, 0.9);  clone_120 = None
        add_214 = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
        squeeze_122 = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
        mul_283 = torch.ops.aten.mul.Tensor(squeeze_122, 1.032258064516129);  squeeze_122 = None
        mul_284 = torch.ops.aten.mul.Tensor(mul_283, 0.1);  mul_283 = None
        mul_285 = torch.ops.aten.mul.Tensor(clone_121, 0.9);  clone_121 = None
        add_215 = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
        unsqueeze_160 = torch.ops.aten.unsqueeze.default(primals_122, -1)
        unsqueeze_161 = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        unsqueeze_162 = torch.ops.aten.unsqueeze.default(primals_123, -1);  primals_123 = None
        unsqueeze_163 = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        mul_286 = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
        add_216 = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None
        relu_37 = torch.ops.aten.relu.default(add_216);  add_216 = None
        convolution_41 = torch.ops.aten.convolution.default(relu_37, primals_124, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_217 = torch.ops.aten.add.Tensor(clone_125, 1);  clone_125 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
        getitem_84 = var_mean_41[0]
        getitem_85 = var_mean_41[1];  var_mean_41 = None
        add_218 = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
        sub_41 = torch.ops.aten.sub.Tensor(convolution_41, getitem_85)
        mul_287 = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
        squeeze_123 = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
        squeeze_124 = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_288 = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
        mul_289 = torch.ops.aten.mul.Tensor(clone_123, 0.9);  clone_123 = None
        add_219 = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
        squeeze_125 = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
        mul_290 = torch.ops.aten.mul.Tensor(squeeze_125, 1.032258064516129);  squeeze_125 = None
        mul_291 = torch.ops.aten.mul.Tensor(mul_290, 0.1);  mul_290 = None
        mul_292 = torch.ops.aten.mul.Tensor(clone_124, 0.9);  clone_124 = None
        add_220 = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
        unsqueeze_164 = torch.ops.aten.unsqueeze.default(primals_125, -1)
        unsqueeze_165 = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        unsqueeze_166 = torch.ops.aten.unsqueeze.default(primals_126, -1);  primals_126 = None
        unsqueeze_167 = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        mul_293 = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_165);  mul_287 = unsqueeze_165 = None
        add_221 = torch.ops.aten.add.Tensor(mul_293, unsqueeze_167);  mul_293 = unsqueeze_167 = None
        relu_38 = torch.ops.aten.relu.default(add_221);  add_221 = None
        convolution_42 = torch.ops.aten.convolution.default(relu_38, primals_127, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_222 = torch.ops.aten.add.Tensor(clone_128, 1);  clone_128 = None
        var_mean_42 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
        getitem_86 = var_mean_42[0]
        getitem_87 = var_mean_42[1];  var_mean_42 = None
        add_223 = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
        rsqrt_42 = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
        sub_42 = torch.ops.aten.sub.Tensor(convolution_42, getitem_87)
        mul_294 = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        squeeze_126 = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        squeeze_127 = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_295 = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
        mul_296 = torch.ops.aten.mul.Tensor(clone_126, 0.9);  clone_126 = None
        add_224 = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
        squeeze_128 = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
        mul_297 = torch.ops.aten.mul.Tensor(squeeze_128, 1.032258064516129);  squeeze_128 = None
        mul_298 = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
        mul_299 = torch.ops.aten.mul.Tensor(clone_127, 0.9);  clone_127 = None
        add_225 = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
        unsqueeze_168 = torch.ops.aten.unsqueeze.default(primals_128, -1)
        unsqueeze_169 = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        unsqueeze_170 = torch.ops.aten.unsqueeze.default(primals_129, -1);  primals_129 = None
        unsqueeze_171 = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        mul_300 = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_169);  mul_294 = unsqueeze_169 = None
        add_226 = torch.ops.aten.add.Tensor(mul_300, unsqueeze_171);  mul_300 = unsqueeze_171 = None
        add_227 = torch.ops.aten.add.Tensor(add_226, relu_36);  add_226 = None
        relu_39 = torch.ops.aten.relu.default(add_227);  add_227 = None
        convolution_43 = torch.ops.aten.convolution.default(relu_39, primals_130, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_228 = torch.ops.aten.add.Tensor(clone_131, 1);  clone_131 = None
        var_mean_43 = torch.ops.aten.var_mean.correction(convolution_43, [0, 2, 3], correction = 0, keepdim = True)
        getitem_88 = var_mean_43[0]
        getitem_89 = var_mean_43[1];  var_mean_43 = None
        add_229 = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
        rsqrt_43 = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        sub_43 = torch.ops.aten.sub.Tensor(convolution_43, getitem_89)
        mul_301 = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        squeeze_129 = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
        squeeze_130 = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
        mul_302 = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
        mul_303 = torch.ops.aten.mul.Tensor(clone_129, 0.9);  clone_129 = None
        add_230 = torch.ops.aten.add.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
        squeeze_131 = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
        mul_304 = torch.ops.aten.mul.Tensor(squeeze_131, 1.032258064516129);  squeeze_131 = None
        mul_305 = torch.ops.aten.mul.Tensor(mul_304, 0.1);  mul_304 = None
        mul_306 = torch.ops.aten.mul.Tensor(clone_130, 0.9);  clone_130 = None
        add_231 = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
        unsqueeze_172 = torch.ops.aten.unsqueeze.default(primals_131, -1)
        unsqueeze_173 = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        unsqueeze_174 = torch.ops.aten.unsqueeze.default(primals_132, -1);  primals_132 = None
        unsqueeze_175 = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        mul_307 = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_173);  mul_301 = unsqueeze_173 = None
        add_232 = torch.ops.aten.add.Tensor(mul_307, unsqueeze_175);  mul_307 = unsqueeze_175 = None
        relu_40 = torch.ops.aten.relu.default(add_232);  add_232 = None
        convolution_44 = torch.ops.aten.convolution.default(relu_40, primals_133, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add_233 = torch.ops.aten.add.Tensor(clone_134, 1);  clone_134 = None
        var_mean_44 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
        getitem_90 = var_mean_44[0]
        getitem_91 = var_mean_44[1];  var_mean_44 = None
        add_234 = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
        rsqrt_44 = torch.ops.aten.rsqrt.default(add_234);  add_234 = None
        sub_44 = torch.ops.aten.sub.Tensor(convolution_44, getitem_91)
        mul_308 = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = rsqrt_44 = None
        squeeze_132 = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
        mul_309 = torch.ops.aten.mul.Tensor(squeeze_132, 0.1);  squeeze_132 = None
        mul_310 = torch.ops.aten.mul.Tensor(clone_132, 0.9);  clone_132 = None
        add_235 = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
        squeeze_134 = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
        mul_311 = torch.ops.aten.mul.Tensor(squeeze_134, 1.1428571428571428);  squeeze_134 = None
        mul_312 = torch.ops.aten.mul.Tensor(mul_311, 0.1);  mul_311 = None
        mul_313 = torch.ops.aten.mul.Tensor(clone_133, 0.9);  clone_133 = None
        add_236 = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
        unsqueeze_176 = torch.ops.aten.unsqueeze.default(primals_134, -1)
        unsqueeze_177 = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        unsqueeze_178 = torch.ops.aten.unsqueeze.default(primals_135, -1);  primals_135 = None
        unsqueeze_179 = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
        mul_314 = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_177);  mul_308 = unsqueeze_177 = None
        add_237 = torch.ops.aten.add.Tensor(mul_314, unsqueeze_179);  mul_314 = unsqueeze_179 = None
        relu_41 = torch.ops.aten.relu.default(add_237);  add_237 = None
        convolution_45 = torch.ops.aten.convolution.default(relu_41, primals_136, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_238 = torch.ops.aten.add.Tensor(clone_137, 1);  clone_137 = None
        var_mean_45 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
        getitem_92 = var_mean_45[0]
        getitem_93 = var_mean_45[1];  var_mean_45 = None
        add_239 = torch.ops.aten.add.Tensor(getitem_92, 1e-05)
        rsqrt_45 = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
        sub_45 = torch.ops.aten.sub.Tensor(convolution_45, getitem_93)
        mul_315 = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = rsqrt_45 = None
        squeeze_135 = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
        mul_316 = torch.ops.aten.mul.Tensor(squeeze_135, 0.1);  squeeze_135 = None
        mul_317 = torch.ops.aten.mul.Tensor(clone_135, 0.9);  clone_135 = None
        add_240 = torch.ops.aten.add.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
        squeeze_137 = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
        mul_318 = torch.ops.aten.mul.Tensor(squeeze_137, 1.1428571428571428);  squeeze_137 = None
        mul_319 = torch.ops.aten.mul.Tensor(mul_318, 0.1);  mul_318 = None
        mul_320 = torch.ops.aten.mul.Tensor(clone_136, 0.9);  clone_136 = None
        add_241 = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
        unsqueeze_180 = torch.ops.aten.unsqueeze.default(primals_137, -1)
        unsqueeze_181 = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        unsqueeze_182 = torch.ops.aten.unsqueeze.default(primals_138, -1);  primals_138 = None
        unsqueeze_183 = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        mul_321 = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_181);  mul_315 = unsqueeze_181 = None
        add_242 = torch.ops.aten.add.Tensor(mul_321, unsqueeze_183);  mul_321 = unsqueeze_183 = None
        convolution_46 = torch.ops.aten.convolution.default(relu_39, primals_139, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_243 = torch.ops.aten.add.Tensor(clone_140, 1);  clone_140 = None
        var_mean_46 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
        getitem_94 = var_mean_46[0]
        getitem_95 = var_mean_46[1];  var_mean_46 = None
        add_244 = torch.ops.aten.add.Tensor(getitem_94, 1e-05)
        rsqrt_46 = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
        sub_46 = torch.ops.aten.sub.Tensor(convolution_46, getitem_95)
        mul_322 = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = rsqrt_46 = None
        squeeze_138 = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
        mul_323 = torch.ops.aten.mul.Tensor(squeeze_138, 0.1);  squeeze_138 = None
        mul_324 = torch.ops.aten.mul.Tensor(clone_138, 0.9);  clone_138 = None
        add_245 = torch.ops.aten.add.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
        squeeze_140 = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
        mul_325 = torch.ops.aten.mul.Tensor(squeeze_140, 1.1428571428571428);  squeeze_140 = None
        mul_326 = torch.ops.aten.mul.Tensor(mul_325, 0.1);  mul_325 = None
        mul_327 = torch.ops.aten.mul.Tensor(clone_139, 0.9);  clone_139 = None
        add_246 = torch.ops.aten.add.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
        unsqueeze_184 = torch.ops.aten.unsqueeze.default(primals_140, -1)
        unsqueeze_185 = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        unsqueeze_186 = torch.ops.aten.unsqueeze.default(primals_141, -1);  primals_141 = None
        unsqueeze_187 = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        mul_328 = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_185);  mul_322 = unsqueeze_185 = None
        add_247 = torch.ops.aten.add.Tensor(mul_328, unsqueeze_187);  mul_328 = unsqueeze_187 = None
        add_248 = torch.ops.aten.add.Tensor(add_242, add_247);  add_242 = add_247 = None
        relu_42 = torch.ops.aten.relu.default(add_248);  add_248 = None
        convolution_47 = torch.ops.aten.convolution.default(relu_42, primals_142, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_249 = torch.ops.aten.add.Tensor(clone_143, 1);  clone_143 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
        getitem_96 = var_mean_47[0]
        getitem_97 = var_mean_47[1];  var_mean_47 = None
        add_250 = torch.ops.aten.add.Tensor(getitem_96, 1e-05)
        rsqrt_47 = torch.ops.aten.rsqrt.default(add_250);  add_250 = None
        sub_47 = torch.ops.aten.sub.Tensor(convolution_47, getitem_97)
        mul_329 = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = rsqrt_47 = None
        squeeze_141 = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
        mul_330 = torch.ops.aten.mul.Tensor(squeeze_141, 0.1);  squeeze_141 = None
        mul_331 = torch.ops.aten.mul.Tensor(clone_141, 0.9);  clone_141 = None
        add_251 = torch.ops.aten.add.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
        squeeze_143 = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
        mul_332 = torch.ops.aten.mul.Tensor(squeeze_143, 1.1428571428571428);  squeeze_143 = None
        mul_333 = torch.ops.aten.mul.Tensor(mul_332, 0.1);  mul_332 = None
        mul_334 = torch.ops.aten.mul.Tensor(clone_142, 0.9);  clone_142 = None
        add_252 = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
        unsqueeze_188 = torch.ops.aten.unsqueeze.default(primals_143, -1)
        unsqueeze_189 = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        unsqueeze_190 = torch.ops.aten.unsqueeze.default(primals_144, -1);  primals_144 = None
        unsqueeze_191 = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        mul_335 = torch.ops.aten.mul.Tensor(mul_329, unsqueeze_189);  mul_329 = unsqueeze_189 = None
        add_253 = torch.ops.aten.add.Tensor(mul_335, unsqueeze_191);  mul_335 = unsqueeze_191 = None
        relu_43 = torch.ops.aten.relu.default(add_253);  add_253 = None
        convolution_48 = torch.ops.aten.convolution.default(relu_43, primals_145, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_254 = torch.ops.aten.add.Tensor(clone_146, 1);  clone_146 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
        getitem_98 = var_mean_48[0]
        getitem_99 = var_mean_48[1];  var_mean_48 = None
        add_255 = torch.ops.aten.add.Tensor(getitem_98, 1e-05)
        rsqrt_48 = torch.ops.aten.rsqrt.default(add_255);  add_255 = None
        sub_48 = torch.ops.aten.sub.Tensor(convolution_48, getitem_99)
        mul_336 = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = rsqrt_48 = None
        squeeze_144 = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
        mul_337 = torch.ops.aten.mul.Tensor(squeeze_144, 0.1);  squeeze_144 = None
        mul_338 = torch.ops.aten.mul.Tensor(clone_144, 0.9);  clone_144 = None
        add_256 = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
        squeeze_146 = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
        mul_339 = torch.ops.aten.mul.Tensor(squeeze_146, 1.1428571428571428);  squeeze_146 = None
        mul_340 = torch.ops.aten.mul.Tensor(mul_339, 0.1);  mul_339 = None
        mul_341 = torch.ops.aten.mul.Tensor(clone_145, 0.9);  clone_145 = None
        add_257 = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
        unsqueeze_192 = torch.ops.aten.unsqueeze.default(primals_146, -1)
        unsqueeze_193 = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        unsqueeze_194 = torch.ops.aten.unsqueeze.default(primals_147, -1);  primals_147 = None
        unsqueeze_195 = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        mul_342 = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_193);  mul_336 = unsqueeze_193 = None
        add_258 = torch.ops.aten.add.Tensor(mul_342, unsqueeze_195);  mul_342 = unsqueeze_195 = None
        relu_44 = torch.ops.aten.relu.default(add_258);  add_258 = None
        convolution_49 = torch.ops.aten.convolution.default(relu_44, primals_148, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_259 = torch.ops.aten.add.Tensor(clone_149, 1);  clone_149 = None
        var_mean_49 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
        getitem_100 = var_mean_49[0]
        getitem_101 = var_mean_49[1];  var_mean_49 = None
        add_260 = torch.ops.aten.add.Tensor(getitem_100, 1e-05)
        rsqrt_49 = torch.ops.aten.rsqrt.default(add_260);  add_260 = None
        sub_49 = torch.ops.aten.sub.Tensor(convolution_49, getitem_101)
        mul_343 = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = rsqrt_49 = None
        squeeze_147 = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
        mul_344 = torch.ops.aten.mul.Tensor(squeeze_147, 0.1);  squeeze_147 = None
        mul_345 = torch.ops.aten.mul.Tensor(clone_147, 0.9);  clone_147 = None
        add_261 = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
        squeeze_149 = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
        mul_346 = torch.ops.aten.mul.Tensor(squeeze_149, 1.1428571428571428);  squeeze_149 = None
        mul_347 = torch.ops.aten.mul.Tensor(mul_346, 0.1);  mul_346 = None
        mul_348 = torch.ops.aten.mul.Tensor(clone_148, 0.9);  clone_148 = None
        add_262 = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
        unsqueeze_196 = torch.ops.aten.unsqueeze.default(primals_149, -1)
        unsqueeze_197 = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        unsqueeze_198 = torch.ops.aten.unsqueeze.default(primals_150, -1);  primals_150 = None
        unsqueeze_199 = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        mul_349 = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_197);  mul_343 = unsqueeze_197 = None
        add_263 = torch.ops.aten.add.Tensor(mul_349, unsqueeze_199);  mul_349 = unsqueeze_199 = None
        add_264 = torch.ops.aten.add.Tensor(add_263, relu_42);  add_263 = None
        relu_45 = torch.ops.aten.relu.default(add_264);  add_264 = None
        convolution_50 = torch.ops.aten.convolution.default(relu_45, primals_151, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_265 = torch.ops.aten.add.Tensor(clone_152, 1);  clone_152 = None
        var_mean_50 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
        getitem_102 = var_mean_50[0]
        getitem_103 = var_mean_50[1];  var_mean_50 = None
        add_266 = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
        rsqrt_50 = torch.ops.aten.rsqrt.default(add_266);  add_266 = None
        sub_50 = torch.ops.aten.sub.Tensor(convolution_50, getitem_103)
        mul_350 = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = rsqrt_50 = None
        squeeze_150 = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
        mul_351 = torch.ops.aten.mul.Tensor(squeeze_150, 0.1);  squeeze_150 = None
        mul_352 = torch.ops.aten.mul.Tensor(clone_150, 0.9);  clone_150 = None
        add_267 = torch.ops.aten.add.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
        squeeze_152 = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
        mul_353 = torch.ops.aten.mul.Tensor(squeeze_152, 1.1428571428571428);  squeeze_152 = None
        mul_354 = torch.ops.aten.mul.Tensor(mul_353, 0.1);  mul_353 = None
        mul_355 = torch.ops.aten.mul.Tensor(clone_151, 0.9);  clone_151 = None
        add_268 = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
        unsqueeze_200 = torch.ops.aten.unsqueeze.default(primals_152, -1)
        unsqueeze_201 = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        unsqueeze_202 = torch.ops.aten.unsqueeze.default(primals_153, -1);  primals_153 = None
        unsqueeze_203 = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        mul_356 = torch.ops.aten.mul.Tensor(mul_350, unsqueeze_201);  mul_350 = unsqueeze_201 = None
        add_269 = torch.ops.aten.add.Tensor(mul_356, unsqueeze_203);  mul_356 = unsqueeze_203 = None
        relu_46 = torch.ops.aten.relu.default(add_269);  add_269 = None
        convolution_51 = torch.ops.aten.convolution.default(relu_46, primals_154, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_270 = torch.ops.aten.add.Tensor(clone_155, 1);  clone_155 = None
        var_mean_51 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
        getitem_104 = var_mean_51[0]
        getitem_105 = var_mean_51[1];  var_mean_51 = None
        add_271 = torch.ops.aten.add.Tensor(getitem_104, 1e-05)
        rsqrt_51 = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
        sub_51 = torch.ops.aten.sub.Tensor(convolution_51, getitem_105)
        mul_357 = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = rsqrt_51 = None
        squeeze_153 = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
        mul_358 = torch.ops.aten.mul.Tensor(squeeze_153, 0.1);  squeeze_153 = None
        mul_359 = torch.ops.aten.mul.Tensor(clone_153, 0.9);  clone_153 = None
        add_272 = torch.ops.aten.add.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
        squeeze_155 = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
        mul_360 = torch.ops.aten.mul.Tensor(squeeze_155, 1.1428571428571428);  squeeze_155 = None
        mul_361 = torch.ops.aten.mul.Tensor(mul_360, 0.1);  mul_360 = None
        mul_362 = torch.ops.aten.mul.Tensor(clone_154, 0.9);  clone_154 = None
        add_273 = torch.ops.aten.add.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
        unsqueeze_204 = torch.ops.aten.unsqueeze.default(primals_155, -1)
        unsqueeze_205 = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        unsqueeze_206 = torch.ops.aten.unsqueeze.default(primals_156, -1);  primals_156 = None
        unsqueeze_207 = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        mul_363 = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
        add_274 = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None
        relu_47 = torch.ops.aten.relu.default(add_274);  add_274 = None
        convolution_52 = torch.ops.aten.convolution.default(relu_47, primals_157, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_275 = torch.ops.aten.add.Tensor(clone_158, 1);  clone_158 = None
        var_mean_52 = torch.ops.aten.var_mean.correction(convolution_52, [0, 2, 3], correction = 0, keepdim = True)
        getitem_106 = var_mean_52[0]
        getitem_107 = var_mean_52[1];  var_mean_52 = None
        add_276 = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
        rsqrt_52 = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
        sub_52 = torch.ops.aten.sub.Tensor(convolution_52, getitem_107)
        mul_364 = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = rsqrt_52 = None
        squeeze_156 = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
        mul_365 = torch.ops.aten.mul.Tensor(squeeze_156, 0.1);  squeeze_156 = None
        mul_366 = torch.ops.aten.mul.Tensor(clone_156, 0.9);  clone_156 = None
        add_277 = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
        squeeze_158 = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
        mul_367 = torch.ops.aten.mul.Tensor(squeeze_158, 1.1428571428571428);  squeeze_158 = None
        mul_368 = torch.ops.aten.mul.Tensor(mul_367, 0.1);  mul_367 = None
        mul_369 = torch.ops.aten.mul.Tensor(clone_157, 0.9);  clone_157 = None
        add_278 = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
        unsqueeze_208 = torch.ops.aten.unsqueeze.default(primals_158, -1)
        unsqueeze_209 = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
        unsqueeze_210 = torch.ops.aten.unsqueeze.default(primals_159, -1);  primals_159 = None
        unsqueeze_211 = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
        mul_370 = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_209);  mul_364 = unsqueeze_209 = None
        add_279 = torch.ops.aten.add.Tensor(mul_370, unsqueeze_211);  mul_370 = unsqueeze_211 = None
        add_280 = torch.ops.aten.add.Tensor(add_279, relu_45);  add_279 = None
        relu_48 = torch.ops.aten.relu.default(add_280);  add_280 = None
        mean = torch.ops.aten.mean.dim(relu_48, [-1, -2], True)
        view = torch.ops.aten.view.default(mean, [8, 2048]);  mean = None
        permute = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
        addmm = torch.ops.aten.addmm.default(primals_161, view, permute);  primals_161 = None
        permute_1 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        le = torch.ops.aten.le.Scalar(relu_48, 0);  relu_48 = None
        unsqueeze_320 = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
        unsqueeze_321 = torch.ops.aten.unsqueeze.default(unsqueeze_320, 2);  unsqueeze_320 = None
        unsqueeze_322 = torch.ops.aten.unsqueeze.default(unsqueeze_321, 3);  unsqueeze_321 = None
        unsqueeze_332 = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
        unsqueeze_333 = torch.ops.aten.unsqueeze.default(unsqueeze_332, 2);  unsqueeze_332 = None
        unsqueeze_334 = torch.ops.aten.unsqueeze.default(unsqueeze_333, 3);  unsqueeze_333 = None
        unsqueeze_344 = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
        unsqueeze_345 = torch.ops.aten.unsqueeze.default(unsqueeze_344, 2);  unsqueeze_344 = None
        unsqueeze_346 = torch.ops.aten.unsqueeze.default(unsqueeze_345, 3);  unsqueeze_345 = None
        unsqueeze_356 = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
        unsqueeze_357 = torch.ops.aten.unsqueeze.default(unsqueeze_356, 2);  unsqueeze_356 = None
        unsqueeze_358 = torch.ops.aten.unsqueeze.default(unsqueeze_357, 3);  unsqueeze_357 = None
        unsqueeze_368 = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
        unsqueeze_369 = torch.ops.aten.unsqueeze.default(unsqueeze_368, 2);  unsqueeze_368 = None
        unsqueeze_370 = torch.ops.aten.unsqueeze.default(unsqueeze_369, 3);  unsqueeze_369 = None
        unsqueeze_380 = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
        unsqueeze_381 = torch.ops.aten.unsqueeze.default(unsqueeze_380, 2);  unsqueeze_380 = None
        unsqueeze_382 = torch.ops.aten.unsqueeze.default(unsqueeze_381, 3);  unsqueeze_381 = None
        unsqueeze_392 = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
        unsqueeze_393 = torch.ops.aten.unsqueeze.default(unsqueeze_392, 2);  unsqueeze_392 = None
        unsqueeze_394 = torch.ops.aten.unsqueeze.default(unsqueeze_393, 3);  unsqueeze_393 = None
        unsqueeze_404 = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
        unsqueeze_405 = torch.ops.aten.unsqueeze.default(unsqueeze_404, 2);  unsqueeze_404 = None
        unsqueeze_406 = torch.ops.aten.unsqueeze.default(unsqueeze_405, 3);  unsqueeze_405 = None
        unsqueeze_416 = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
        unsqueeze_417 = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
        unsqueeze_418 = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None
        unsqueeze_428 = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
        unsqueeze_429 = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
        unsqueeze_430 = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None
        unsqueeze_440 = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
        unsqueeze_441 = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
        unsqueeze_442 = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None
        unsqueeze_452 = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
        unsqueeze_453 = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
        unsqueeze_454 = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
        unsqueeze_464 = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
        unsqueeze_465 = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
        unsqueeze_466 = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
        unsqueeze_476 = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
        unsqueeze_477 = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
        unsqueeze_478 = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
        unsqueeze_488 = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
        unsqueeze_489 = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
        unsqueeze_490 = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
        unsqueeze_500 = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
        unsqueeze_501 = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
        unsqueeze_502 = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
        unsqueeze_512 = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
        unsqueeze_513 = torch.ops.aten.unsqueeze.default(unsqueeze_512, 2);  unsqueeze_512 = None
        unsqueeze_514 = torch.ops.aten.unsqueeze.default(unsqueeze_513, 3);  unsqueeze_513 = None
        unsqueeze_524 = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
        unsqueeze_525 = torch.ops.aten.unsqueeze.default(unsqueeze_524, 2);  unsqueeze_524 = None
        unsqueeze_526 = torch.ops.aten.unsqueeze.default(unsqueeze_525, 3);  unsqueeze_525 = None
        unsqueeze_536 = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
        unsqueeze_537 = torch.ops.aten.unsqueeze.default(unsqueeze_536, 2);  unsqueeze_536 = None
        unsqueeze_538 = torch.ops.aten.unsqueeze.default(unsqueeze_537, 3);  unsqueeze_537 = None
        unsqueeze_548 = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
        unsqueeze_549 = torch.ops.aten.unsqueeze.default(unsqueeze_548, 2);  unsqueeze_548 = None
        unsqueeze_550 = torch.ops.aten.unsqueeze.default(unsqueeze_549, 3);  unsqueeze_549 = None
        unsqueeze_560 = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
        unsqueeze_561 = torch.ops.aten.unsqueeze.default(unsqueeze_560, 2);  unsqueeze_560 = None
        unsqueeze_562 = torch.ops.aten.unsqueeze.default(unsqueeze_561, 3);  unsqueeze_561 = None
        unsqueeze_572 = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
        unsqueeze_573 = torch.ops.aten.unsqueeze.default(unsqueeze_572, 2);  unsqueeze_572 = None
        unsqueeze_574 = torch.ops.aten.unsqueeze.default(unsqueeze_573, 3);  unsqueeze_573 = None
        unsqueeze_584 = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
        unsqueeze_585 = torch.ops.aten.unsqueeze.default(unsqueeze_584, 2);  unsqueeze_584 = None
        unsqueeze_586 = torch.ops.aten.unsqueeze.default(unsqueeze_585, 3);  unsqueeze_585 = None
        unsqueeze_596 = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
        unsqueeze_597 = torch.ops.aten.unsqueeze.default(unsqueeze_596, 2);  unsqueeze_596 = None
        unsqueeze_598 = torch.ops.aten.unsqueeze.default(unsqueeze_597, 3);  unsqueeze_597 = None
        unsqueeze_608 = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_609 = torch.ops.aten.unsqueeze.default(unsqueeze_608, 2);  unsqueeze_608 = None
        unsqueeze_610 = torch.ops.aten.unsqueeze.default(unsqueeze_609, 3);  unsqueeze_609 = None
        unsqueeze_620 = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_621 = torch.ops.aten.unsqueeze.default(unsqueeze_620, 2);  unsqueeze_620 = None
        unsqueeze_622 = torch.ops.aten.unsqueeze.default(unsqueeze_621, 3);  unsqueeze_621 = None
        unsqueeze_632 = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_633 = torch.ops.aten.unsqueeze.default(unsqueeze_632, 2);  unsqueeze_632 = None
        unsqueeze_634 = torch.ops.aten.unsqueeze.default(unsqueeze_633, 3);  unsqueeze_633 = None
        unsqueeze_644 = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_645 = torch.ops.aten.unsqueeze.default(unsqueeze_644, 2);  unsqueeze_644 = None
        unsqueeze_646 = torch.ops.aten.unsqueeze.default(unsqueeze_645, 3);  unsqueeze_645 = None
        unsqueeze_656 = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_657 = torch.ops.aten.unsqueeze.default(unsqueeze_656, 2);  unsqueeze_656 = None
        unsqueeze_658 = torch.ops.aten.unsqueeze.default(unsqueeze_657, 3);  unsqueeze_657 = None
        unsqueeze_668 = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_669 = torch.ops.aten.unsqueeze.default(unsqueeze_668, 2);  unsqueeze_668 = None
        unsqueeze_670 = torch.ops.aten.unsqueeze.default(unsqueeze_669, 3);  unsqueeze_669 = None
        unsqueeze_680 = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_681 = torch.ops.aten.unsqueeze.default(unsqueeze_680, 2);  unsqueeze_680 = None
        unsqueeze_682 = torch.ops.aten.unsqueeze.default(unsqueeze_681, 3);  unsqueeze_681 = None
        unsqueeze_692 = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_693 = torch.ops.aten.unsqueeze.default(unsqueeze_692, 2);  unsqueeze_692 = None
        unsqueeze_694 = torch.ops.aten.unsqueeze.default(unsqueeze_693, 3);  unsqueeze_693 = None
        unsqueeze_704 = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_705 = torch.ops.aten.unsqueeze.default(unsqueeze_704, 2);  unsqueeze_704 = None
        unsqueeze_706 = torch.ops.aten.unsqueeze.default(unsqueeze_705, 3);  unsqueeze_705 = None
        unsqueeze_716 = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_717 = torch.ops.aten.unsqueeze.default(unsqueeze_716, 2);  unsqueeze_716 = None
        unsqueeze_718 = torch.ops.aten.unsqueeze.default(unsqueeze_717, 3);  unsqueeze_717 = None
        unsqueeze_728 = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_729 = torch.ops.aten.unsqueeze.default(unsqueeze_728, 2);  unsqueeze_728 = None
        unsqueeze_730 = torch.ops.aten.unsqueeze.default(unsqueeze_729, 3);  unsqueeze_729 = None
        unsqueeze_740 = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_741 = torch.ops.aten.unsqueeze.default(unsqueeze_740, 2);  unsqueeze_740 = None
        unsqueeze_742 = torch.ops.aten.unsqueeze.default(unsqueeze_741, 3);  unsqueeze_741 = None
        unsqueeze_752 = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_753 = torch.ops.aten.unsqueeze.default(unsqueeze_752, 2);  unsqueeze_752 = None
        unsqueeze_754 = torch.ops.aten.unsqueeze.default(unsqueeze_753, 3);  unsqueeze_753 = None
        unsqueeze_764 = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_765 = torch.ops.aten.unsqueeze.default(unsqueeze_764, 2);  unsqueeze_764 = None
        unsqueeze_766 = torch.ops.aten.unsqueeze.default(unsqueeze_765, 3);  unsqueeze_765 = None
        unsqueeze_776 = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_777 = torch.ops.aten.unsqueeze.default(unsqueeze_776, 2);  unsqueeze_776 = None
        unsqueeze_778 = torch.ops.aten.unsqueeze.default(unsqueeze_777, 3);  unsqueeze_777 = None
        unsqueeze_788 = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_789 = torch.ops.aten.unsqueeze.default(unsqueeze_788, 2);  unsqueeze_788 = None
        unsqueeze_790 = torch.ops.aten.unsqueeze.default(unsqueeze_789, 3);  unsqueeze_789 = None
        unsqueeze_800 = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_801 = torch.ops.aten.unsqueeze.default(unsqueeze_800, 2);  unsqueeze_800 = None
        unsqueeze_802 = torch.ops.aten.unsqueeze.default(unsqueeze_801, 3);  unsqueeze_801 = None
        unsqueeze_812 = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_813 = torch.ops.aten.unsqueeze.default(unsqueeze_812, 2);  unsqueeze_812 = None
        unsqueeze_814 = torch.ops.aten.unsqueeze.default(unsqueeze_813, 3);  unsqueeze_813 = None
        unsqueeze_824 = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_825 = torch.ops.aten.unsqueeze.default(unsqueeze_824, 2);  unsqueeze_824 = None
        unsqueeze_826 = torch.ops.aten.unsqueeze.default(unsqueeze_825, 3);  unsqueeze_825 = None
        unsqueeze_836 = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_837 = torch.ops.aten.unsqueeze.default(unsqueeze_836, 2);  unsqueeze_836 = None
        unsqueeze_838 = torch.ops.aten.unsqueeze.default(unsqueeze_837, 3);  unsqueeze_837 = None
        return [add_2, add_3, add, add_7, add_8, add_5, add_12, add_13, add_10, add_17, add_18, add_15, add_22, add_23, add_20, add_28, add_29, add_26, add_33, add_34, add_31, add_38, add_39, add_36, add_44, add_45, add_42, add_49, add_50, add_47, add_54, add_55, add_52, add_60, add_61, add_58, add_65, add_66, add_63, add_70, add_71, add_68, add_75, add_76, add_73, add_81, add_82, add_79, add_86, add_87, add_84, add_91, add_92, add_89, add_97, add_98, add_95, add_102, add_103, add_100, add_107, add_108, add_105, add_113, add_114, add_111, add_118, add_119, add_116, add_123, add_124, add_121, add_129, add_130, add_127, add_134, add_135, add_132, add_139, add_140, add_137, add_144, add_145, add_142, add_150, add_151, add_148, add_155, add_156, add_153, add_160, add_161, add_158, add_166, add_167, add_164, add_171, add_172, add_169, add_176, add_177, add_174, add_182, add_183, add_180, add_187, add_188, add_185, add_192, add_193, add_190, add_198, add_199, add_196, add_203, add_204, add_201, add_208, add_209, add_206, add_214, add_215, add_212, add_219, add_220, add_217, add_224, add_225, add_222, add_230, add_231, add_228, add_235, add_236, add_233, add_240, add_241, add_238, add_245, add_246, add_243, add_251, add_252, add_249, add_256, add_257, add_254, add_261, add_262, add_259, add_267, add_268, add_265, add_272, add_273, add_270, add_277, add_278, add_275, addmm, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_321, convolution, squeeze_1, relu, getitem_2, getitem_3, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, relu_4, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, relu_8, convolution_10, squeeze_31, relu_9, convolution_11, squeeze_34, relu_10, convolution_12, squeeze_37, relu_11, convolution_13, squeeze_40, convolution_14, squeeze_43, relu_12, convolution_15, squeeze_46, relu_13, convolution_16, squeeze_49, relu_14, convolution_17, squeeze_52, relu_15, convolution_18, squeeze_55, relu_16, convolution_19, squeeze_58, relu_17, convolution_20, squeeze_61, relu_18, convolution_21, squeeze_64, relu_19, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, relu_21, convolution_24, squeeze_73, relu_22, convolution_25, squeeze_76, relu_23, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_24, convolution_28, squeeze_85, relu_25, convolution_29, squeeze_88, relu_26, convolution_30, squeeze_91, relu_27, convolution_31, squeeze_94, relu_28, convolution_32, squeeze_97, relu_29, convolution_33, squeeze_100, relu_30, convolution_34, squeeze_103, relu_31, convolution_35, squeeze_106, relu_32, convolution_36, squeeze_109, relu_33, convolution_37, squeeze_112, relu_34, convolution_38, squeeze_115, relu_35, convolution_39, squeeze_118, relu_36, convolution_40, squeeze_121, relu_37, convolution_41, squeeze_124, relu_38, convolution_42, squeeze_127, relu_39, convolution_43, squeeze_130, relu_40, convolution_44, relu_41, convolution_45, convolution_46, relu_42, convolution_47, relu_43, convolution_48, relu_44, convolution_49, relu_45, convolution_50, relu_46, convolution_51, relu_47, convolution_52, view, permute_1, le, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838]
        
args = [((64, 3, 7, 7), (147, 49, 7, 1), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64, 64, 3, 3), (576, 9, 3, 1), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((256, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((64, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64, 64, 3, 3), (576, 9, 3, 1), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((256, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((64, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64, 64, 3, 3), (576, 9, 3, 1), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((256, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((128, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((512, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((128, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((512, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((128, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((512, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((128, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((512, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((256, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((512, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512, 3, 3), (4608, 9, 3, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((2048, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((2048, 1024, 1, 1), (1024, 1, 1, 1), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((512, 2048, 1, 1), (2048, 1, 1, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512, 3, 3), (4608, 9, 3, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((2048, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((512, 2048, 1, 1), (2048, 1, 1, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512, 3, 3), (4608, 9, 3, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((2048, 512, 1, 1), (512, 1, 1, 1), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((1000, 2048), (2048, 1), torch.float32, 'cpu'), ((1000,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((64,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((128,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((256,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((), (), torch.int64, 'cpu'), ((8, 3, 32, 32), (3072, 1024, 32, 1), torch.float32, 'cpu')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro(), tracing_mode='real')(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
ref = compiled(args)

