class GraphModule(torch.nn.Module):
    def forward(self, primals_1: f32[64, 3, 7, 7], primals_2: f32[64], primals_4: f32[64, 64, 1, 1], primals_5: f32[64], primals_7: f32[64, 64, 3, 3], primals_8: f32[64], primals_10: f32[256, 64, 1, 1], primals_11: f32[256], primals_13: f32[256, 64, 1, 1], primals_14: f32[256], primals_16: f32[64, 256, 1, 1], primals_17: f32[64], primals_19: f32[64, 64, 3, 3], primals_20: f32[64], primals_22: f32[256, 64, 1, 1], primals_23: f32[256], primals_25: f32[64, 256, 1, 1], primals_26: f32[64], primals_28: f32[64, 64, 3, 3], primals_29: f32[64], primals_31: f32[256, 64, 1, 1], primals_32: f32[256], primals_34: f32[128, 256, 1, 1], primals_35: f32[128], primals_37: f32[128, 128, 3, 3], primals_38: f32[128], primals_40: f32[512, 128, 1, 1], primals_41: f32[512], primals_43: f32[512, 256, 1, 1], primals_44: f32[512], primals_46: f32[128, 512, 1, 1], primals_47: f32[128], primals_49: f32[128, 128, 3, 3], primals_50: f32[128], primals_52: f32[512, 128, 1, 1], primals_53: f32[512], primals_55: f32[128, 512, 1, 1], primals_56: f32[128], primals_58: f32[128, 128, 3, 3], primals_59: f32[128], primals_61: f32[512, 128, 1, 1], primals_62: f32[512], primals_64: f32[128, 512, 1, 1], primals_65: f32[128], primals_67: f32[128, 128, 3, 3], primals_68: f32[128], primals_70: f32[512, 128, 1, 1], primals_71: f32[512], primals_73: f32[256, 512, 1, 1], primals_74: f32[256], primals_76: f32[256, 256, 3, 3], primals_77: f32[256], primals_79: f32[1024, 256, 1, 1], primals_80: f32[1024], primals_82: f32[1024, 512, 1, 1], primals_83: f32[1024], primals_85: f32[256, 1024, 1, 1], primals_86: f32[256], primals_88: f32[256, 256, 3, 3], primals_89: f32[256], primals_91: f32[1024, 256, 1, 1], primals_92: f32[1024], primals_94: f32[256, 1024, 1, 1], primals_95: f32[256], primals_97: f32[256, 256, 3, 3], primals_98: f32[256], primals_100: f32[1024, 256, 1, 1], primals_101: f32[1024], primals_103: f32[256, 1024, 1, 1], primals_104: f32[256], primals_106: f32[256, 256, 3, 3], primals_107: f32[256], primals_109: f32[1024, 256, 1, 1], primals_110: f32[1024], primals_112: f32[256, 1024, 1, 1], primals_113: f32[256], primals_115: f32[256, 256, 3, 3], primals_116: f32[256], primals_118: f32[1024, 256, 1, 1], primals_119: f32[1024], primals_121: f32[256, 1024, 1, 1], primals_122: f32[256], primals_124: f32[256, 256, 3, 3], primals_125: f32[256], primals_127: f32[1024, 256, 1, 1], primals_128: f32[1024], primals_130: f32[512, 1024, 1, 1], primals_131: f32[512], primals_133: f32[512, 512, 3, 3], primals_134: f32[512], primals_136: f32[2048, 512, 1, 1], primals_137: f32[2048], primals_139: f32[2048, 1024, 1, 1], primals_140: f32[2048], primals_142: f32[512, 2048, 1, 1], primals_143: f32[512], primals_145: f32[512, 512, 3, 3], primals_146: f32[512], primals_148: f32[2048, 512, 1, 1], primals_149: f32[2048], primals_151: f32[512, 2048, 1, 1], primals_152: f32[512], primals_154: f32[512, 512, 3, 3], primals_155: f32[512], primals_157: f32[2048, 512, 1, 1], primals_158: f32[2048], primals_321: f32[64, 3, 32, 32], convolution: f32[64, 64, 16, 16], squeeze_1: f32[64], relu: f32[64, 64, 16, 16], getitem_2: f32[64, 64, 8, 8], getitem_3: i64[64, 64, 8, 8], convolution_1: f32[64, 64, 8, 8], squeeze_4: f32[64], relu_1: f32[64, 64, 8, 8], convolution_2: f32[64, 64, 8, 8], squeeze_7: f32[64], relu_2: f32[64, 64, 8, 8], convolution_3: f32[64, 256, 8, 8], squeeze_10: f32[256], convolution_4: f32[64, 256, 8, 8], squeeze_13: f32[256], relu_3: f32[64, 256, 8, 8], convolution_5: f32[64, 64, 8, 8], squeeze_16: f32[64], relu_4: f32[64, 64, 8, 8], convolution_6: f32[64, 64, 8, 8], squeeze_19: f32[64], relu_5: f32[64, 64, 8, 8], convolution_7: f32[64, 256, 8, 8], squeeze_22: f32[256], relu_6: f32[64, 256, 8, 8], convolution_8: f32[64, 64, 8, 8], squeeze_25: f32[64], relu_7: f32[64, 64, 8, 8], convolution_9: f32[64, 64, 8, 8], squeeze_28: f32[64], relu_8: f32[64, 64, 8, 8], convolution_10: f32[64, 256, 8, 8], squeeze_31: f32[256], relu_9: f32[64, 256, 8, 8], convolution_11: f32[64, 128, 8, 8], squeeze_34: f32[128], relu_10: f32[64, 128, 8, 8], convolution_12: f32[64, 128, 4, 4], squeeze_37: f32[128], relu_11: f32[64, 128, 4, 4], convolution_13: f32[64, 512, 4, 4], squeeze_40: f32[512], convolution_14: f32[64, 512, 4, 4], squeeze_43: f32[512], relu_12: f32[64, 512, 4, 4], convolution_15: f32[64, 128, 4, 4], squeeze_46: f32[128], relu_13: f32[64, 128, 4, 4], convolution_16: f32[64, 128, 4, 4], squeeze_49: f32[128], relu_14: f32[64, 128, 4, 4], convolution_17: f32[64, 512, 4, 4], squeeze_52: f32[512], relu_15: f32[64, 512, 4, 4], convolution_18: f32[64, 128, 4, 4], squeeze_55: f32[128], relu_16: f32[64, 128, 4, 4], convolution_19: f32[64, 128, 4, 4], squeeze_58: f32[128], relu_17: f32[64, 128, 4, 4], convolution_20: f32[64, 512, 4, 4], squeeze_61: f32[512], relu_18: f32[64, 512, 4, 4], convolution_21: f32[64, 128, 4, 4], squeeze_64: f32[128], relu_19: f32[64, 128, 4, 4], convolution_22: f32[64, 128, 4, 4], squeeze_67: f32[128], relu_20: f32[64, 128, 4, 4], convolution_23: f32[64, 512, 4, 4], squeeze_70: f32[512], relu_21: f32[64, 512, 4, 4], convolution_24: f32[64, 256, 4, 4], squeeze_73: f32[256], relu_22: f32[64, 256, 4, 4], convolution_25: f32[64, 256, 2, 2], squeeze_76: f32[256], relu_23: f32[64, 256, 2, 2], convolution_26: f32[64, 1024, 2, 2], squeeze_79: f32[1024], convolution_27: f32[64, 1024, 2, 2], squeeze_82: f32[1024], relu_24: f32[64, 1024, 2, 2], convolution_28: f32[64, 256, 2, 2], squeeze_85: f32[256], relu_25: f32[64, 256, 2, 2], convolution_29: f32[64, 256, 2, 2], squeeze_88: f32[256], relu_26: f32[64, 256, 2, 2], convolution_30: f32[64, 1024, 2, 2], squeeze_91: f32[1024], relu_27: f32[64, 1024, 2, 2], convolution_31: f32[64, 256, 2, 2], squeeze_94: f32[256], relu_28: f32[64, 256, 2, 2], convolution_32: f32[64, 256, 2, 2], squeeze_97: f32[256], relu_29: f32[64, 256, 2, 2], convolution_33: f32[64, 1024, 2, 2], squeeze_100: f32[1024], relu_30: f32[64, 1024, 2, 2], convolution_34: f32[64, 256, 2, 2], squeeze_103: f32[256], relu_31: f32[64, 256, 2, 2], convolution_35: f32[64, 256, 2, 2], squeeze_106: f32[256], relu_32: f32[64, 256, 2, 2], convolution_36: f32[64, 1024, 2, 2], squeeze_109: f32[1024], relu_33: f32[64, 1024, 2, 2], convolution_37: f32[64, 256, 2, 2], squeeze_112: f32[256], relu_34: f32[64, 256, 2, 2], convolution_38: f32[64, 256, 2, 2], squeeze_115: f32[256], relu_35: f32[64, 256, 2, 2], convolution_39: f32[64, 1024, 2, 2], squeeze_118: f32[1024], relu_36: f32[64, 1024, 2, 2], convolution_40: f32[64, 256, 2, 2], squeeze_121: f32[256], relu_37: f32[64, 256, 2, 2], convolution_41: f32[64, 256, 2, 2], squeeze_124: f32[256], relu_38: f32[64, 256, 2, 2], convolution_42: f32[64, 1024, 2, 2], squeeze_127: f32[1024], relu_39: f32[64, 1024, 2, 2], convolution_43: f32[64, 512, 2, 2], squeeze_130: f32[512], relu_40: f32[64, 512, 2, 2], convolution_44: f32[64, 512, 1, 1], squeeze_133: f32[512], relu_41: f32[64, 512, 1, 1], convolution_45: f32[64, 2048, 1, 1], squeeze_136: f32[2048], convolution_46: f32[64, 2048, 1, 1], squeeze_139: f32[2048], relu_42: f32[64, 2048, 1, 1], convolution_47: f32[64, 512, 1, 1], squeeze_142: f32[512], relu_43: f32[64, 512, 1, 1], convolution_48: f32[64, 512, 1, 1], squeeze_145: f32[512], relu_44: f32[64, 512, 1, 1], convolution_49: f32[64, 2048, 1, 1], squeeze_148: f32[2048], relu_45: f32[64, 2048, 1, 1], convolution_50: f32[64, 512, 1, 1], squeeze_151: f32[512], relu_46: f32[64, 512, 1, 1], convolution_51: f32[64, 512, 1, 1], squeeze_154: f32[512], relu_47: f32[64, 512, 1, 1], convolution_52: f32[64, 2048, 1, 1], squeeze_157: f32[2048], view: f32[64, 2048], permute_1: f32[1000, 2048], le: b8[64, 2048, 1, 1], unsqueeze_214: f32[1, 2048, 1, 1], unsqueeze_226: f32[1, 512, 1, 1], unsqueeze_238: f32[1, 512, 1, 1], unsqueeze_250: f32[1, 2048, 1, 1], unsqueeze_262: f32[1, 512, 1, 1], unsqueeze_274: f32[1, 512, 1, 1], unsqueeze_286: f32[1, 2048, 1, 1], unsqueeze_298: f32[1, 2048, 1, 1], unsqueeze_310: f32[1, 512, 1, 1], unsqueeze_322: f32[1, 512, 1, 1], unsqueeze_334: f32[1, 1024, 1, 1], unsqueeze_346: f32[1, 256, 1, 1], unsqueeze_358: f32[1, 256, 1, 1], unsqueeze_370: f32[1, 1024, 1, 1], unsqueeze_382: f32[1, 256, 1, 1], unsqueeze_394: f32[1, 256, 1, 1], unsqueeze_406: f32[1, 1024, 1, 1], unsqueeze_418: f32[1, 256, 1, 1], unsqueeze_430: f32[1, 256, 1, 1], unsqueeze_442: f32[1, 1024, 1, 1], unsqueeze_454: f32[1, 256, 1, 1], unsqueeze_466: f32[1, 256, 1, 1], unsqueeze_478: f32[1, 1024, 1, 1], unsqueeze_490: f32[1, 256, 1, 1], unsqueeze_502: f32[1, 256, 1, 1], unsqueeze_514: f32[1, 1024, 1, 1], unsqueeze_526: f32[1, 1024, 1, 1], unsqueeze_538: f32[1, 256, 1, 1], unsqueeze_550: f32[1, 256, 1, 1], unsqueeze_562: f32[1, 512, 1, 1], unsqueeze_574: f32[1, 128, 1, 1], unsqueeze_586: f32[1, 128, 1, 1], unsqueeze_598: f32[1, 512, 1, 1], unsqueeze_610: f32[1, 128, 1, 1], unsqueeze_622: f32[1, 128, 1, 1], unsqueeze_634: f32[1, 512, 1, 1], unsqueeze_646: f32[1, 128, 1, 1], unsqueeze_658: f32[1, 128, 1, 1], unsqueeze_670: f32[1, 512, 1, 1], unsqueeze_682: f32[1, 512, 1, 1], unsqueeze_694: f32[1, 128, 1, 1], unsqueeze_706: f32[1, 128, 1, 1], unsqueeze_718: f32[1, 256, 1, 1], unsqueeze_730: f32[1, 64, 1, 1], unsqueeze_742: f32[1, 64, 1, 1], unsqueeze_754: f32[1, 256, 1, 1], unsqueeze_766: f32[1, 64, 1, 1], unsqueeze_778: f32[1, 64, 1, 1], unsqueeze_790: f32[1, 256, 1, 1], unsqueeze_802: f32[1, 256, 1, 1], unsqueeze_814: f32[1, 64, 1, 1], unsqueeze_826: f32[1, 64, 1, 1], unsqueeze_838: f32[1, 64, 1, 1], tangents_1: f32[64], tangents_2: f32[64], tangents_3: i64[], tangents_4: f32[64], tangents_5: f32[64], tangents_6: i64[], tangents_7: f32[64], tangents_8: f32[64], tangents_9: i64[], tangents_10: f32[256], tangents_11: f32[256], tangents_12: i64[], tangents_13: f32[256], tangents_14: f32[256], tangents_15: i64[], tangents_16: f32[64], tangents_17: f32[64], tangents_18: i64[], tangents_19: f32[64], tangents_20: f32[64], tangents_21: i64[], tangents_22: f32[256], tangents_23: f32[256], tangents_24: i64[], tangents_25: f32[64], tangents_26: f32[64], tangents_27: i64[], tangents_28: f32[64], tangents_29: f32[64], tangents_30: i64[], tangents_31: f32[256], tangents_32: f32[256], tangents_33: i64[], tangents_34: f32[128], tangents_35: f32[128], tangents_36: i64[], tangents_37: f32[128], tangents_38: f32[128], tangents_39: i64[], tangents_40: f32[512], tangents_41: f32[512], tangents_42: i64[], tangents_43: f32[512], tangents_44: f32[512], tangents_45: i64[], tangents_46: f32[128], tangents_47: f32[128], tangents_48: i64[], tangents_49: f32[128], tangents_50: f32[128], tangents_51: i64[], tangents_52: f32[512], tangents_53: f32[512], tangents_54: i64[], tangents_55: f32[128], tangents_56: f32[128], tangents_57: i64[], tangents_58: f32[128], tangents_59: f32[128], tangents_60: i64[], tangents_61: f32[512], tangents_62: f32[512], tangents_63: i64[], tangents_64: f32[128], tangents_65: f32[128], tangents_66: i64[], tangents_67: f32[128], tangents_68: f32[128], tangents_69: i64[], tangents_70: f32[512], tangents_71: f32[512], tangents_72: i64[], tangents_73: f32[256], tangents_74: f32[256], tangents_75: i64[], tangents_76: f32[256], tangents_77: f32[256], tangents_78: i64[], tangents_79: f32[1024], tangents_80: f32[1024], tangents_81: i64[], tangents_82: f32[1024], tangents_83: f32[1024], tangents_84: i64[], tangents_85: f32[256], tangents_86: f32[256], tangents_87: i64[], tangents_88: f32[256], tangents_89: f32[256], tangents_90: i64[], tangents_91: f32[1024], tangents_92: f32[1024], tangents_93: i64[], tangents_94: f32[256], tangents_95: f32[256], tangents_96: i64[], tangents_97: f32[256], tangents_98: f32[256], tangents_99: i64[], tangents_100: f32[1024], tangents_101: f32[1024], tangents_102: i64[], tangents_103: f32[256], tangents_104: f32[256], tangents_105: i64[], tangents_106: f32[256], tangents_107: f32[256], tangents_108: i64[], tangents_109: f32[1024], tangents_110: f32[1024], tangents_111: i64[], tangents_112: f32[256], tangents_113: f32[256], tangents_114: i64[], tangents_115: f32[256], tangents_116: f32[256], tangents_117: i64[], tangents_118: f32[1024], tangents_119: f32[1024], tangents_120: i64[], tangents_121: f32[256], tangents_122: f32[256], tangents_123: i64[], tangents_124: f32[256], tangents_125: f32[256], tangents_126: i64[], tangents_127: f32[1024], tangents_128: f32[1024], tangents_129: i64[], tangents_130: f32[512], tangents_131: f32[512], tangents_132: i64[], tangents_133: f32[512], tangents_134: f32[512], tangents_135: i64[], tangents_136: f32[2048], tangents_137: f32[2048], tangents_138: i64[], tangents_139: f32[2048], tangents_140: f32[2048], tangents_141: i64[], tangents_142: f32[512], tangents_143: f32[512], tangents_144: i64[], tangents_145: f32[512], tangents_146: f32[512], tangents_147: i64[], tangents_148: f32[2048], tangents_149: f32[2048], tangents_150: i64[], tangents_151: f32[512], tangents_152: f32[512], tangents_153: i64[], tangents_154: f32[512], tangents_155: f32[512], tangents_156: i64[], tangents_157: f32[2048], tangents_158: f32[2048], tangents_159: i64[], tangents_160: f32[64, 1000]):
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:280, code: x = self.fc(x)
        mm: f32[64, 2048] = torch.ops.aten.mm.default(tangents_160, permute_1);  permute_1 = None
        permute_2: f32[1000, 64] = torch.ops.aten.permute.default(tangents_160, [1, 0])
        mm_1: f32[1000, 2048] = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
        permute_3: f32[2048, 1000] = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        sum_1: f32[1, 1000] = torch.ops.aten.sum.dim_IntList(tangents_160, [0], True);  tangents_160 = None
        view_1: f32[1000] = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
        permute_4: f32[1000, 2048] = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:279, code: x = torch.flatten(x, 1)
        view_2: f32[64, 2048, 1, 1] = torch.ops.aten.view.default(mm, [64, 2048, 1, 1]);  mm = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:278, code: x = self.avgpool(x)
        expand: f32[64, 2048, 1, 1] = torch.ops.aten.expand.default(view_2, [64, 2048, 1, 1]);  view_2 = None
        div: f32[64, 2048, 1, 1] = torch.ops.aten.div.Scalar(expand, 1);  expand = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:161, code: out = self.relu(out)
        scalar_tensor: f32[] = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
        where: f32[64, 2048, 1, 1] = torch.ops.aten.where.self(le, scalar_tensor, div);  le = div = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:155, code: out = self.bn3(out)
        sum_2: f32[2048] = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        sub_53: f32[64, 2048, 1, 1] = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_214);  convolution_52 = unsqueeze_214 = None
        mul_371: f32[64, 2048, 1, 1] = torch.ops.aten.mul.Tensor(where, sub_53)
        sum_3: f32[2048] = torch.ops.aten.sum.dim_IntList(mul_371, [0, 2, 3]);  mul_371 = None
        mul_372: f32[2048] = torch.ops.aten.mul.Tensor(sum_2, 0.015625)
        unsqueeze_215: f32[1, 2048] = torch.ops.aten.unsqueeze.default(mul_372, 0);  mul_372 = None
        unsqueeze_216: f32[1, 2048, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_215, 2);  unsqueeze_215 = None
        unsqueeze_217: f32[1, 2048, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_216, 3);  unsqueeze_216 = None
        mul_373: f32[2048] = torch.ops.aten.mul.Tensor(sum_3, 0.015625)
        mul_374: f32[2048] = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
        mul_375: f32[2048] = torch.ops.aten.mul.Tensor(mul_373, mul_374);  mul_373 = mul_374 = None
        unsqueeze_218: f32[1, 2048] = torch.ops.aten.unsqueeze.default(mul_375, 0);  mul_375 = None
        unsqueeze_219: f32[1, 2048, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_218, 2);  unsqueeze_218 = None
        unsqueeze_220: f32[1, 2048, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_219, 3);  unsqueeze_219 = None
        mul_376: f32[2048] = torch.ops.aten.mul.Tensor(squeeze_157, primals_158);  primals_158 = None
        unsqueeze_221: f32[1, 2048] = torch.ops.aten.unsqueeze.default(mul_376, 0);  mul_376 = None
        unsqueeze_222: f32[1, 2048, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_221, 2);  unsqueeze_221 = None
        unsqueeze_223: f32[1, 2048, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_222, 3);  unsqueeze_222 = None
        mul_377: f32[64, 2048, 1, 1] = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_220);  sub_53 = unsqueeze_220 = None
        sub_55: f32[64, 2048, 1, 1] = torch.ops.aten.sub.Tensor(where, mul_377);  mul_377 = None
        sub_56: f32[64, 2048, 1, 1] = torch.ops.aten.sub.Tensor(sub_55, unsqueeze_217);  sub_55 = unsqueeze_217 = None
        mul_378: f32[64, 2048, 1, 1] = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_223);  sub_56 = unsqueeze_223 = None
        mul_379: f32[2048] = torch.ops.aten.mul.Tensor(sum_3, squeeze_157);  sum_3 = squeeze_157 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:154, code: out = self.conv3(out)
        convolution_backward = torch.ops.aten.convolution_backward.default(mul_378, relu_47, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_378 = primals_157 = None
        getitem_108: f32[64, 512, 1, 1] = convolution_backward[0]
        getitem_109: f32[2048, 512, 1, 1] = convolution_backward[1];  convolution_backward = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:152, code: out = self.relu(out)
        le_1: b8[64, 512, 1, 1] = torch.ops.aten.le.Scalar(relu_47, 0);  relu_47 = None
        where_1: f32[64, 512, 1, 1] = torch.ops.aten.where.self(le_1, scalar_tensor, getitem_108);  le_1 = getitem_108 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:151, code: out = self.bn2(out)
        sum_4: f32[512] = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
        sub_57: f32[64, 512, 1, 1] = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_226);  convolution_51 = unsqueeze_226 = None
        mul_380: f32[64, 512, 1, 1] = torch.ops.aten.mul.Tensor(where_1, sub_57)
        sum_5: f32[512] = torch.ops.aten.sum.dim_IntList(mul_380, [0, 2, 3]);  mul_380 = None
        mul_381: f32[512] = torch.ops.aten.mul.Tensor(sum_4, 0.015625)
        unsqueeze_227: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_381, 0);  mul_381 = None
        unsqueeze_228: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_227, 2);  unsqueeze_227 = None
        unsqueeze_229: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_228, 3);  unsqueeze_228 = None
        mul_382: f32[512] = torch.ops.aten.mul.Tensor(sum_5, 0.015625)
        mul_383: f32[512] = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
        mul_384: f32[512] = torch.ops.aten.mul.Tensor(mul_382, mul_383);  mul_382 = mul_383 = None
        unsqueeze_230: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_384, 0);  mul_384 = None
        unsqueeze_231: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_230, 2);  unsqueeze_230 = None
        unsqueeze_232: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_231, 3);  unsqueeze_231 = None
        mul_385: f32[512] = torch.ops.aten.mul.Tensor(squeeze_154, primals_155);  primals_155 = None
        unsqueeze_233: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_385, 0);  mul_385 = None
        unsqueeze_234: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_233, 2);  unsqueeze_233 = None
        unsqueeze_235: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_234, 3);  unsqueeze_234 = None
        mul_386: f32[64, 512, 1, 1] = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_232);  sub_57 = unsqueeze_232 = None
        sub_59: f32[64, 512, 1, 1] = torch.ops.aten.sub.Tensor(where_1, mul_386);  where_1 = mul_386 = None
        sub_60: f32[64, 512, 1, 1] = torch.ops.aten.sub.Tensor(sub_59, unsqueeze_229);  sub_59 = unsqueeze_229 = None
        mul_387: f32[64, 512, 1, 1] = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_235);  sub_60 = unsqueeze_235 = None
        mul_388: f32[512] = torch.ops.aten.mul.Tensor(sum_5, squeeze_154);  sum_5 = squeeze_154 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:150, code: out = self.conv2(out)
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_387, relu_46, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_387 = primals_154 = None
        getitem_111: f32[64, 512, 1, 1] = convolution_backward_1[0]
        getitem_112: f32[512, 512, 3, 3] = convolution_backward_1[1];  convolution_backward_1 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:148, code: out = self.relu(out)
        le_2: b8[64, 512, 1, 1] = torch.ops.aten.le.Scalar(relu_46, 0);  relu_46 = None
        where_2: f32[64, 512, 1, 1] = torch.ops.aten.where.self(le_2, scalar_tensor, getitem_111);  le_2 = getitem_111 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:147, code: out = self.bn1(out)
        sum_6: f32[512] = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        sub_61: f32[64, 512, 1, 1] = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_238);  convolution_50 = unsqueeze_238 = None
        mul_389: f32[64, 512, 1, 1] = torch.ops.aten.mul.Tensor(where_2, sub_61)
        sum_7: f32[512] = torch.ops.aten.sum.dim_IntList(mul_389, [0, 2, 3]);  mul_389 = None
        mul_390: f32[512] = torch.ops.aten.mul.Tensor(sum_6, 0.015625)
        unsqueeze_239: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_390, 0);  mul_390 = None
        unsqueeze_240: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_239, 2);  unsqueeze_239 = None
        unsqueeze_241: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_240, 3);  unsqueeze_240 = None
        mul_391: f32[512] = torch.ops.aten.mul.Tensor(sum_7, 0.015625)
        mul_392: f32[512] = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
        mul_393: f32[512] = torch.ops.aten.mul.Tensor(mul_391, mul_392);  mul_391 = mul_392 = None
        unsqueeze_242: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_393, 0);  mul_393 = None
        unsqueeze_243: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_242, 2);  unsqueeze_242 = None
        unsqueeze_244: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_243, 3);  unsqueeze_243 = None
        mul_394: f32[512] = torch.ops.aten.mul.Tensor(squeeze_151, primals_152);  primals_152 = None
        unsqueeze_245: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_394, 0);  mul_394 = None
        unsqueeze_246: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_245, 2);  unsqueeze_245 = None
        unsqueeze_247: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_246, 3);  unsqueeze_246 = None
        mul_395: f32[64, 512, 1, 1] = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_244);  sub_61 = unsqueeze_244 = None
        sub_63: f32[64, 512, 1, 1] = torch.ops.aten.sub.Tensor(where_2, mul_395);  where_2 = mul_395 = None
        sub_64: f32[64, 512, 1, 1] = torch.ops.aten.sub.Tensor(sub_63, unsqueeze_241);  sub_63 = unsqueeze_241 = None
        mul_396: f32[64, 512, 1, 1] = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_247);  sub_64 = unsqueeze_247 = None
        mul_397: f32[512] = torch.ops.aten.mul.Tensor(sum_7, squeeze_151);  sum_7 = squeeze_151 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_396, relu_45, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_396 = primals_151 = None
        getitem_114: f32[64, 2048, 1, 1] = convolution_backward_2[0]
        getitem_115: f32[512, 2048, 1, 1] = convolution_backward_2[1];  convolution_backward_2 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        add_281: f32[64, 2048, 1, 1] = torch.ops.aten.add.Tensor(where, getitem_114);  where = getitem_114 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:161, code: out = self.relu(out)
        le_3: b8[64, 2048, 1, 1] = torch.ops.aten.le.Scalar(relu_45, 0);  relu_45 = None
        where_3: f32[64, 2048, 1, 1] = torch.ops.aten.where.self(le_3, scalar_tensor, add_281);  le_3 = add_281 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:155, code: out = self.bn3(out)
        sum_8: f32[2048] = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
        sub_65: f32[64, 2048, 1, 1] = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_250);  convolution_49 = unsqueeze_250 = None
        mul_398: f32[64, 2048, 1, 1] = torch.ops.aten.mul.Tensor(where_3, sub_65)
        sum_9: f32[2048] = torch.ops.aten.sum.dim_IntList(mul_398, [0, 2, 3]);  mul_398 = None
        mul_399: f32[2048] = torch.ops.aten.mul.Tensor(sum_8, 0.015625)
        unsqueeze_251: f32[1, 2048] = torch.ops.aten.unsqueeze.default(mul_399, 0);  mul_399 = None
        unsqueeze_252: f32[1, 2048, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
        unsqueeze_253: f32[1, 2048, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_252, 3);  unsqueeze_252 = None
        mul_400: f32[2048] = torch.ops.aten.mul.Tensor(sum_9, 0.015625)
        mul_401: f32[2048] = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
        mul_402: f32[2048] = torch.ops.aten.mul.Tensor(mul_400, mul_401);  mul_400 = mul_401 = None
        unsqueeze_254: f32[1, 2048] = torch.ops.aten.unsqueeze.default(mul_402, 0);  mul_402 = None
        unsqueeze_255: f32[1, 2048, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_254, 2);  unsqueeze_254 = None
        unsqueeze_256: f32[1, 2048, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_255, 3);  unsqueeze_255 = None
        mul_403: f32[2048] = torch.ops.aten.mul.Tensor(squeeze_148, primals_149);  primals_149 = None
        unsqueeze_257: f32[1, 2048] = torch.ops.aten.unsqueeze.default(mul_403, 0);  mul_403 = None
        unsqueeze_258: f32[1, 2048, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
        unsqueeze_259: f32[1, 2048, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_258, 3);  unsqueeze_258 = None
        mul_404: f32[64, 2048, 1, 1] = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_256);  sub_65 = unsqueeze_256 = None
        sub_67: f32[64, 2048, 1, 1] = torch.ops.aten.sub.Tensor(where_3, mul_404);  mul_404 = None
        sub_68: f32[64, 2048, 1, 1] = torch.ops.aten.sub.Tensor(sub_67, unsqueeze_253);  sub_67 = unsqueeze_253 = None
        mul_405: f32[64, 2048, 1, 1] = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_259);  sub_68 = unsqueeze_259 = None
        mul_406: f32[2048] = torch.ops.aten.mul.Tensor(sum_9, squeeze_148);  sum_9 = squeeze_148 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:154, code: out = self.conv3(out)
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_405, relu_44, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_405 = primals_148 = None
        getitem_117: f32[64, 512, 1, 1] = convolution_backward_3[0]
        getitem_118: f32[2048, 512, 1, 1] = convolution_backward_3[1];  convolution_backward_3 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:152, code: out = self.relu(out)
        le_4: b8[64, 512, 1, 1] = torch.ops.aten.le.Scalar(relu_44, 0);  relu_44 = None
        where_4: f32[64, 512, 1, 1] = torch.ops.aten.where.self(le_4, scalar_tensor, getitem_117);  le_4 = getitem_117 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:151, code: out = self.bn2(out)
        sum_10: f32[512] = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        sub_69: f32[64, 512, 1, 1] = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_262);  convolution_48 = unsqueeze_262 = None
        mul_407: f32[64, 512, 1, 1] = torch.ops.aten.mul.Tensor(where_4, sub_69)
        sum_11: f32[512] = torch.ops.aten.sum.dim_IntList(mul_407, [0, 2, 3]);  mul_407 = None
        mul_408: f32[512] = torch.ops.aten.mul.Tensor(sum_10, 0.015625)
        unsqueeze_263: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_408, 0);  mul_408 = None
        unsqueeze_264: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
        unsqueeze_265: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
        mul_409: f32[512] = torch.ops.aten.mul.Tensor(sum_11, 0.015625)
        mul_410: f32[512] = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
        mul_411: f32[512] = torch.ops.aten.mul.Tensor(mul_409, mul_410);  mul_409 = mul_410 = None
        unsqueeze_266: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_411, 0);  mul_411 = None
        unsqueeze_267: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
        unsqueeze_268: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
        mul_412: f32[512] = torch.ops.aten.mul.Tensor(squeeze_145, primals_146);  primals_146 = None
        unsqueeze_269: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_412, 0);  mul_412 = None
        unsqueeze_270: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
        unsqueeze_271: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
        mul_413: f32[64, 512, 1, 1] = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_268);  sub_69 = unsqueeze_268 = None
        sub_71: f32[64, 512, 1, 1] = torch.ops.aten.sub.Tensor(where_4, mul_413);  where_4 = mul_413 = None
        sub_72: f32[64, 512, 1, 1] = torch.ops.aten.sub.Tensor(sub_71, unsqueeze_265);  sub_71 = unsqueeze_265 = None
        mul_414: f32[64, 512, 1, 1] = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_271);  sub_72 = unsqueeze_271 = None
        mul_415: f32[512] = torch.ops.aten.mul.Tensor(sum_11, squeeze_145);  sum_11 = squeeze_145 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:150, code: out = self.conv2(out)
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_414, relu_43, primals_145, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_414 = primals_145 = None
        getitem_120: f32[64, 512, 1, 1] = convolution_backward_4[0]
        getitem_121: f32[512, 512, 3, 3] = convolution_backward_4[1];  convolution_backward_4 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:148, code: out = self.relu(out)
        le_5: b8[64, 512, 1, 1] = torch.ops.aten.le.Scalar(relu_43, 0);  relu_43 = None
        where_5: f32[64, 512, 1, 1] = torch.ops.aten.where.self(le_5, scalar_tensor, getitem_120);  le_5 = getitem_120 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:147, code: out = self.bn1(out)
        sum_12: f32[512] = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        sub_73: f32[64, 512, 1, 1] = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_274);  convolution_47 = unsqueeze_274 = None
        mul_416: f32[64, 512, 1, 1] = torch.ops.aten.mul.Tensor(where_5, sub_73)
        sum_13: f32[512] = torch.ops.aten.sum.dim_IntList(mul_416, [0, 2, 3]);  mul_416 = None
        mul_417: f32[512] = torch.ops.aten.mul.Tensor(sum_12, 0.015625)
        unsqueeze_275: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_417, 0);  mul_417 = None
        unsqueeze_276: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
        unsqueeze_277: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
        mul_418: f32[512] = torch.ops.aten.mul.Tensor(sum_13, 0.015625)
        mul_419: f32[512] = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
        mul_420: f32[512] = torch.ops.aten.mul.Tensor(mul_418, mul_419);  mul_418 = mul_419 = None
        unsqueeze_278: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_420, 0);  mul_420 = None
        unsqueeze_279: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
        unsqueeze_280: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
        mul_421: f32[512] = torch.ops.aten.mul.Tensor(squeeze_142, primals_143);  primals_143 = None
        unsqueeze_281: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_421, 0);  mul_421 = None
        unsqueeze_282: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
        unsqueeze_283: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
        mul_422: f32[64, 512, 1, 1] = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_280);  sub_73 = unsqueeze_280 = None
        sub_75: f32[64, 512, 1, 1] = torch.ops.aten.sub.Tensor(where_5, mul_422);  where_5 = mul_422 = None
        sub_76: f32[64, 512, 1, 1] = torch.ops.aten.sub.Tensor(sub_75, unsqueeze_277);  sub_75 = unsqueeze_277 = None
        mul_423: f32[64, 512, 1, 1] = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_283);  sub_76 = unsqueeze_283 = None
        mul_424: f32[512] = torch.ops.aten.mul.Tensor(sum_13, squeeze_142);  sum_13 = squeeze_142 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_423, relu_42, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_423 = primals_142 = None
        getitem_123: f32[64, 2048, 1, 1] = convolution_backward_5[0]
        getitem_124: f32[512, 2048, 1, 1] = convolution_backward_5[1];  convolution_backward_5 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        add_282: f32[64, 2048, 1, 1] = torch.ops.aten.add.Tensor(where_3, getitem_123);  where_3 = getitem_123 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:161, code: out = self.relu(out)
        le_6: b8[64, 2048, 1, 1] = torch.ops.aten.le.Scalar(relu_42, 0);  relu_42 = None
        where_6: f32[64, 2048, 1, 1] = torch.ops.aten.where.self(le_6, scalar_tensor, add_282);  le_6 = add_282 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:158, code: identity = self.downsample(x)
        sum_14: f32[2048] = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        sub_77: f32[64, 2048, 1, 1] = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_286);  convolution_46 = unsqueeze_286 = None
        mul_425: f32[64, 2048, 1, 1] = torch.ops.aten.mul.Tensor(where_6, sub_77)
        sum_15: f32[2048] = torch.ops.aten.sum.dim_IntList(mul_425, [0, 2, 3]);  mul_425 = None
        mul_426: f32[2048] = torch.ops.aten.mul.Tensor(sum_14, 0.015625)
        unsqueeze_287: f32[1, 2048] = torch.ops.aten.unsqueeze.default(mul_426, 0);  mul_426 = None
        unsqueeze_288: f32[1, 2048, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
        unsqueeze_289: f32[1, 2048, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
        mul_427: f32[2048] = torch.ops.aten.mul.Tensor(sum_15, 0.015625)
        mul_428: f32[2048] = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
        mul_429: f32[2048] = torch.ops.aten.mul.Tensor(mul_427, mul_428);  mul_427 = mul_428 = None
        unsqueeze_290: f32[1, 2048] = torch.ops.aten.unsqueeze.default(mul_429, 0);  mul_429 = None
        unsqueeze_291: f32[1, 2048, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
        unsqueeze_292: f32[1, 2048, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
        mul_430: f32[2048] = torch.ops.aten.mul.Tensor(squeeze_139, primals_140);  primals_140 = None
        unsqueeze_293: f32[1, 2048] = torch.ops.aten.unsqueeze.default(mul_430, 0);  mul_430 = None
        unsqueeze_294: f32[1, 2048, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
        unsqueeze_295: f32[1, 2048, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
        mul_431: f32[64, 2048, 1, 1] = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_292);  sub_77 = unsqueeze_292 = None
        sub_79: f32[64, 2048, 1, 1] = torch.ops.aten.sub.Tensor(where_6, mul_431);  mul_431 = None
        sub_80: f32[64, 2048, 1, 1] = torch.ops.aten.sub.Tensor(sub_79, unsqueeze_289);  sub_79 = None
        mul_432: f32[64, 2048, 1, 1] = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_295);  sub_80 = unsqueeze_295 = None
        mul_433: f32[2048] = torch.ops.aten.mul.Tensor(sum_15, squeeze_139);  sum_15 = squeeze_139 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_432, relu_39, primals_139, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_432 = primals_139 = None
        getitem_126: f32[64, 1024, 2, 2] = convolution_backward_6[0]
        getitem_127: f32[2048, 1024, 1, 1] = convolution_backward_6[1];  convolution_backward_6 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:155, code: out = self.bn3(out)
        sub_81: f32[64, 2048, 1, 1] = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_298);  convolution_45 = unsqueeze_298 = None
        mul_434: f32[64, 2048, 1, 1] = torch.ops.aten.mul.Tensor(where_6, sub_81)
        sum_17: f32[2048] = torch.ops.aten.sum.dim_IntList(mul_434, [0, 2, 3]);  mul_434 = None
        mul_436: f32[2048] = torch.ops.aten.mul.Tensor(sum_17, 0.015625)
        mul_437: f32[2048] = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
        mul_438: f32[2048] = torch.ops.aten.mul.Tensor(mul_436, mul_437);  mul_436 = mul_437 = None
        unsqueeze_302: f32[1, 2048] = torch.ops.aten.unsqueeze.default(mul_438, 0);  mul_438 = None
        unsqueeze_303: f32[1, 2048, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
        unsqueeze_304: f32[1, 2048, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
        mul_439: f32[2048] = torch.ops.aten.mul.Tensor(squeeze_136, primals_137);  primals_137 = None
        unsqueeze_305: f32[1, 2048] = torch.ops.aten.unsqueeze.default(mul_439, 0);  mul_439 = None
        unsqueeze_306: f32[1, 2048, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
        unsqueeze_307: f32[1, 2048, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
        mul_440: f32[64, 2048, 1, 1] = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_304);  sub_81 = unsqueeze_304 = None
        sub_83: f32[64, 2048, 1, 1] = torch.ops.aten.sub.Tensor(where_6, mul_440);  where_6 = mul_440 = None
        sub_84: f32[64, 2048, 1, 1] = torch.ops.aten.sub.Tensor(sub_83, unsqueeze_289);  sub_83 = unsqueeze_289 = None
        mul_441: f32[64, 2048, 1, 1] = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_307);  sub_84 = unsqueeze_307 = None
        mul_442: f32[2048] = torch.ops.aten.mul.Tensor(sum_17, squeeze_136);  sum_17 = squeeze_136 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:154, code: out = self.conv3(out)
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_441, relu_41, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_441 = primals_136 = None
        getitem_129: f32[64, 512, 1, 1] = convolution_backward_7[0]
        getitem_130: f32[2048, 512, 1, 1] = convolution_backward_7[1];  convolution_backward_7 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:152, code: out = self.relu(out)
        le_7: b8[64, 512, 1, 1] = torch.ops.aten.le.Scalar(relu_41, 0);  relu_41 = None
        where_7: f32[64, 512, 1, 1] = torch.ops.aten.where.self(le_7, scalar_tensor, getitem_129);  le_7 = getitem_129 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:151, code: out = self.bn2(out)
        sum_18: f32[512] = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
        sub_85: f32[64, 512, 1, 1] = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_310);  convolution_44 = unsqueeze_310 = None
        mul_443: f32[64, 512, 1, 1] = torch.ops.aten.mul.Tensor(where_7, sub_85)
        sum_19: f32[512] = torch.ops.aten.sum.dim_IntList(mul_443, [0, 2, 3]);  mul_443 = None
        mul_444: f32[512] = torch.ops.aten.mul.Tensor(sum_18, 0.015625)
        unsqueeze_311: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_444, 0);  mul_444 = None
        unsqueeze_312: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
        unsqueeze_313: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
        mul_445: f32[512] = torch.ops.aten.mul.Tensor(sum_19, 0.015625)
        mul_446: f32[512] = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
        mul_447: f32[512] = torch.ops.aten.mul.Tensor(mul_445, mul_446);  mul_445 = mul_446 = None
        unsqueeze_314: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_447, 0);  mul_447 = None
        unsqueeze_315: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
        unsqueeze_316: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
        mul_448: f32[512] = torch.ops.aten.mul.Tensor(squeeze_133, primals_134);  primals_134 = None
        unsqueeze_317: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_448, 0);  mul_448 = None
        unsqueeze_318: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
        unsqueeze_319: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
        mul_449: f32[64, 512, 1, 1] = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_316);  sub_85 = unsqueeze_316 = None
        sub_87: f32[64, 512, 1, 1] = torch.ops.aten.sub.Tensor(where_7, mul_449);  where_7 = mul_449 = None
        sub_88: f32[64, 512, 1, 1] = torch.ops.aten.sub.Tensor(sub_87, unsqueeze_313);  sub_87 = unsqueeze_313 = None
        mul_450: f32[64, 512, 1, 1] = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_319);  sub_88 = unsqueeze_319 = None
        mul_451: f32[512] = torch.ops.aten.mul.Tensor(sum_19, squeeze_133);  sum_19 = squeeze_133 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:150, code: out = self.conv2(out)
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_450, relu_40, primals_133, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_450 = primals_133 = None
        getitem_132: f32[64, 512, 2, 2] = convolution_backward_8[0]
        getitem_133: f32[512, 512, 3, 3] = convolution_backward_8[1];  convolution_backward_8 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:148, code: out = self.relu(out)
        le_8: b8[64, 512, 2, 2] = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None
        where_8: f32[64, 512, 2, 2] = torch.ops.aten.where.self(le_8, scalar_tensor, getitem_132);  le_8 = getitem_132 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:147, code: out = self.bn1(out)
        sum_20: f32[512] = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
        sub_89: f32[64, 512, 2, 2] = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_322);  convolution_43 = unsqueeze_322 = None
        mul_452: f32[64, 512, 2, 2] = torch.ops.aten.mul.Tensor(where_8, sub_89)
        sum_21: f32[512] = torch.ops.aten.sum.dim_IntList(mul_452, [0, 2, 3]);  mul_452 = None
        mul_453: f32[512] = torch.ops.aten.mul.Tensor(sum_20, 0.00390625)
        unsqueeze_323: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_453, 0);  mul_453 = None
        unsqueeze_324: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
        unsqueeze_325: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
        mul_454: f32[512] = torch.ops.aten.mul.Tensor(sum_21, 0.00390625)
        mul_455: f32[512] = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
        mul_456: f32[512] = torch.ops.aten.mul.Tensor(mul_454, mul_455);  mul_454 = mul_455 = None
        unsqueeze_326: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_456, 0);  mul_456 = None
        unsqueeze_327: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
        unsqueeze_328: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
        mul_457: f32[512] = torch.ops.aten.mul.Tensor(squeeze_130, primals_131);  primals_131 = None
        unsqueeze_329: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_457, 0);  mul_457 = None
        unsqueeze_330: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
        unsqueeze_331: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
        mul_458: f32[64, 512, 2, 2] = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_328);  sub_89 = unsqueeze_328 = None
        sub_91: f32[64, 512, 2, 2] = torch.ops.aten.sub.Tensor(where_8, mul_458);  where_8 = mul_458 = None
        sub_92: f32[64, 512, 2, 2] = torch.ops.aten.sub.Tensor(sub_91, unsqueeze_325);  sub_91 = unsqueeze_325 = None
        mul_459: f32[64, 512, 2, 2] = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_331);  sub_92 = unsqueeze_331 = None
        mul_460: f32[512] = torch.ops.aten.mul.Tensor(sum_21, squeeze_130);  sum_21 = squeeze_130 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_459, relu_39, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_459 = primals_130 = None
        getitem_135: f32[64, 1024, 2, 2] = convolution_backward_9[0]
        getitem_136: f32[512, 1024, 1, 1] = convolution_backward_9[1];  convolution_backward_9 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        add_283: f32[64, 1024, 2, 2] = torch.ops.aten.add.Tensor(getitem_126, getitem_135);  getitem_126 = getitem_135 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:161, code: out = self.relu(out)
        le_9: b8[64, 1024, 2, 2] = torch.ops.aten.le.Scalar(relu_39, 0);  relu_39 = None
        where_9: f32[64, 1024, 2, 2] = torch.ops.aten.where.self(le_9, scalar_tensor, add_283);  le_9 = add_283 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:155, code: out = self.bn3(out)
        sum_22: f32[1024] = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
        sub_93: f32[64, 1024, 2, 2] = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_334);  convolution_42 = unsqueeze_334 = None
        mul_461: f32[64, 1024, 2, 2] = torch.ops.aten.mul.Tensor(where_9, sub_93)
        sum_23: f32[1024] = torch.ops.aten.sum.dim_IntList(mul_461, [0, 2, 3]);  mul_461 = None
        mul_462: f32[1024] = torch.ops.aten.mul.Tensor(sum_22, 0.00390625)
        unsqueeze_335: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_462, 0);  mul_462 = None
        unsqueeze_336: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
        unsqueeze_337: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
        mul_463: f32[1024] = torch.ops.aten.mul.Tensor(sum_23, 0.00390625)
        mul_464: f32[1024] = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
        mul_465: f32[1024] = torch.ops.aten.mul.Tensor(mul_463, mul_464);  mul_463 = mul_464 = None
        unsqueeze_338: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_465, 0);  mul_465 = None
        unsqueeze_339: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
        unsqueeze_340: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
        mul_466: f32[1024] = torch.ops.aten.mul.Tensor(squeeze_127, primals_128);  primals_128 = None
        unsqueeze_341: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_466, 0);  mul_466 = None
        unsqueeze_342: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
        unsqueeze_343: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
        mul_467: f32[64, 1024, 2, 2] = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_340);  sub_93 = unsqueeze_340 = None
        sub_95: f32[64, 1024, 2, 2] = torch.ops.aten.sub.Tensor(where_9, mul_467);  mul_467 = None
        sub_96: f32[64, 1024, 2, 2] = torch.ops.aten.sub.Tensor(sub_95, unsqueeze_337);  sub_95 = unsqueeze_337 = None
        mul_468: f32[64, 1024, 2, 2] = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_343);  sub_96 = unsqueeze_343 = None
        mul_469: f32[1024] = torch.ops.aten.mul.Tensor(sum_23, squeeze_127);  sum_23 = squeeze_127 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:154, code: out = self.conv3(out)
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_468, relu_38, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_468 = primals_127 = None
        getitem_138: f32[64, 256, 2, 2] = convolution_backward_10[0]
        getitem_139: f32[1024, 256, 1, 1] = convolution_backward_10[1];  convolution_backward_10 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:152, code: out = self.relu(out)
        le_10: b8[64, 256, 2, 2] = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
        where_10: f32[64, 256, 2, 2] = torch.ops.aten.where.self(le_10, scalar_tensor, getitem_138);  le_10 = getitem_138 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:151, code: out = self.bn2(out)
        sum_24: f32[256] = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        sub_97: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_346);  convolution_41 = unsqueeze_346 = None
        mul_470: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(where_10, sub_97)
        sum_25: f32[256] = torch.ops.aten.sum.dim_IntList(mul_470, [0, 2, 3]);  mul_470 = None
        mul_471: f32[256] = torch.ops.aten.mul.Tensor(sum_24, 0.00390625)
        unsqueeze_347: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_471, 0);  mul_471 = None
        unsqueeze_348: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
        unsqueeze_349: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
        mul_472: f32[256] = torch.ops.aten.mul.Tensor(sum_25, 0.00390625)
        mul_473: f32[256] = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
        mul_474: f32[256] = torch.ops.aten.mul.Tensor(mul_472, mul_473);  mul_472 = mul_473 = None
        unsqueeze_350: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_474, 0);  mul_474 = None
        unsqueeze_351: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
        unsqueeze_352: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
        mul_475: f32[256] = torch.ops.aten.mul.Tensor(squeeze_124, primals_125);  primals_125 = None
        unsqueeze_353: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_475, 0);  mul_475 = None
        unsqueeze_354: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
        unsqueeze_355: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
        mul_476: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_352);  sub_97 = unsqueeze_352 = None
        sub_99: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(where_10, mul_476);  where_10 = mul_476 = None
        sub_100: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(sub_99, unsqueeze_349);  sub_99 = unsqueeze_349 = None
        mul_477: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_355);  sub_100 = unsqueeze_355 = None
        mul_478: f32[256] = torch.ops.aten.mul.Tensor(sum_25, squeeze_124);  sum_25 = squeeze_124 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:150, code: out = self.conv2(out)
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_477, relu_37, primals_124, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_477 = primals_124 = None
        getitem_141: f32[64, 256, 2, 2] = convolution_backward_11[0]
        getitem_142: f32[256, 256, 3, 3] = convolution_backward_11[1];  convolution_backward_11 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:148, code: out = self.relu(out)
        le_11: b8[64, 256, 2, 2] = torch.ops.aten.le.Scalar(relu_37, 0);  relu_37 = None
        where_11: f32[64, 256, 2, 2] = torch.ops.aten.where.self(le_11, scalar_tensor, getitem_141);  le_11 = getitem_141 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:147, code: out = self.bn1(out)
        sum_26: f32[256] = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
        sub_101: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_358);  convolution_40 = unsqueeze_358 = None
        mul_479: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(where_11, sub_101)
        sum_27: f32[256] = torch.ops.aten.sum.dim_IntList(mul_479, [0, 2, 3]);  mul_479 = None
        mul_480: f32[256] = torch.ops.aten.mul.Tensor(sum_26, 0.00390625)
        unsqueeze_359: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_480, 0);  mul_480 = None
        unsqueeze_360: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
        unsqueeze_361: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_360, 3);  unsqueeze_360 = None
        mul_481: f32[256] = torch.ops.aten.mul.Tensor(sum_27, 0.00390625)
        mul_482: f32[256] = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
        mul_483: f32[256] = torch.ops.aten.mul.Tensor(mul_481, mul_482);  mul_481 = mul_482 = None
        unsqueeze_362: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_483, 0);  mul_483 = None
        unsqueeze_363: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_362, 2);  unsqueeze_362 = None
        unsqueeze_364: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_363, 3);  unsqueeze_363 = None
        mul_484: f32[256] = torch.ops.aten.mul.Tensor(squeeze_121, primals_122);  primals_122 = None
        unsqueeze_365: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_484, 0);  mul_484 = None
        unsqueeze_366: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
        unsqueeze_367: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_366, 3);  unsqueeze_366 = None
        mul_485: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_364);  sub_101 = unsqueeze_364 = None
        sub_103: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(where_11, mul_485);  where_11 = mul_485 = None
        sub_104: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(sub_103, unsqueeze_361);  sub_103 = unsqueeze_361 = None
        mul_486: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_367);  sub_104 = unsqueeze_367 = None
        mul_487: f32[256] = torch.ops.aten.mul.Tensor(sum_27, squeeze_121);  sum_27 = squeeze_121 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_486, relu_36, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_486 = primals_121 = None
        getitem_144: f32[64, 1024, 2, 2] = convolution_backward_12[0]
        getitem_145: f32[256, 1024, 1, 1] = convolution_backward_12[1];  convolution_backward_12 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        add_284: f32[64, 1024, 2, 2] = torch.ops.aten.add.Tensor(where_9, getitem_144);  where_9 = getitem_144 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:161, code: out = self.relu(out)
        le_12: b8[64, 1024, 2, 2] = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None
        where_12: f32[64, 1024, 2, 2] = torch.ops.aten.where.self(le_12, scalar_tensor, add_284);  le_12 = add_284 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:155, code: out = self.bn3(out)
        sum_28: f32[1024] = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
        sub_105: f32[64, 1024, 2, 2] = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_370);  convolution_39 = unsqueeze_370 = None
        mul_488: f32[64, 1024, 2, 2] = torch.ops.aten.mul.Tensor(where_12, sub_105)
        sum_29: f32[1024] = torch.ops.aten.sum.dim_IntList(mul_488, [0, 2, 3]);  mul_488 = None
        mul_489: f32[1024] = torch.ops.aten.mul.Tensor(sum_28, 0.00390625)
        unsqueeze_371: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_489, 0);  mul_489 = None
        unsqueeze_372: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
        unsqueeze_373: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_372, 3);  unsqueeze_372 = None
        mul_490: f32[1024] = torch.ops.aten.mul.Tensor(sum_29, 0.00390625)
        mul_491: f32[1024] = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
        mul_492: f32[1024] = torch.ops.aten.mul.Tensor(mul_490, mul_491);  mul_490 = mul_491 = None
        unsqueeze_374: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_492, 0);  mul_492 = None
        unsqueeze_375: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_374, 2);  unsqueeze_374 = None
        unsqueeze_376: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_375, 3);  unsqueeze_375 = None
        mul_493: f32[1024] = torch.ops.aten.mul.Tensor(squeeze_118, primals_119);  primals_119 = None
        unsqueeze_377: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_493, 0);  mul_493 = None
        unsqueeze_378: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
        unsqueeze_379: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_378, 3);  unsqueeze_378 = None
        mul_494: f32[64, 1024, 2, 2] = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_376);  sub_105 = unsqueeze_376 = None
        sub_107: f32[64, 1024, 2, 2] = torch.ops.aten.sub.Tensor(where_12, mul_494);  mul_494 = None
        sub_108: f32[64, 1024, 2, 2] = torch.ops.aten.sub.Tensor(sub_107, unsqueeze_373);  sub_107 = unsqueeze_373 = None
        mul_495: f32[64, 1024, 2, 2] = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_379);  sub_108 = unsqueeze_379 = None
        mul_496: f32[1024] = torch.ops.aten.mul.Tensor(sum_29, squeeze_118);  sum_29 = squeeze_118 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:154, code: out = self.conv3(out)
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_495, relu_35, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_495 = primals_118 = None
        getitem_147: f32[64, 256, 2, 2] = convolution_backward_13[0]
        getitem_148: f32[1024, 256, 1, 1] = convolution_backward_13[1];  convolution_backward_13 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:152, code: out = self.relu(out)
        le_13: b8[64, 256, 2, 2] = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
        where_13: f32[64, 256, 2, 2] = torch.ops.aten.where.self(le_13, scalar_tensor, getitem_147);  le_13 = getitem_147 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:151, code: out = self.bn2(out)
        sum_30: f32[256] = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
        sub_109: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_382);  convolution_38 = unsqueeze_382 = None
        mul_497: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(where_13, sub_109)
        sum_31: f32[256] = torch.ops.aten.sum.dim_IntList(mul_497, [0, 2, 3]);  mul_497 = None
        mul_498: f32[256] = torch.ops.aten.mul.Tensor(sum_30, 0.00390625)
        unsqueeze_383: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_498, 0);  mul_498 = None
        unsqueeze_384: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
        unsqueeze_385: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_384, 3);  unsqueeze_384 = None
        mul_499: f32[256] = torch.ops.aten.mul.Tensor(sum_31, 0.00390625)
        mul_500: f32[256] = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
        mul_501: f32[256] = torch.ops.aten.mul.Tensor(mul_499, mul_500);  mul_499 = mul_500 = None
        unsqueeze_386: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_501, 0);  mul_501 = None
        unsqueeze_387: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_386, 2);  unsqueeze_386 = None
        unsqueeze_388: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_387, 3);  unsqueeze_387 = None
        mul_502: f32[256] = torch.ops.aten.mul.Tensor(squeeze_115, primals_116);  primals_116 = None
        unsqueeze_389: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_502, 0);  mul_502 = None
        unsqueeze_390: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
        unsqueeze_391: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_390, 3);  unsqueeze_390 = None
        mul_503: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_388);  sub_109 = unsqueeze_388 = None
        sub_111: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(where_13, mul_503);  where_13 = mul_503 = None
        sub_112: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(sub_111, unsqueeze_385);  sub_111 = unsqueeze_385 = None
        mul_504: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_391);  sub_112 = unsqueeze_391 = None
        mul_505: f32[256] = torch.ops.aten.mul.Tensor(sum_31, squeeze_115);  sum_31 = squeeze_115 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:150, code: out = self.conv2(out)
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_504, relu_34, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_504 = primals_115 = None
        getitem_150: f32[64, 256, 2, 2] = convolution_backward_14[0]
        getitem_151: f32[256, 256, 3, 3] = convolution_backward_14[1];  convolution_backward_14 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:148, code: out = self.relu(out)
        le_14: b8[64, 256, 2, 2] = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
        where_14: f32[64, 256, 2, 2] = torch.ops.aten.where.self(le_14, scalar_tensor, getitem_150);  le_14 = getitem_150 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:147, code: out = self.bn1(out)
        sum_32: f32[256] = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
        sub_113: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_394);  convolution_37 = unsqueeze_394 = None
        mul_506: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(where_14, sub_113)
        sum_33: f32[256] = torch.ops.aten.sum.dim_IntList(mul_506, [0, 2, 3]);  mul_506 = None
        mul_507: f32[256] = torch.ops.aten.mul.Tensor(sum_32, 0.00390625)
        unsqueeze_395: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_507, 0);  mul_507 = None
        unsqueeze_396: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
        unsqueeze_397: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_396, 3);  unsqueeze_396 = None
        mul_508: f32[256] = torch.ops.aten.mul.Tensor(sum_33, 0.00390625)
        mul_509: f32[256] = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
        mul_510: f32[256] = torch.ops.aten.mul.Tensor(mul_508, mul_509);  mul_508 = mul_509 = None
        unsqueeze_398: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_510, 0);  mul_510 = None
        unsqueeze_399: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_398, 2);  unsqueeze_398 = None
        unsqueeze_400: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_399, 3);  unsqueeze_399 = None
        mul_511: f32[256] = torch.ops.aten.mul.Tensor(squeeze_112, primals_113);  primals_113 = None
        unsqueeze_401: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_511, 0);  mul_511 = None
        unsqueeze_402: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
        unsqueeze_403: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_402, 3);  unsqueeze_402 = None
        mul_512: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_400);  sub_113 = unsqueeze_400 = None
        sub_115: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(where_14, mul_512);  where_14 = mul_512 = None
        sub_116: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(sub_115, unsqueeze_397);  sub_115 = unsqueeze_397 = None
        mul_513: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_403);  sub_116 = unsqueeze_403 = None
        mul_514: f32[256] = torch.ops.aten.mul.Tensor(sum_33, squeeze_112);  sum_33 = squeeze_112 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_513, relu_33, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_513 = primals_112 = None
        getitem_153: f32[64, 1024, 2, 2] = convolution_backward_15[0]
        getitem_154: f32[256, 1024, 1, 1] = convolution_backward_15[1];  convolution_backward_15 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        add_285: f32[64, 1024, 2, 2] = torch.ops.aten.add.Tensor(where_12, getitem_153);  where_12 = getitem_153 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:161, code: out = self.relu(out)
        le_15: b8[64, 1024, 2, 2] = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
        where_15: f32[64, 1024, 2, 2] = torch.ops.aten.where.self(le_15, scalar_tensor, add_285);  le_15 = add_285 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:155, code: out = self.bn3(out)
        sum_34: f32[1024] = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
        sub_117: f32[64, 1024, 2, 2] = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_406);  convolution_36 = unsqueeze_406 = None
        mul_515: f32[64, 1024, 2, 2] = torch.ops.aten.mul.Tensor(where_15, sub_117)
        sum_35: f32[1024] = torch.ops.aten.sum.dim_IntList(mul_515, [0, 2, 3]);  mul_515 = None
        mul_516: f32[1024] = torch.ops.aten.mul.Tensor(sum_34, 0.00390625)
        unsqueeze_407: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_516, 0);  mul_516 = None
        unsqueeze_408: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
        unsqueeze_409: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_408, 3);  unsqueeze_408 = None
        mul_517: f32[1024] = torch.ops.aten.mul.Tensor(sum_35, 0.00390625)
        mul_518: f32[1024] = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_519: f32[1024] = torch.ops.aten.mul.Tensor(mul_517, mul_518);  mul_517 = mul_518 = None
        unsqueeze_410: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_519, 0);  mul_519 = None
        unsqueeze_411: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_410, 2);  unsqueeze_410 = None
        unsqueeze_412: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_411, 3);  unsqueeze_411 = None
        mul_520: f32[1024] = torch.ops.aten.mul.Tensor(squeeze_109, primals_110);  primals_110 = None
        unsqueeze_413: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
        unsqueeze_414: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
        unsqueeze_415: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_414, 3);  unsqueeze_414 = None
        mul_521: f32[64, 1024, 2, 2] = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_412);  sub_117 = unsqueeze_412 = None
        sub_119: f32[64, 1024, 2, 2] = torch.ops.aten.sub.Tensor(where_15, mul_521);  mul_521 = None
        sub_120: f32[64, 1024, 2, 2] = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_409);  sub_119 = unsqueeze_409 = None
        mul_522: f32[64, 1024, 2, 2] = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_415);  sub_120 = unsqueeze_415 = None
        mul_523: f32[1024] = torch.ops.aten.mul.Tensor(sum_35, squeeze_109);  sum_35 = squeeze_109 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:154, code: out = self.conv3(out)
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_522, relu_32, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_522 = primals_109 = None
        getitem_156: f32[64, 256, 2, 2] = convolution_backward_16[0]
        getitem_157: f32[1024, 256, 1, 1] = convolution_backward_16[1];  convolution_backward_16 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:152, code: out = self.relu(out)
        le_16: b8[64, 256, 2, 2] = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
        where_16: f32[64, 256, 2, 2] = torch.ops.aten.where.self(le_16, scalar_tensor, getitem_156);  le_16 = getitem_156 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:151, code: out = self.bn2(out)
        sum_36: f32[256] = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
        sub_121: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_418);  convolution_35 = unsqueeze_418 = None
        mul_524: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(where_16, sub_121)
        sum_37: f32[256] = torch.ops.aten.sum.dim_IntList(mul_524, [0, 2, 3]);  mul_524 = None
        mul_525: f32[256] = torch.ops.aten.mul.Tensor(sum_36, 0.00390625)
        unsqueeze_419: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_525, 0);  mul_525 = None
        unsqueeze_420: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
        unsqueeze_421: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_420, 3);  unsqueeze_420 = None
        mul_526: f32[256] = torch.ops.aten.mul.Tensor(sum_37, 0.00390625)
        mul_527: f32[256] = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
        mul_528: f32[256] = torch.ops.aten.mul.Tensor(mul_526, mul_527);  mul_526 = mul_527 = None
        unsqueeze_422: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_528, 0);  mul_528 = None
        unsqueeze_423: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_422, 2);  unsqueeze_422 = None
        unsqueeze_424: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_423, 3);  unsqueeze_423 = None
        mul_529: f32[256] = torch.ops.aten.mul.Tensor(squeeze_106, primals_107);  primals_107 = None
        unsqueeze_425: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_529, 0);  mul_529 = None
        unsqueeze_426: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
        unsqueeze_427: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
        mul_530: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_424);  sub_121 = unsqueeze_424 = None
        sub_123: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(where_16, mul_530);  where_16 = mul_530 = None
        sub_124: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(sub_123, unsqueeze_421);  sub_123 = unsqueeze_421 = None
        mul_531: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_427);  sub_124 = unsqueeze_427 = None
        mul_532: f32[256] = torch.ops.aten.mul.Tensor(sum_37, squeeze_106);  sum_37 = squeeze_106 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:150, code: out = self.conv2(out)
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_531, relu_31, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_531 = primals_106 = None
        getitem_159: f32[64, 256, 2, 2] = convolution_backward_17[0]
        getitem_160: f32[256, 256, 3, 3] = convolution_backward_17[1];  convolution_backward_17 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:148, code: out = self.relu(out)
        le_17: b8[64, 256, 2, 2] = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
        where_17: f32[64, 256, 2, 2] = torch.ops.aten.where.self(le_17, scalar_tensor, getitem_159);  le_17 = getitem_159 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:147, code: out = self.bn1(out)
        sum_38: f32[256] = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
        sub_125: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_430);  convolution_34 = unsqueeze_430 = None
        mul_533: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(where_17, sub_125)
        sum_39: f32[256] = torch.ops.aten.sum.dim_IntList(mul_533, [0, 2, 3]);  mul_533 = None
        mul_534: f32[256] = torch.ops.aten.mul.Tensor(sum_38, 0.00390625)
        unsqueeze_431: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_534, 0);  mul_534 = None
        unsqueeze_432: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
        unsqueeze_433: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_432, 3);  unsqueeze_432 = None
        mul_535: f32[256] = torch.ops.aten.mul.Tensor(sum_39, 0.00390625)
        mul_536: f32[256] = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_537: f32[256] = torch.ops.aten.mul.Tensor(mul_535, mul_536);  mul_535 = mul_536 = None
        unsqueeze_434: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_537, 0);  mul_537 = None
        unsqueeze_435: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_434, 2);  unsqueeze_434 = None
        unsqueeze_436: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_435, 3);  unsqueeze_435 = None
        mul_538: f32[256] = torch.ops.aten.mul.Tensor(squeeze_103, primals_104);  primals_104 = None
        unsqueeze_437: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_538, 0);  mul_538 = None
        unsqueeze_438: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
        unsqueeze_439: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
        mul_539: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_436);  sub_125 = unsqueeze_436 = None
        sub_127: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(where_17, mul_539);  where_17 = mul_539 = None
        sub_128: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(sub_127, unsqueeze_433);  sub_127 = unsqueeze_433 = None
        mul_540: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_439);  sub_128 = unsqueeze_439 = None
        mul_541: f32[256] = torch.ops.aten.mul.Tensor(sum_39, squeeze_103);  sum_39 = squeeze_103 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_540, relu_30, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_540 = primals_103 = None
        getitem_162: f32[64, 1024, 2, 2] = convolution_backward_18[0]
        getitem_163: f32[256, 1024, 1, 1] = convolution_backward_18[1];  convolution_backward_18 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        add_286: f32[64, 1024, 2, 2] = torch.ops.aten.add.Tensor(where_15, getitem_162);  where_15 = getitem_162 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:161, code: out = self.relu(out)
        le_18: b8[64, 1024, 2, 2] = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
        where_18: f32[64, 1024, 2, 2] = torch.ops.aten.where.self(le_18, scalar_tensor, add_286);  le_18 = add_286 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:155, code: out = self.bn3(out)
        sum_40: f32[1024] = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
        sub_129: f32[64, 1024, 2, 2] = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_442);  convolution_33 = unsqueeze_442 = None
        mul_542: f32[64, 1024, 2, 2] = torch.ops.aten.mul.Tensor(where_18, sub_129)
        sum_41: f32[1024] = torch.ops.aten.sum.dim_IntList(mul_542, [0, 2, 3]);  mul_542 = None
        mul_543: f32[1024] = torch.ops.aten.mul.Tensor(sum_40, 0.00390625)
        unsqueeze_443: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_543, 0);  mul_543 = None
        unsqueeze_444: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
        unsqueeze_445: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
        mul_544: f32[1024] = torch.ops.aten.mul.Tensor(sum_41, 0.00390625)
        mul_545: f32[1024] = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
        mul_546: f32[1024] = torch.ops.aten.mul.Tensor(mul_544, mul_545);  mul_544 = mul_545 = None
        unsqueeze_446: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_546, 0);  mul_546 = None
        unsqueeze_447: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
        unsqueeze_448: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
        mul_547: f32[1024] = torch.ops.aten.mul.Tensor(squeeze_100, primals_101);  primals_101 = None
        unsqueeze_449: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_547, 0);  mul_547 = None
        unsqueeze_450: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
        unsqueeze_451: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
        mul_548: f32[64, 1024, 2, 2] = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_448);  sub_129 = unsqueeze_448 = None
        sub_131: f32[64, 1024, 2, 2] = torch.ops.aten.sub.Tensor(where_18, mul_548);  mul_548 = None
        sub_132: f32[64, 1024, 2, 2] = torch.ops.aten.sub.Tensor(sub_131, unsqueeze_445);  sub_131 = unsqueeze_445 = None
        mul_549: f32[64, 1024, 2, 2] = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_451);  sub_132 = unsqueeze_451 = None
        mul_550: f32[1024] = torch.ops.aten.mul.Tensor(sum_41, squeeze_100);  sum_41 = squeeze_100 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:154, code: out = self.conv3(out)
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_549, relu_29, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_549 = primals_100 = None
        getitem_165: f32[64, 256, 2, 2] = convolution_backward_19[0]
        getitem_166: f32[1024, 256, 1, 1] = convolution_backward_19[1];  convolution_backward_19 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:152, code: out = self.relu(out)
        le_19: b8[64, 256, 2, 2] = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
        where_19: f32[64, 256, 2, 2] = torch.ops.aten.where.self(le_19, scalar_tensor, getitem_165);  le_19 = getitem_165 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:151, code: out = self.bn2(out)
        sum_42: f32[256] = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
        sub_133: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_454);  convolution_32 = unsqueeze_454 = None
        mul_551: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(where_19, sub_133)
        sum_43: f32[256] = torch.ops.aten.sum.dim_IntList(mul_551, [0, 2, 3]);  mul_551 = None
        mul_552: f32[256] = torch.ops.aten.mul.Tensor(sum_42, 0.00390625)
        unsqueeze_455: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_552, 0);  mul_552 = None
        unsqueeze_456: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
        unsqueeze_457: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
        mul_553: f32[256] = torch.ops.aten.mul.Tensor(sum_43, 0.00390625)
        mul_554: f32[256] = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_555: f32[256] = torch.ops.aten.mul.Tensor(mul_553, mul_554);  mul_553 = mul_554 = None
        unsqueeze_458: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_555, 0);  mul_555 = None
        unsqueeze_459: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
        unsqueeze_460: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
        mul_556: f32[256] = torch.ops.aten.mul.Tensor(squeeze_97, primals_98);  primals_98 = None
        unsqueeze_461: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_556, 0);  mul_556 = None
        unsqueeze_462: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
        unsqueeze_463: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
        mul_557: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_460);  sub_133 = unsqueeze_460 = None
        sub_135: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(where_19, mul_557);  where_19 = mul_557 = None
        sub_136: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_457);  sub_135 = unsqueeze_457 = None
        mul_558: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_463);  sub_136 = unsqueeze_463 = None
        mul_559: f32[256] = torch.ops.aten.mul.Tensor(sum_43, squeeze_97);  sum_43 = squeeze_97 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:150, code: out = self.conv2(out)
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_558, relu_28, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_558 = primals_97 = None
        getitem_168: f32[64, 256, 2, 2] = convolution_backward_20[0]
        getitem_169: f32[256, 256, 3, 3] = convolution_backward_20[1];  convolution_backward_20 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:148, code: out = self.relu(out)
        le_20: b8[64, 256, 2, 2] = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
        where_20: f32[64, 256, 2, 2] = torch.ops.aten.where.self(le_20, scalar_tensor, getitem_168);  le_20 = getitem_168 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:147, code: out = self.bn1(out)
        sum_44: f32[256] = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
        sub_137: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_466);  convolution_31 = unsqueeze_466 = None
        mul_560: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(where_20, sub_137)
        sum_45: f32[256] = torch.ops.aten.sum.dim_IntList(mul_560, [0, 2, 3]);  mul_560 = None
        mul_561: f32[256] = torch.ops.aten.mul.Tensor(sum_44, 0.00390625)
        unsqueeze_467: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_561, 0);  mul_561 = None
        unsqueeze_468: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
        unsqueeze_469: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
        mul_562: f32[256] = torch.ops.aten.mul.Tensor(sum_45, 0.00390625)
        mul_563: f32[256] = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_564: f32[256] = torch.ops.aten.mul.Tensor(mul_562, mul_563);  mul_562 = mul_563 = None
        unsqueeze_470: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_564, 0);  mul_564 = None
        unsqueeze_471: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_470, 2);  unsqueeze_470 = None
        unsqueeze_472: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_471, 3);  unsqueeze_471 = None
        mul_565: f32[256] = torch.ops.aten.mul.Tensor(squeeze_94, primals_95);  primals_95 = None
        unsqueeze_473: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_565, 0);  mul_565 = None
        unsqueeze_474: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
        unsqueeze_475: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
        mul_566: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_472);  sub_137 = unsqueeze_472 = None
        sub_139: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(where_20, mul_566);  where_20 = mul_566 = None
        sub_140: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(sub_139, unsqueeze_469);  sub_139 = unsqueeze_469 = None
        mul_567: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_475);  sub_140 = unsqueeze_475 = None
        mul_568: f32[256] = torch.ops.aten.mul.Tensor(sum_45, squeeze_94);  sum_45 = squeeze_94 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_567, relu_27, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_567 = primals_94 = None
        getitem_171: f32[64, 1024, 2, 2] = convolution_backward_21[0]
        getitem_172: f32[256, 1024, 1, 1] = convolution_backward_21[1];  convolution_backward_21 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        add_287: f32[64, 1024, 2, 2] = torch.ops.aten.add.Tensor(where_18, getitem_171);  where_18 = getitem_171 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:161, code: out = self.relu(out)
        le_21: b8[64, 1024, 2, 2] = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
        where_21: f32[64, 1024, 2, 2] = torch.ops.aten.where.self(le_21, scalar_tensor, add_287);  le_21 = add_287 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:155, code: out = self.bn3(out)
        sum_46: f32[1024] = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
        sub_141: f32[64, 1024, 2, 2] = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_478);  convolution_30 = unsqueeze_478 = None
        mul_569: f32[64, 1024, 2, 2] = torch.ops.aten.mul.Tensor(where_21, sub_141)
        sum_47: f32[1024] = torch.ops.aten.sum.dim_IntList(mul_569, [0, 2, 3]);  mul_569 = None
        mul_570: f32[1024] = torch.ops.aten.mul.Tensor(sum_46, 0.00390625)
        unsqueeze_479: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_570, 0);  mul_570 = None
        unsqueeze_480: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
        unsqueeze_481: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
        mul_571: f32[1024] = torch.ops.aten.mul.Tensor(sum_47, 0.00390625)
        mul_572: f32[1024] = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_573: f32[1024] = torch.ops.aten.mul.Tensor(mul_571, mul_572);  mul_571 = mul_572 = None
        unsqueeze_482: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_573, 0);  mul_573 = None
        unsqueeze_483: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
        unsqueeze_484: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
        mul_574: f32[1024] = torch.ops.aten.mul.Tensor(squeeze_91, primals_92);  primals_92 = None
        unsqueeze_485: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_574, 0);  mul_574 = None
        unsqueeze_486: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
        unsqueeze_487: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
        mul_575: f32[64, 1024, 2, 2] = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_484);  sub_141 = unsqueeze_484 = None
        sub_143: f32[64, 1024, 2, 2] = torch.ops.aten.sub.Tensor(where_21, mul_575);  mul_575 = None
        sub_144: f32[64, 1024, 2, 2] = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_481);  sub_143 = unsqueeze_481 = None
        mul_576: f32[64, 1024, 2, 2] = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_487);  sub_144 = unsqueeze_487 = None
        mul_577: f32[1024] = torch.ops.aten.mul.Tensor(sum_47, squeeze_91);  sum_47 = squeeze_91 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:154, code: out = self.conv3(out)
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_576, relu_26, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_576 = primals_91 = None
        getitem_174: f32[64, 256, 2, 2] = convolution_backward_22[0]
        getitem_175: f32[1024, 256, 1, 1] = convolution_backward_22[1];  convolution_backward_22 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:152, code: out = self.relu(out)
        le_22: b8[64, 256, 2, 2] = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
        where_22: f32[64, 256, 2, 2] = torch.ops.aten.where.self(le_22, scalar_tensor, getitem_174);  le_22 = getitem_174 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:151, code: out = self.bn2(out)
        sum_48: f32[256] = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
        sub_145: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_490);  convolution_29 = unsqueeze_490 = None
        mul_578: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(where_22, sub_145)
        sum_49: f32[256] = torch.ops.aten.sum.dim_IntList(mul_578, [0, 2, 3]);  mul_578 = None
        mul_579: f32[256] = torch.ops.aten.mul.Tensor(sum_48, 0.00390625)
        unsqueeze_491: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_579, 0);  mul_579 = None
        unsqueeze_492: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
        unsqueeze_493: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
        mul_580: f32[256] = torch.ops.aten.mul.Tensor(sum_49, 0.00390625)
        mul_581: f32[256] = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_582: f32[256] = torch.ops.aten.mul.Tensor(mul_580, mul_581);  mul_580 = mul_581 = None
        unsqueeze_494: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_582, 0);  mul_582 = None
        unsqueeze_495: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_494, 2);  unsqueeze_494 = None
        unsqueeze_496: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_495, 3);  unsqueeze_495 = None
        mul_583: f32[256] = torch.ops.aten.mul.Tensor(squeeze_88, primals_89);  primals_89 = None
        unsqueeze_497: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_583, 0);  mul_583 = None
        unsqueeze_498: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
        unsqueeze_499: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
        mul_584: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_496);  sub_145 = unsqueeze_496 = None
        sub_147: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(where_22, mul_584);  where_22 = mul_584 = None
        sub_148: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(sub_147, unsqueeze_493);  sub_147 = unsqueeze_493 = None
        mul_585: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_499);  sub_148 = unsqueeze_499 = None
        mul_586: f32[256] = torch.ops.aten.mul.Tensor(sum_49, squeeze_88);  sum_49 = squeeze_88 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:150, code: out = self.conv2(out)
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_585, relu_25, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_585 = primals_88 = None
        getitem_177: f32[64, 256, 2, 2] = convolution_backward_23[0]
        getitem_178: f32[256, 256, 3, 3] = convolution_backward_23[1];  convolution_backward_23 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:148, code: out = self.relu(out)
        le_23: b8[64, 256, 2, 2] = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
        where_23: f32[64, 256, 2, 2] = torch.ops.aten.where.self(le_23, scalar_tensor, getitem_177);  le_23 = getitem_177 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:147, code: out = self.bn1(out)
        sum_50: f32[256] = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
        sub_149: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_502);  convolution_28 = unsqueeze_502 = None
        mul_587: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(where_23, sub_149)
        sum_51: f32[256] = torch.ops.aten.sum.dim_IntList(mul_587, [0, 2, 3]);  mul_587 = None
        mul_588: f32[256] = torch.ops.aten.mul.Tensor(sum_50, 0.00390625)
        unsqueeze_503: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_588, 0);  mul_588 = None
        unsqueeze_504: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
        unsqueeze_505: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
        mul_589: f32[256] = torch.ops.aten.mul.Tensor(sum_51, 0.00390625)
        mul_590: f32[256] = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_591: f32[256] = torch.ops.aten.mul.Tensor(mul_589, mul_590);  mul_589 = mul_590 = None
        unsqueeze_506: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_591, 0);  mul_591 = None
        unsqueeze_507: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_506, 2);  unsqueeze_506 = None
        unsqueeze_508: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_507, 3);  unsqueeze_507 = None
        mul_592: f32[256] = torch.ops.aten.mul.Tensor(squeeze_85, primals_86);  primals_86 = None
        unsqueeze_509: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_592, 0);  mul_592 = None
        unsqueeze_510: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
        unsqueeze_511: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
        mul_593: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_508);  sub_149 = unsqueeze_508 = None
        sub_151: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(where_23, mul_593);  where_23 = mul_593 = None
        sub_152: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_505);  sub_151 = unsqueeze_505 = None
        mul_594: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_511);  sub_152 = unsqueeze_511 = None
        mul_595: f32[256] = torch.ops.aten.mul.Tensor(sum_51, squeeze_85);  sum_51 = squeeze_85 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_594, relu_24, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_594 = primals_85 = None
        getitem_180: f32[64, 1024, 2, 2] = convolution_backward_24[0]
        getitem_181: f32[256, 1024, 1, 1] = convolution_backward_24[1];  convolution_backward_24 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        add_288: f32[64, 1024, 2, 2] = torch.ops.aten.add.Tensor(where_21, getitem_180);  where_21 = getitem_180 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:161, code: out = self.relu(out)
        le_24: b8[64, 1024, 2, 2] = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
        where_24: f32[64, 1024, 2, 2] = torch.ops.aten.where.self(le_24, scalar_tensor, add_288);  le_24 = add_288 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:158, code: identity = self.downsample(x)
        sum_52: f32[1024] = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
        sub_153: f32[64, 1024, 2, 2] = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_514);  convolution_27 = unsqueeze_514 = None
        mul_596: f32[64, 1024, 2, 2] = torch.ops.aten.mul.Tensor(where_24, sub_153)
        sum_53: f32[1024] = torch.ops.aten.sum.dim_IntList(mul_596, [0, 2, 3]);  mul_596 = None
        mul_597: f32[1024] = torch.ops.aten.mul.Tensor(sum_52, 0.00390625)
        unsqueeze_515: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_597, 0);  mul_597 = None
        unsqueeze_516: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_515, 2);  unsqueeze_515 = None
        unsqueeze_517: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_516, 3);  unsqueeze_516 = None
        mul_598: f32[1024] = torch.ops.aten.mul.Tensor(sum_53, 0.00390625)
        mul_599: f32[1024] = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_600: f32[1024] = torch.ops.aten.mul.Tensor(mul_598, mul_599);  mul_598 = mul_599 = None
        unsqueeze_518: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_600, 0);  mul_600 = None
        unsqueeze_519: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_518, 2);  unsqueeze_518 = None
        unsqueeze_520: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_519, 3);  unsqueeze_519 = None
        mul_601: f32[1024] = torch.ops.aten.mul.Tensor(squeeze_82, primals_83);  primals_83 = None
        unsqueeze_521: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_601, 0);  mul_601 = None
        unsqueeze_522: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_521, 2);  unsqueeze_521 = None
        unsqueeze_523: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_522, 3);  unsqueeze_522 = None
        mul_602: f32[64, 1024, 2, 2] = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_520);  sub_153 = unsqueeze_520 = None
        sub_155: f32[64, 1024, 2, 2] = torch.ops.aten.sub.Tensor(where_24, mul_602);  mul_602 = None
        sub_156: f32[64, 1024, 2, 2] = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_517);  sub_155 = None
        mul_603: f32[64, 1024, 2, 2] = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_523);  sub_156 = unsqueeze_523 = None
        mul_604: f32[1024] = torch.ops.aten.mul.Tensor(sum_53, squeeze_82);  sum_53 = squeeze_82 = None
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_603, relu_21, primals_82, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_603 = primals_82 = None
        getitem_183: f32[64, 512, 4, 4] = convolution_backward_25[0]
        getitem_184: f32[1024, 512, 1, 1] = convolution_backward_25[1];  convolution_backward_25 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:155, code: out = self.bn3(out)
        sub_157: f32[64, 1024, 2, 2] = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_526);  convolution_26 = unsqueeze_526 = None
        mul_605: f32[64, 1024, 2, 2] = torch.ops.aten.mul.Tensor(where_24, sub_157)
        sum_55: f32[1024] = torch.ops.aten.sum.dim_IntList(mul_605, [0, 2, 3]);  mul_605 = None
        mul_607: f32[1024] = torch.ops.aten.mul.Tensor(sum_55, 0.00390625)
        mul_608: f32[1024] = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_609: f32[1024] = torch.ops.aten.mul.Tensor(mul_607, mul_608);  mul_607 = mul_608 = None
        unsqueeze_530: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_609, 0);  mul_609 = None
        unsqueeze_531: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_530, 2);  unsqueeze_530 = None
        unsqueeze_532: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_531, 3);  unsqueeze_531 = None
        mul_610: f32[1024] = torch.ops.aten.mul.Tensor(squeeze_79, primals_80);  primals_80 = None
        unsqueeze_533: f32[1, 1024] = torch.ops.aten.unsqueeze.default(mul_610, 0);  mul_610 = None
        unsqueeze_534: f32[1, 1024, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_533, 2);  unsqueeze_533 = None
        unsqueeze_535: f32[1, 1024, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_534, 3);  unsqueeze_534 = None
        mul_611: f32[64, 1024, 2, 2] = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_532);  sub_157 = unsqueeze_532 = None
        sub_159: f32[64, 1024, 2, 2] = torch.ops.aten.sub.Tensor(where_24, mul_611);  where_24 = mul_611 = None
        sub_160: f32[64, 1024, 2, 2] = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_517);  sub_159 = unsqueeze_517 = None
        mul_612: f32[64, 1024, 2, 2] = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_535);  sub_160 = unsqueeze_535 = None
        mul_613: f32[1024] = torch.ops.aten.mul.Tensor(sum_55, squeeze_79);  sum_55 = squeeze_79 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:154, code: out = self.conv3(out)
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_612, relu_23, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_612 = primals_79 = None
        getitem_186: f32[64, 256, 2, 2] = convolution_backward_26[0]
        getitem_187: f32[1024, 256, 1, 1] = convolution_backward_26[1];  convolution_backward_26 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:152, code: out = self.relu(out)
        le_25: b8[64, 256, 2, 2] = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
        where_25: f32[64, 256, 2, 2] = torch.ops.aten.where.self(le_25, scalar_tensor, getitem_186);  le_25 = getitem_186 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:151, code: out = self.bn2(out)
        sum_56: f32[256] = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
        sub_161: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_538);  convolution_25 = unsqueeze_538 = None
        mul_614: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(where_25, sub_161)
        sum_57: f32[256] = torch.ops.aten.sum.dim_IntList(mul_614, [0, 2, 3]);  mul_614 = None
        mul_615: f32[256] = torch.ops.aten.mul.Tensor(sum_56, 0.00390625)
        unsqueeze_539: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_615, 0);  mul_615 = None
        unsqueeze_540: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_539, 2);  unsqueeze_539 = None
        unsqueeze_541: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_540, 3);  unsqueeze_540 = None
        mul_616: f32[256] = torch.ops.aten.mul.Tensor(sum_57, 0.00390625)
        mul_617: f32[256] = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_618: f32[256] = torch.ops.aten.mul.Tensor(mul_616, mul_617);  mul_616 = mul_617 = None
        unsqueeze_542: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_618, 0);  mul_618 = None
        unsqueeze_543: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_542, 2);  unsqueeze_542 = None
        unsqueeze_544: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_543, 3);  unsqueeze_543 = None
        mul_619: f32[256] = torch.ops.aten.mul.Tensor(squeeze_76, primals_77);  primals_77 = None
        unsqueeze_545: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_619, 0);  mul_619 = None
        unsqueeze_546: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_545, 2);  unsqueeze_545 = None
        unsqueeze_547: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_546, 3);  unsqueeze_546 = None
        mul_620: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_544);  sub_161 = unsqueeze_544 = None
        sub_163: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(where_25, mul_620);  where_25 = mul_620 = None
        sub_164: f32[64, 256, 2, 2] = torch.ops.aten.sub.Tensor(sub_163, unsqueeze_541);  sub_163 = unsqueeze_541 = None
        mul_621: f32[64, 256, 2, 2] = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_547);  sub_164 = unsqueeze_547 = None
        mul_622: f32[256] = torch.ops.aten.mul.Tensor(sum_57, squeeze_76);  sum_57 = squeeze_76 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:150, code: out = self.conv2(out)
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_621, relu_22, primals_76, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_621 = primals_76 = None
        getitem_189: f32[64, 256, 4, 4] = convolution_backward_27[0]
        getitem_190: f32[256, 256, 3, 3] = convolution_backward_27[1];  convolution_backward_27 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:148, code: out = self.relu(out)
        le_26: b8[64, 256, 4, 4] = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
        where_26: f32[64, 256, 4, 4] = torch.ops.aten.where.self(le_26, scalar_tensor, getitem_189);  le_26 = getitem_189 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:147, code: out = self.bn1(out)
        sum_58: f32[256] = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
        sub_165: f32[64, 256, 4, 4] = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_550);  convolution_24 = unsqueeze_550 = None
        mul_623: f32[64, 256, 4, 4] = torch.ops.aten.mul.Tensor(where_26, sub_165)
        sum_59: f32[256] = torch.ops.aten.sum.dim_IntList(mul_623, [0, 2, 3]);  mul_623 = None
        mul_624: f32[256] = torch.ops.aten.mul.Tensor(sum_58, 0.0009765625)
        unsqueeze_551: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_624, 0);  mul_624 = None
        unsqueeze_552: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_551, 2);  unsqueeze_551 = None
        unsqueeze_553: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_552, 3);  unsqueeze_552 = None
        mul_625: f32[256] = torch.ops.aten.mul.Tensor(sum_59, 0.0009765625)
        mul_626: f32[256] = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_627: f32[256] = torch.ops.aten.mul.Tensor(mul_625, mul_626);  mul_625 = mul_626 = None
        unsqueeze_554: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_627, 0);  mul_627 = None
        unsqueeze_555: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_554, 2);  unsqueeze_554 = None
        unsqueeze_556: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_555, 3);  unsqueeze_555 = None
        mul_628: f32[256] = torch.ops.aten.mul.Tensor(squeeze_73, primals_74);  primals_74 = None
        unsqueeze_557: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_628, 0);  mul_628 = None
        unsqueeze_558: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_557, 2);  unsqueeze_557 = None
        unsqueeze_559: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_558, 3);  unsqueeze_558 = None
        mul_629: f32[64, 256, 4, 4] = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_556);  sub_165 = unsqueeze_556 = None
        sub_167: f32[64, 256, 4, 4] = torch.ops.aten.sub.Tensor(where_26, mul_629);  where_26 = mul_629 = None
        sub_168: f32[64, 256, 4, 4] = torch.ops.aten.sub.Tensor(sub_167, unsqueeze_553);  sub_167 = unsqueeze_553 = None
        mul_630: f32[64, 256, 4, 4] = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_559);  sub_168 = unsqueeze_559 = None
        mul_631: f32[256] = torch.ops.aten.mul.Tensor(sum_59, squeeze_73);  sum_59 = squeeze_73 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_630, relu_21, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_630 = primals_73 = None
        getitem_192: f32[64, 512, 4, 4] = convolution_backward_28[0]
        getitem_193: f32[256, 512, 1, 1] = convolution_backward_28[1];  convolution_backward_28 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        add_289: f32[64, 512, 4, 4] = torch.ops.aten.add.Tensor(getitem_183, getitem_192);  getitem_183 = getitem_192 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:161, code: out = self.relu(out)
        le_27: b8[64, 512, 4, 4] = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
        where_27: f32[64, 512, 4, 4] = torch.ops.aten.where.self(le_27, scalar_tensor, add_289);  le_27 = add_289 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:155, code: out = self.bn3(out)
        sum_60: f32[512] = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
        sub_169: f32[64, 512, 4, 4] = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_562);  convolution_23 = unsqueeze_562 = None
        mul_632: f32[64, 512, 4, 4] = torch.ops.aten.mul.Tensor(where_27, sub_169)
        sum_61: f32[512] = torch.ops.aten.sum.dim_IntList(mul_632, [0, 2, 3]);  mul_632 = None
        mul_633: f32[512] = torch.ops.aten.mul.Tensor(sum_60, 0.0009765625)
        unsqueeze_563: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_633, 0);  mul_633 = None
        unsqueeze_564: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_563, 2);  unsqueeze_563 = None
        unsqueeze_565: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_564, 3);  unsqueeze_564 = None
        mul_634: f32[512] = torch.ops.aten.mul.Tensor(sum_61, 0.0009765625)
        mul_635: f32[512] = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_636: f32[512] = torch.ops.aten.mul.Tensor(mul_634, mul_635);  mul_634 = mul_635 = None
        unsqueeze_566: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_636, 0);  mul_636 = None
        unsqueeze_567: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_566, 2);  unsqueeze_566 = None
        unsqueeze_568: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_567, 3);  unsqueeze_567 = None
        mul_637: f32[512] = torch.ops.aten.mul.Tensor(squeeze_70, primals_71);  primals_71 = None
        unsqueeze_569: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_637, 0);  mul_637 = None
        unsqueeze_570: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_569, 2);  unsqueeze_569 = None
        unsqueeze_571: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_570, 3);  unsqueeze_570 = None
        mul_638: f32[64, 512, 4, 4] = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_568);  sub_169 = unsqueeze_568 = None
        sub_171: f32[64, 512, 4, 4] = torch.ops.aten.sub.Tensor(where_27, mul_638);  mul_638 = None
        sub_172: f32[64, 512, 4, 4] = torch.ops.aten.sub.Tensor(sub_171, unsqueeze_565);  sub_171 = unsqueeze_565 = None
        mul_639: f32[64, 512, 4, 4] = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_571);  sub_172 = unsqueeze_571 = None
        mul_640: f32[512] = torch.ops.aten.mul.Tensor(sum_61, squeeze_70);  sum_61 = squeeze_70 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:154, code: out = self.conv3(out)
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_639, relu_20, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_639 = primals_70 = None
        getitem_195: f32[64, 128, 4, 4] = convolution_backward_29[0]
        getitem_196: f32[512, 128, 1, 1] = convolution_backward_29[1];  convolution_backward_29 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:152, code: out = self.relu(out)
        le_28: b8[64, 128, 4, 4] = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
        where_28: f32[64, 128, 4, 4] = torch.ops.aten.where.self(le_28, scalar_tensor, getitem_195);  le_28 = getitem_195 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:151, code: out = self.bn2(out)
        sum_62: f32[128] = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
        sub_173: f32[64, 128, 4, 4] = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_574);  convolution_22 = unsqueeze_574 = None
        mul_641: f32[64, 128, 4, 4] = torch.ops.aten.mul.Tensor(where_28, sub_173)
        sum_63: f32[128] = torch.ops.aten.sum.dim_IntList(mul_641, [0, 2, 3]);  mul_641 = None
        mul_642: f32[128] = torch.ops.aten.mul.Tensor(sum_62, 0.0009765625)
        unsqueeze_575: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_642, 0);  mul_642 = None
        unsqueeze_576: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_575, 2);  unsqueeze_575 = None
        unsqueeze_577: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_576, 3);  unsqueeze_576 = None
        mul_643: f32[128] = torch.ops.aten.mul.Tensor(sum_63, 0.0009765625)
        mul_644: f32[128] = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_645: f32[128] = torch.ops.aten.mul.Tensor(mul_643, mul_644);  mul_643 = mul_644 = None
        unsqueeze_578: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_645, 0);  mul_645 = None
        unsqueeze_579: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_578, 2);  unsqueeze_578 = None
        unsqueeze_580: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_579, 3);  unsqueeze_579 = None
        mul_646: f32[128] = torch.ops.aten.mul.Tensor(squeeze_67, primals_68);  primals_68 = None
        unsqueeze_581: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_646, 0);  mul_646 = None
        unsqueeze_582: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_581, 2);  unsqueeze_581 = None
        unsqueeze_583: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_582, 3);  unsqueeze_582 = None
        mul_647: f32[64, 128, 4, 4] = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_580);  sub_173 = unsqueeze_580 = None
        sub_175: f32[64, 128, 4, 4] = torch.ops.aten.sub.Tensor(where_28, mul_647);  where_28 = mul_647 = None
        sub_176: f32[64, 128, 4, 4] = torch.ops.aten.sub.Tensor(sub_175, unsqueeze_577);  sub_175 = unsqueeze_577 = None
        mul_648: f32[64, 128, 4, 4] = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_583);  sub_176 = unsqueeze_583 = None
        mul_649: f32[128] = torch.ops.aten.mul.Tensor(sum_63, squeeze_67);  sum_63 = squeeze_67 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:150, code: out = self.conv2(out)
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_648, relu_19, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_648 = primals_67 = None
        getitem_198: f32[64, 128, 4, 4] = convolution_backward_30[0]
        getitem_199: f32[128, 128, 3, 3] = convolution_backward_30[1];  convolution_backward_30 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:148, code: out = self.relu(out)
        le_29: b8[64, 128, 4, 4] = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
        where_29: f32[64, 128, 4, 4] = torch.ops.aten.where.self(le_29, scalar_tensor, getitem_198);  le_29 = getitem_198 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:147, code: out = self.bn1(out)
        sum_64: f32[128] = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
        sub_177: f32[64, 128, 4, 4] = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_586);  convolution_21 = unsqueeze_586 = None
        mul_650: f32[64, 128, 4, 4] = torch.ops.aten.mul.Tensor(where_29, sub_177)
        sum_65: f32[128] = torch.ops.aten.sum.dim_IntList(mul_650, [0, 2, 3]);  mul_650 = None
        mul_651: f32[128] = torch.ops.aten.mul.Tensor(sum_64, 0.0009765625)
        unsqueeze_587: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_651, 0);  mul_651 = None
        unsqueeze_588: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_587, 2);  unsqueeze_587 = None
        unsqueeze_589: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_588, 3);  unsqueeze_588 = None
        mul_652: f32[128] = torch.ops.aten.mul.Tensor(sum_65, 0.0009765625)
        mul_653: f32[128] = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_654: f32[128] = torch.ops.aten.mul.Tensor(mul_652, mul_653);  mul_652 = mul_653 = None
        unsqueeze_590: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_654, 0);  mul_654 = None
        unsqueeze_591: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_590, 2);  unsqueeze_590 = None
        unsqueeze_592: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_591, 3);  unsqueeze_591 = None
        mul_655: f32[128] = torch.ops.aten.mul.Tensor(squeeze_64, primals_65);  primals_65 = None
        unsqueeze_593: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_655, 0);  mul_655 = None
        unsqueeze_594: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_593, 2);  unsqueeze_593 = None
        unsqueeze_595: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_594, 3);  unsqueeze_594 = None
        mul_656: f32[64, 128, 4, 4] = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_592);  sub_177 = unsqueeze_592 = None
        sub_179: f32[64, 128, 4, 4] = torch.ops.aten.sub.Tensor(where_29, mul_656);  where_29 = mul_656 = None
        sub_180: f32[64, 128, 4, 4] = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_589);  sub_179 = unsqueeze_589 = None
        mul_657: f32[64, 128, 4, 4] = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_595);  sub_180 = unsqueeze_595 = None
        mul_658: f32[128] = torch.ops.aten.mul.Tensor(sum_65, squeeze_64);  sum_65 = squeeze_64 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_657, relu_18, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_657 = primals_64 = None
        getitem_201: f32[64, 512, 4, 4] = convolution_backward_31[0]
        getitem_202: f32[128, 512, 1, 1] = convolution_backward_31[1];  convolution_backward_31 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        add_290: f32[64, 512, 4, 4] = torch.ops.aten.add.Tensor(where_27, getitem_201);  where_27 = getitem_201 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:161, code: out = self.relu(out)
        le_30: b8[64, 512, 4, 4] = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        where_30: f32[64, 512, 4, 4] = torch.ops.aten.where.self(le_30, scalar_tensor, add_290);  le_30 = add_290 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:155, code: out = self.bn3(out)
        sum_66: f32[512] = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
        sub_181: f32[64, 512, 4, 4] = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_598);  convolution_20 = unsqueeze_598 = None
        mul_659: f32[64, 512, 4, 4] = torch.ops.aten.mul.Tensor(where_30, sub_181)
        sum_67: f32[512] = torch.ops.aten.sum.dim_IntList(mul_659, [0, 2, 3]);  mul_659 = None
        mul_660: f32[512] = torch.ops.aten.mul.Tensor(sum_66, 0.0009765625)
        unsqueeze_599: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_660, 0);  mul_660 = None
        unsqueeze_600: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_599, 2);  unsqueeze_599 = None
        unsqueeze_601: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_600, 3);  unsqueeze_600 = None
        mul_661: f32[512] = torch.ops.aten.mul.Tensor(sum_67, 0.0009765625)
        mul_662: f32[512] = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_663: f32[512] = torch.ops.aten.mul.Tensor(mul_661, mul_662);  mul_661 = mul_662 = None
        unsqueeze_602: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_663, 0);  mul_663 = None
        unsqueeze_603: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_602, 2);  unsqueeze_602 = None
        unsqueeze_604: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_603, 3);  unsqueeze_603 = None
        mul_664: f32[512] = torch.ops.aten.mul.Tensor(squeeze_61, primals_62);  primals_62 = None
        unsqueeze_605: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_664, 0);  mul_664 = None
        unsqueeze_606: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_605, 2);  unsqueeze_605 = None
        unsqueeze_607: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_606, 3);  unsqueeze_606 = None
        mul_665: f32[64, 512, 4, 4] = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_604);  sub_181 = unsqueeze_604 = None
        sub_183: f32[64, 512, 4, 4] = torch.ops.aten.sub.Tensor(where_30, mul_665);  mul_665 = None
        sub_184: f32[64, 512, 4, 4] = torch.ops.aten.sub.Tensor(sub_183, unsqueeze_601);  sub_183 = unsqueeze_601 = None
        mul_666: f32[64, 512, 4, 4] = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_607);  sub_184 = unsqueeze_607 = None
        mul_667: f32[512] = torch.ops.aten.mul.Tensor(sum_67, squeeze_61);  sum_67 = squeeze_61 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:154, code: out = self.conv3(out)
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_666, relu_17, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_666 = primals_61 = None
        getitem_204: f32[64, 128, 4, 4] = convolution_backward_32[0]
        getitem_205: f32[512, 128, 1, 1] = convolution_backward_32[1];  convolution_backward_32 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:152, code: out = self.relu(out)
        le_31: b8[64, 128, 4, 4] = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        where_31: f32[64, 128, 4, 4] = torch.ops.aten.where.self(le_31, scalar_tensor, getitem_204);  le_31 = getitem_204 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:151, code: out = self.bn2(out)
        sum_68: f32[128] = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
        sub_185: f32[64, 128, 4, 4] = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_610);  convolution_19 = unsqueeze_610 = None
        mul_668: f32[64, 128, 4, 4] = torch.ops.aten.mul.Tensor(where_31, sub_185)
        sum_69: f32[128] = torch.ops.aten.sum.dim_IntList(mul_668, [0, 2, 3]);  mul_668 = None
        mul_669: f32[128] = torch.ops.aten.mul.Tensor(sum_68, 0.0009765625)
        unsqueeze_611: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_669, 0);  mul_669 = None
        unsqueeze_612: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_611, 2);  unsqueeze_611 = None
        unsqueeze_613: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_612, 3);  unsqueeze_612 = None
        mul_670: f32[128] = torch.ops.aten.mul.Tensor(sum_69, 0.0009765625)
        mul_671: f32[128] = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_672: f32[128] = torch.ops.aten.mul.Tensor(mul_670, mul_671);  mul_670 = mul_671 = None
        unsqueeze_614: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_672, 0);  mul_672 = None
        unsqueeze_615: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_614, 2);  unsqueeze_614 = None
        unsqueeze_616: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_615, 3);  unsqueeze_615 = None
        mul_673: f32[128] = torch.ops.aten.mul.Tensor(squeeze_58, primals_59);  primals_59 = None
        unsqueeze_617: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_673, 0);  mul_673 = None
        unsqueeze_618: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_617, 2);  unsqueeze_617 = None
        unsqueeze_619: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_618, 3);  unsqueeze_618 = None
        mul_674: f32[64, 128, 4, 4] = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_616);  sub_185 = unsqueeze_616 = None
        sub_187: f32[64, 128, 4, 4] = torch.ops.aten.sub.Tensor(where_31, mul_674);  where_31 = mul_674 = None
        sub_188: f32[64, 128, 4, 4] = torch.ops.aten.sub.Tensor(sub_187, unsqueeze_613);  sub_187 = unsqueeze_613 = None
        mul_675: f32[64, 128, 4, 4] = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_619);  sub_188 = unsqueeze_619 = None
        mul_676: f32[128] = torch.ops.aten.mul.Tensor(sum_69, squeeze_58);  sum_69 = squeeze_58 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:150, code: out = self.conv2(out)
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_675, relu_16, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_675 = primals_58 = None
        getitem_207: f32[64, 128, 4, 4] = convolution_backward_33[0]
        getitem_208: f32[128, 128, 3, 3] = convolution_backward_33[1];  convolution_backward_33 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:148, code: out = self.relu(out)
        le_32: b8[64, 128, 4, 4] = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        where_32: f32[64, 128, 4, 4] = torch.ops.aten.where.self(le_32, scalar_tensor, getitem_207);  le_32 = getitem_207 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:147, code: out = self.bn1(out)
        sum_70: f32[128] = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
        sub_189: f32[64, 128, 4, 4] = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_622);  convolution_18 = unsqueeze_622 = None
        mul_677: f32[64, 128, 4, 4] = torch.ops.aten.mul.Tensor(where_32, sub_189)
        sum_71: f32[128] = torch.ops.aten.sum.dim_IntList(mul_677, [0, 2, 3]);  mul_677 = None
        mul_678: f32[128] = torch.ops.aten.mul.Tensor(sum_70, 0.0009765625)
        unsqueeze_623: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_678, 0);  mul_678 = None
        unsqueeze_624: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_623, 2);  unsqueeze_623 = None
        unsqueeze_625: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_624, 3);  unsqueeze_624 = None
        mul_679: f32[128] = torch.ops.aten.mul.Tensor(sum_71, 0.0009765625)
        mul_680: f32[128] = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_681: f32[128] = torch.ops.aten.mul.Tensor(mul_679, mul_680);  mul_679 = mul_680 = None
        unsqueeze_626: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_681, 0);  mul_681 = None
        unsqueeze_627: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_626, 2);  unsqueeze_626 = None
        unsqueeze_628: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_627, 3);  unsqueeze_627 = None
        mul_682: f32[128] = torch.ops.aten.mul.Tensor(squeeze_55, primals_56);  primals_56 = None
        unsqueeze_629: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_682, 0);  mul_682 = None
        unsqueeze_630: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_629, 2);  unsqueeze_629 = None
        unsqueeze_631: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_630, 3);  unsqueeze_630 = None
        mul_683: f32[64, 128, 4, 4] = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_628);  sub_189 = unsqueeze_628 = None
        sub_191: f32[64, 128, 4, 4] = torch.ops.aten.sub.Tensor(where_32, mul_683);  where_32 = mul_683 = None
        sub_192: f32[64, 128, 4, 4] = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_625);  sub_191 = unsqueeze_625 = None
        mul_684: f32[64, 128, 4, 4] = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_631);  sub_192 = unsqueeze_631 = None
        mul_685: f32[128] = torch.ops.aten.mul.Tensor(sum_71, squeeze_55);  sum_71 = squeeze_55 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_684, relu_15, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_684 = primals_55 = None
        getitem_210: f32[64, 512, 4, 4] = convolution_backward_34[0]
        getitem_211: f32[128, 512, 1, 1] = convolution_backward_34[1];  convolution_backward_34 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        add_291: f32[64, 512, 4, 4] = torch.ops.aten.add.Tensor(where_30, getitem_210);  where_30 = getitem_210 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:161, code: out = self.relu(out)
        le_33: b8[64, 512, 4, 4] = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_33: f32[64, 512, 4, 4] = torch.ops.aten.where.self(le_33, scalar_tensor, add_291);  le_33 = add_291 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:155, code: out = self.bn3(out)
        sum_72: f32[512] = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
        sub_193: f32[64, 512, 4, 4] = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_634);  convolution_17 = unsqueeze_634 = None
        mul_686: f32[64, 512, 4, 4] = torch.ops.aten.mul.Tensor(where_33, sub_193)
        sum_73: f32[512] = torch.ops.aten.sum.dim_IntList(mul_686, [0, 2, 3]);  mul_686 = None
        mul_687: f32[512] = torch.ops.aten.mul.Tensor(sum_72, 0.0009765625)
        unsqueeze_635: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_687, 0);  mul_687 = None
        unsqueeze_636: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_635, 2);  unsqueeze_635 = None
        unsqueeze_637: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_636, 3);  unsqueeze_636 = None
        mul_688: f32[512] = torch.ops.aten.mul.Tensor(sum_73, 0.0009765625)
        mul_689: f32[512] = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_690: f32[512] = torch.ops.aten.mul.Tensor(mul_688, mul_689);  mul_688 = mul_689 = None
        unsqueeze_638: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_690, 0);  mul_690 = None
        unsqueeze_639: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_638, 2);  unsqueeze_638 = None
        unsqueeze_640: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_639, 3);  unsqueeze_639 = None
        mul_691: f32[512] = torch.ops.aten.mul.Tensor(squeeze_52, primals_53);  primals_53 = None
        unsqueeze_641: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_691, 0);  mul_691 = None
        unsqueeze_642: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_641, 2);  unsqueeze_641 = None
        unsqueeze_643: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_642, 3);  unsqueeze_642 = None
        mul_692: f32[64, 512, 4, 4] = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_640);  sub_193 = unsqueeze_640 = None
        sub_195: f32[64, 512, 4, 4] = torch.ops.aten.sub.Tensor(where_33, mul_692);  mul_692 = None
        sub_196: f32[64, 512, 4, 4] = torch.ops.aten.sub.Tensor(sub_195, unsqueeze_637);  sub_195 = unsqueeze_637 = None
        mul_693: f32[64, 512, 4, 4] = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_643);  sub_196 = unsqueeze_643 = None
        mul_694: f32[512] = torch.ops.aten.mul.Tensor(sum_73, squeeze_52);  sum_73 = squeeze_52 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:154, code: out = self.conv3(out)
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_693, relu_14, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_693 = primals_52 = None
        getitem_213: f32[64, 128, 4, 4] = convolution_backward_35[0]
        getitem_214: f32[512, 128, 1, 1] = convolution_backward_35[1];  convolution_backward_35 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:152, code: out = self.relu(out)
        le_34: b8[64, 128, 4, 4] = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_34: f32[64, 128, 4, 4] = torch.ops.aten.where.self(le_34, scalar_tensor, getitem_213);  le_34 = getitem_213 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:151, code: out = self.bn2(out)
        sum_74: f32[128] = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
        sub_197: f32[64, 128, 4, 4] = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_646);  convolution_16 = unsqueeze_646 = None
        mul_695: f32[64, 128, 4, 4] = torch.ops.aten.mul.Tensor(where_34, sub_197)
        sum_75: f32[128] = torch.ops.aten.sum.dim_IntList(mul_695, [0, 2, 3]);  mul_695 = None
        mul_696: f32[128] = torch.ops.aten.mul.Tensor(sum_74, 0.0009765625)
        unsqueeze_647: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_696, 0);  mul_696 = None
        unsqueeze_648: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_647, 2);  unsqueeze_647 = None
        unsqueeze_649: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_648, 3);  unsqueeze_648 = None
        mul_697: f32[128] = torch.ops.aten.mul.Tensor(sum_75, 0.0009765625)
        mul_698: f32[128] = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_699: f32[128] = torch.ops.aten.mul.Tensor(mul_697, mul_698);  mul_697 = mul_698 = None
        unsqueeze_650: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_699, 0);  mul_699 = None
        unsqueeze_651: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_650, 2);  unsqueeze_650 = None
        unsqueeze_652: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_651, 3);  unsqueeze_651 = None
        mul_700: f32[128] = torch.ops.aten.mul.Tensor(squeeze_49, primals_50);  primals_50 = None
        unsqueeze_653: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_700, 0);  mul_700 = None
        unsqueeze_654: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_653, 2);  unsqueeze_653 = None
        unsqueeze_655: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_654, 3);  unsqueeze_654 = None
        mul_701: f32[64, 128, 4, 4] = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_652);  sub_197 = unsqueeze_652 = None
        sub_199: f32[64, 128, 4, 4] = torch.ops.aten.sub.Tensor(where_34, mul_701);  where_34 = mul_701 = None
        sub_200: f32[64, 128, 4, 4] = torch.ops.aten.sub.Tensor(sub_199, unsqueeze_649);  sub_199 = unsqueeze_649 = None
        mul_702: f32[64, 128, 4, 4] = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_655);  sub_200 = unsqueeze_655 = None
        mul_703: f32[128] = torch.ops.aten.mul.Tensor(sum_75, squeeze_49);  sum_75 = squeeze_49 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:150, code: out = self.conv2(out)
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_702, relu_13, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_702 = primals_49 = None
        getitem_216: f32[64, 128, 4, 4] = convolution_backward_36[0]
        getitem_217: f32[128, 128, 3, 3] = convolution_backward_36[1];  convolution_backward_36 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:148, code: out = self.relu(out)
        le_35: b8[64, 128, 4, 4] = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_35: f32[64, 128, 4, 4] = torch.ops.aten.where.self(le_35, scalar_tensor, getitem_216);  le_35 = getitem_216 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:147, code: out = self.bn1(out)
        sum_76: f32[128] = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
        sub_201: f32[64, 128, 4, 4] = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_658);  convolution_15 = unsqueeze_658 = None
        mul_704: f32[64, 128, 4, 4] = torch.ops.aten.mul.Tensor(where_35, sub_201)
        sum_77: f32[128] = torch.ops.aten.sum.dim_IntList(mul_704, [0, 2, 3]);  mul_704 = None
        mul_705: f32[128] = torch.ops.aten.mul.Tensor(sum_76, 0.0009765625)
        unsqueeze_659: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_705, 0);  mul_705 = None
        unsqueeze_660: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_659, 2);  unsqueeze_659 = None
        unsqueeze_661: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_660, 3);  unsqueeze_660 = None
        mul_706: f32[128] = torch.ops.aten.mul.Tensor(sum_77, 0.0009765625)
        mul_707: f32[128] = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_708: f32[128] = torch.ops.aten.mul.Tensor(mul_706, mul_707);  mul_706 = mul_707 = None
        unsqueeze_662: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_708, 0);  mul_708 = None
        unsqueeze_663: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_662, 2);  unsqueeze_662 = None
        unsqueeze_664: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_663, 3);  unsqueeze_663 = None
        mul_709: f32[128] = torch.ops.aten.mul.Tensor(squeeze_46, primals_47);  primals_47 = None
        unsqueeze_665: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_709, 0);  mul_709 = None
        unsqueeze_666: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_665, 2);  unsqueeze_665 = None
        unsqueeze_667: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_666, 3);  unsqueeze_666 = None
        mul_710: f32[64, 128, 4, 4] = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_664);  sub_201 = unsqueeze_664 = None
        sub_203: f32[64, 128, 4, 4] = torch.ops.aten.sub.Tensor(where_35, mul_710);  where_35 = mul_710 = None
        sub_204: f32[64, 128, 4, 4] = torch.ops.aten.sub.Tensor(sub_203, unsqueeze_661);  sub_203 = unsqueeze_661 = None
        mul_711: f32[64, 128, 4, 4] = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_667);  sub_204 = unsqueeze_667 = None
        mul_712: f32[128] = torch.ops.aten.mul.Tensor(sum_77, squeeze_46);  sum_77 = squeeze_46 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_711, relu_12, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_711 = primals_46 = None
        getitem_219: f32[64, 512, 4, 4] = convolution_backward_37[0]
        getitem_220: f32[128, 512, 1, 1] = convolution_backward_37[1];  convolution_backward_37 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        add_292: f32[64, 512, 4, 4] = torch.ops.aten.add.Tensor(where_33, getitem_219);  where_33 = getitem_219 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:161, code: out = self.relu(out)
        le_36: b8[64, 512, 4, 4] = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_36: f32[64, 512, 4, 4] = torch.ops.aten.where.self(le_36, scalar_tensor, add_292);  le_36 = add_292 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:158, code: identity = self.downsample(x)
        sum_78: f32[512] = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
        sub_205: f32[64, 512, 4, 4] = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_670);  convolution_14 = unsqueeze_670 = None
        mul_713: f32[64, 512, 4, 4] = torch.ops.aten.mul.Tensor(where_36, sub_205)
        sum_79: f32[512] = torch.ops.aten.sum.dim_IntList(mul_713, [0, 2, 3]);  mul_713 = None
        mul_714: f32[512] = torch.ops.aten.mul.Tensor(sum_78, 0.0009765625)
        unsqueeze_671: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_714, 0);  mul_714 = None
        unsqueeze_672: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_671, 2);  unsqueeze_671 = None
        unsqueeze_673: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_672, 3);  unsqueeze_672 = None
        mul_715: f32[512] = torch.ops.aten.mul.Tensor(sum_79, 0.0009765625)
        mul_716: f32[512] = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_717: f32[512] = torch.ops.aten.mul.Tensor(mul_715, mul_716);  mul_715 = mul_716 = None
        unsqueeze_674: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_717, 0);  mul_717 = None
        unsqueeze_675: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_674, 2);  unsqueeze_674 = None
        unsqueeze_676: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_675, 3);  unsqueeze_675 = None
        mul_718: f32[512] = torch.ops.aten.mul.Tensor(squeeze_43, primals_44);  primals_44 = None
        unsqueeze_677: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_718, 0);  mul_718 = None
        unsqueeze_678: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_677, 2);  unsqueeze_677 = None
        unsqueeze_679: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_678, 3);  unsqueeze_678 = None
        mul_719: f32[64, 512, 4, 4] = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_676);  sub_205 = unsqueeze_676 = None
        sub_207: f32[64, 512, 4, 4] = torch.ops.aten.sub.Tensor(where_36, mul_719);  mul_719 = None
        sub_208: f32[64, 512, 4, 4] = torch.ops.aten.sub.Tensor(sub_207, unsqueeze_673);  sub_207 = None
        mul_720: f32[64, 512, 4, 4] = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_679);  sub_208 = unsqueeze_679 = None
        mul_721: f32[512] = torch.ops.aten.mul.Tensor(sum_79, squeeze_43);  sum_79 = squeeze_43 = None
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_720, relu_9, primals_43, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_720 = primals_43 = None
        getitem_222: f32[64, 256, 8, 8] = convolution_backward_38[0]
        getitem_223: f32[512, 256, 1, 1] = convolution_backward_38[1];  convolution_backward_38 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:155, code: out = self.bn3(out)
        sub_209: f32[64, 512, 4, 4] = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_682);  convolution_13 = unsqueeze_682 = None
        mul_722: f32[64, 512, 4, 4] = torch.ops.aten.mul.Tensor(where_36, sub_209)
        sum_81: f32[512] = torch.ops.aten.sum.dim_IntList(mul_722, [0, 2, 3]);  mul_722 = None
        mul_724: f32[512] = torch.ops.aten.mul.Tensor(sum_81, 0.0009765625)
        mul_725: f32[512] = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_726: f32[512] = torch.ops.aten.mul.Tensor(mul_724, mul_725);  mul_724 = mul_725 = None
        unsqueeze_686: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_726, 0);  mul_726 = None
        unsqueeze_687: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_686, 2);  unsqueeze_686 = None
        unsqueeze_688: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_687, 3);  unsqueeze_687 = None
        mul_727: f32[512] = torch.ops.aten.mul.Tensor(squeeze_40, primals_41);  primals_41 = None
        unsqueeze_689: f32[1, 512] = torch.ops.aten.unsqueeze.default(mul_727, 0);  mul_727 = None
        unsqueeze_690: f32[1, 512, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_689, 2);  unsqueeze_689 = None
        unsqueeze_691: f32[1, 512, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_690, 3);  unsqueeze_690 = None
        mul_728: f32[64, 512, 4, 4] = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_688);  sub_209 = unsqueeze_688 = None
        sub_211: f32[64, 512, 4, 4] = torch.ops.aten.sub.Tensor(where_36, mul_728);  where_36 = mul_728 = None
        sub_212: f32[64, 512, 4, 4] = torch.ops.aten.sub.Tensor(sub_211, unsqueeze_673);  sub_211 = unsqueeze_673 = None
        mul_729: f32[64, 512, 4, 4] = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_691);  sub_212 = unsqueeze_691 = None
        mul_730: f32[512] = torch.ops.aten.mul.Tensor(sum_81, squeeze_40);  sum_81 = squeeze_40 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:154, code: out = self.conv3(out)
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_729, relu_11, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_729 = primals_40 = None
        getitem_225: f32[64, 128, 4, 4] = convolution_backward_39[0]
        getitem_226: f32[512, 128, 1, 1] = convolution_backward_39[1];  convolution_backward_39 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:152, code: out = self.relu(out)
        le_37: b8[64, 128, 4, 4] = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_37: f32[64, 128, 4, 4] = torch.ops.aten.where.self(le_37, scalar_tensor, getitem_225);  le_37 = getitem_225 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:151, code: out = self.bn2(out)
        sum_82: f32[128] = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
        sub_213: f32[64, 128, 4, 4] = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_694);  convolution_12 = unsqueeze_694 = None
        mul_731: f32[64, 128, 4, 4] = torch.ops.aten.mul.Tensor(where_37, sub_213)
        sum_83: f32[128] = torch.ops.aten.sum.dim_IntList(mul_731, [0, 2, 3]);  mul_731 = None
        mul_732: f32[128] = torch.ops.aten.mul.Tensor(sum_82, 0.0009765625)
        unsqueeze_695: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_732, 0);  mul_732 = None
        unsqueeze_696: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_695, 2);  unsqueeze_695 = None
        unsqueeze_697: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_696, 3);  unsqueeze_696 = None
        mul_733: f32[128] = torch.ops.aten.mul.Tensor(sum_83, 0.0009765625)
        mul_734: f32[128] = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_735: f32[128] = torch.ops.aten.mul.Tensor(mul_733, mul_734);  mul_733 = mul_734 = None
        unsqueeze_698: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_735, 0);  mul_735 = None
        unsqueeze_699: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_698, 2);  unsqueeze_698 = None
        unsqueeze_700: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_699, 3);  unsqueeze_699 = None
        mul_736: f32[128] = torch.ops.aten.mul.Tensor(squeeze_37, primals_38);  primals_38 = None
        unsqueeze_701: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_736, 0);  mul_736 = None
        unsqueeze_702: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_701, 2);  unsqueeze_701 = None
        unsqueeze_703: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_702, 3);  unsqueeze_702 = None
        mul_737: f32[64, 128, 4, 4] = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_700);  sub_213 = unsqueeze_700 = None
        sub_215: f32[64, 128, 4, 4] = torch.ops.aten.sub.Tensor(where_37, mul_737);  where_37 = mul_737 = None
        sub_216: f32[64, 128, 4, 4] = torch.ops.aten.sub.Tensor(sub_215, unsqueeze_697);  sub_215 = unsqueeze_697 = None
        mul_738: f32[64, 128, 4, 4] = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_703);  sub_216 = unsqueeze_703 = None
        mul_739: f32[128] = torch.ops.aten.mul.Tensor(sum_83, squeeze_37);  sum_83 = squeeze_37 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:150, code: out = self.conv2(out)
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_738, relu_10, primals_37, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_738 = primals_37 = None
        getitem_228: f32[64, 128, 8, 8] = convolution_backward_40[0]
        getitem_229: f32[128, 128, 3, 3] = convolution_backward_40[1];  convolution_backward_40 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:148, code: out = self.relu(out)
        le_38: b8[64, 128, 8, 8] = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_38: f32[64, 128, 8, 8] = torch.ops.aten.where.self(le_38, scalar_tensor, getitem_228);  le_38 = getitem_228 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:147, code: out = self.bn1(out)
        sum_84: f32[128] = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
        sub_217: f32[64, 128, 8, 8] = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_706);  convolution_11 = unsqueeze_706 = None
        mul_740: f32[64, 128, 8, 8] = torch.ops.aten.mul.Tensor(where_38, sub_217)
        sum_85: f32[128] = torch.ops.aten.sum.dim_IntList(mul_740, [0, 2, 3]);  mul_740 = None
        mul_741: f32[128] = torch.ops.aten.mul.Tensor(sum_84, 0.000244140625)
        unsqueeze_707: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_741, 0);  mul_741 = None
        unsqueeze_708: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_707, 2);  unsqueeze_707 = None
        unsqueeze_709: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_708, 3);  unsqueeze_708 = None
        mul_742: f32[128] = torch.ops.aten.mul.Tensor(sum_85, 0.000244140625)
        mul_743: f32[128] = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_744: f32[128] = torch.ops.aten.mul.Tensor(mul_742, mul_743);  mul_742 = mul_743 = None
        unsqueeze_710: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_744, 0);  mul_744 = None
        unsqueeze_711: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_710, 2);  unsqueeze_710 = None
        unsqueeze_712: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_711, 3);  unsqueeze_711 = None
        mul_745: f32[128] = torch.ops.aten.mul.Tensor(squeeze_34, primals_35);  primals_35 = None
        unsqueeze_713: f32[1, 128] = torch.ops.aten.unsqueeze.default(mul_745, 0);  mul_745 = None
        unsqueeze_714: f32[1, 128, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_713, 2);  unsqueeze_713 = None
        unsqueeze_715: f32[1, 128, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_714, 3);  unsqueeze_714 = None
        mul_746: f32[64, 128, 8, 8] = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_712);  sub_217 = unsqueeze_712 = None
        sub_219: f32[64, 128, 8, 8] = torch.ops.aten.sub.Tensor(where_38, mul_746);  where_38 = mul_746 = None
        sub_220: f32[64, 128, 8, 8] = torch.ops.aten.sub.Tensor(sub_219, unsqueeze_709);  sub_219 = unsqueeze_709 = None
        mul_747: f32[64, 128, 8, 8] = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_715);  sub_220 = unsqueeze_715 = None
        mul_748: f32[128] = torch.ops.aten.mul.Tensor(sum_85, squeeze_34);  sum_85 = squeeze_34 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_747, relu_9, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_747 = primals_34 = None
        getitem_231: f32[64, 256, 8, 8] = convolution_backward_41[0]
        getitem_232: f32[128, 256, 1, 1] = convolution_backward_41[1];  convolution_backward_41 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        add_293: f32[64, 256, 8, 8] = torch.ops.aten.add.Tensor(getitem_222, getitem_231);  getitem_222 = getitem_231 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:161, code: out = self.relu(out)
        le_39: b8[64, 256, 8, 8] = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_39: f32[64, 256, 8, 8] = torch.ops.aten.where.self(le_39, scalar_tensor, add_293);  le_39 = add_293 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:155, code: out = self.bn3(out)
        sum_86: f32[256] = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
        sub_221: f32[64, 256, 8, 8] = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_718);  convolution_10 = unsqueeze_718 = None
        mul_749: f32[64, 256, 8, 8] = torch.ops.aten.mul.Tensor(where_39, sub_221)
        sum_87: f32[256] = torch.ops.aten.sum.dim_IntList(mul_749, [0, 2, 3]);  mul_749 = None
        mul_750: f32[256] = torch.ops.aten.mul.Tensor(sum_86, 0.000244140625)
        unsqueeze_719: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_750, 0);  mul_750 = None
        unsqueeze_720: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_719, 2);  unsqueeze_719 = None
        unsqueeze_721: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_720, 3);  unsqueeze_720 = None
        mul_751: f32[256] = torch.ops.aten.mul.Tensor(sum_87, 0.000244140625)
        mul_752: f32[256] = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_753: f32[256] = torch.ops.aten.mul.Tensor(mul_751, mul_752);  mul_751 = mul_752 = None
        unsqueeze_722: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_753, 0);  mul_753 = None
        unsqueeze_723: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_722, 2);  unsqueeze_722 = None
        unsqueeze_724: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_723, 3);  unsqueeze_723 = None
        mul_754: f32[256] = torch.ops.aten.mul.Tensor(squeeze_31, primals_32);  primals_32 = None
        unsqueeze_725: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_754, 0);  mul_754 = None
        unsqueeze_726: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_725, 2);  unsqueeze_725 = None
        unsqueeze_727: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_726, 3);  unsqueeze_726 = None
        mul_755: f32[64, 256, 8, 8] = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_724);  sub_221 = unsqueeze_724 = None
        sub_223: f32[64, 256, 8, 8] = torch.ops.aten.sub.Tensor(where_39, mul_755);  mul_755 = None
        sub_224: f32[64, 256, 8, 8] = torch.ops.aten.sub.Tensor(sub_223, unsqueeze_721);  sub_223 = unsqueeze_721 = None
        mul_756: f32[64, 256, 8, 8] = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_727);  sub_224 = unsqueeze_727 = None
        mul_757: f32[256] = torch.ops.aten.mul.Tensor(sum_87, squeeze_31);  sum_87 = squeeze_31 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:154, code: out = self.conv3(out)
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_756, relu_8, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_756 = primals_31 = None
        getitem_234: f32[64, 64, 8, 8] = convolution_backward_42[0]
        getitem_235: f32[256, 64, 1, 1] = convolution_backward_42[1];  convolution_backward_42 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:152, code: out = self.relu(out)
        le_40: b8[64, 64, 8, 8] = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_40: f32[64, 64, 8, 8] = torch.ops.aten.where.self(le_40, scalar_tensor, getitem_234);  le_40 = getitem_234 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:151, code: out = self.bn2(out)
        sum_88: f32[64] = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
        sub_225: f32[64, 64, 8, 8] = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_730);  convolution_9 = unsqueeze_730 = None
        mul_758: f32[64, 64, 8, 8] = torch.ops.aten.mul.Tensor(where_40, sub_225)
        sum_89: f32[64] = torch.ops.aten.sum.dim_IntList(mul_758, [0, 2, 3]);  mul_758 = None
        mul_759: f32[64] = torch.ops.aten.mul.Tensor(sum_88, 0.000244140625)
        unsqueeze_731: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_759, 0);  mul_759 = None
        unsqueeze_732: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_731, 2);  unsqueeze_731 = None
        unsqueeze_733: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_732, 3);  unsqueeze_732 = None
        mul_760: f32[64] = torch.ops.aten.mul.Tensor(sum_89, 0.000244140625)
        mul_761: f32[64] = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_762: f32[64] = torch.ops.aten.mul.Tensor(mul_760, mul_761);  mul_760 = mul_761 = None
        unsqueeze_734: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_762, 0);  mul_762 = None
        unsqueeze_735: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_734, 2);  unsqueeze_734 = None
        unsqueeze_736: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_735, 3);  unsqueeze_735 = None
        mul_763: f32[64] = torch.ops.aten.mul.Tensor(squeeze_28, primals_29);  primals_29 = None
        unsqueeze_737: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_763, 0);  mul_763 = None
        unsqueeze_738: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_737, 2);  unsqueeze_737 = None
        unsqueeze_739: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_738, 3);  unsqueeze_738 = None
        mul_764: f32[64, 64, 8, 8] = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_736);  sub_225 = unsqueeze_736 = None
        sub_227: f32[64, 64, 8, 8] = torch.ops.aten.sub.Tensor(where_40, mul_764);  where_40 = mul_764 = None
        sub_228: f32[64, 64, 8, 8] = torch.ops.aten.sub.Tensor(sub_227, unsqueeze_733);  sub_227 = unsqueeze_733 = None
        mul_765: f32[64, 64, 8, 8] = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_739);  sub_228 = unsqueeze_739 = None
        mul_766: f32[64] = torch.ops.aten.mul.Tensor(sum_89, squeeze_28);  sum_89 = squeeze_28 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:150, code: out = self.conv2(out)
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_765, relu_7, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_765 = primals_28 = None
        getitem_237: f32[64, 64, 8, 8] = convolution_backward_43[0]
        getitem_238: f32[64, 64, 3, 3] = convolution_backward_43[1];  convolution_backward_43 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:148, code: out = self.relu(out)
        le_41: b8[64, 64, 8, 8] = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_41: f32[64, 64, 8, 8] = torch.ops.aten.where.self(le_41, scalar_tensor, getitem_237);  le_41 = getitem_237 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:147, code: out = self.bn1(out)
        sum_90: f32[64] = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
        sub_229: f32[64, 64, 8, 8] = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_742);  convolution_8 = unsqueeze_742 = None
        mul_767: f32[64, 64, 8, 8] = torch.ops.aten.mul.Tensor(where_41, sub_229)
        sum_91: f32[64] = torch.ops.aten.sum.dim_IntList(mul_767, [0, 2, 3]);  mul_767 = None
        mul_768: f32[64] = torch.ops.aten.mul.Tensor(sum_90, 0.000244140625)
        unsqueeze_743: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_768, 0);  mul_768 = None
        unsqueeze_744: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_743, 2);  unsqueeze_743 = None
        unsqueeze_745: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_744, 3);  unsqueeze_744 = None
        mul_769: f32[64] = torch.ops.aten.mul.Tensor(sum_91, 0.000244140625)
        mul_770: f32[64] = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_771: f32[64] = torch.ops.aten.mul.Tensor(mul_769, mul_770);  mul_769 = mul_770 = None
        unsqueeze_746: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_771, 0);  mul_771 = None
        unsqueeze_747: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_746, 2);  unsqueeze_746 = None
        unsqueeze_748: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_747, 3);  unsqueeze_747 = None
        mul_772: f32[64] = torch.ops.aten.mul.Tensor(squeeze_25, primals_26);  primals_26 = None
        unsqueeze_749: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_772, 0);  mul_772 = None
        unsqueeze_750: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_749, 2);  unsqueeze_749 = None
        unsqueeze_751: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_750, 3);  unsqueeze_750 = None
        mul_773: f32[64, 64, 8, 8] = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_748);  sub_229 = unsqueeze_748 = None
        sub_231: f32[64, 64, 8, 8] = torch.ops.aten.sub.Tensor(where_41, mul_773);  where_41 = mul_773 = None
        sub_232: f32[64, 64, 8, 8] = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_745);  sub_231 = unsqueeze_745 = None
        mul_774: f32[64, 64, 8, 8] = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_751);  sub_232 = unsqueeze_751 = None
        mul_775: f32[64] = torch.ops.aten.mul.Tensor(sum_91, squeeze_25);  sum_91 = squeeze_25 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_774, relu_6, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_774 = primals_25 = None
        getitem_240: f32[64, 256, 8, 8] = convolution_backward_44[0]
        getitem_241: f32[64, 256, 1, 1] = convolution_backward_44[1];  convolution_backward_44 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        add_294: f32[64, 256, 8, 8] = torch.ops.aten.add.Tensor(where_39, getitem_240);  where_39 = getitem_240 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:161, code: out = self.relu(out)
        le_42: b8[64, 256, 8, 8] = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_42: f32[64, 256, 8, 8] = torch.ops.aten.where.self(le_42, scalar_tensor, add_294);  le_42 = add_294 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:155, code: out = self.bn3(out)
        sum_92: f32[256] = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
        sub_233: f32[64, 256, 8, 8] = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_754);  convolution_7 = unsqueeze_754 = None
        mul_776: f32[64, 256, 8, 8] = torch.ops.aten.mul.Tensor(where_42, sub_233)
        sum_93: f32[256] = torch.ops.aten.sum.dim_IntList(mul_776, [0, 2, 3]);  mul_776 = None
        mul_777: f32[256] = torch.ops.aten.mul.Tensor(sum_92, 0.000244140625)
        unsqueeze_755: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_777, 0);  mul_777 = None
        unsqueeze_756: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_755, 2);  unsqueeze_755 = None
        unsqueeze_757: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_756, 3);  unsqueeze_756 = None
        mul_778: f32[256] = torch.ops.aten.mul.Tensor(sum_93, 0.000244140625)
        mul_779: f32[256] = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_780: f32[256] = torch.ops.aten.mul.Tensor(mul_778, mul_779);  mul_778 = mul_779 = None
        unsqueeze_758: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_780, 0);  mul_780 = None
        unsqueeze_759: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_758, 2);  unsqueeze_758 = None
        unsqueeze_760: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_759, 3);  unsqueeze_759 = None
        mul_781: f32[256] = torch.ops.aten.mul.Tensor(squeeze_22, primals_23);  primals_23 = None
        unsqueeze_761: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_781, 0);  mul_781 = None
        unsqueeze_762: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_761, 2);  unsqueeze_761 = None
        unsqueeze_763: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_762, 3);  unsqueeze_762 = None
        mul_782: f32[64, 256, 8, 8] = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_760);  sub_233 = unsqueeze_760 = None
        sub_235: f32[64, 256, 8, 8] = torch.ops.aten.sub.Tensor(where_42, mul_782);  mul_782 = None
        sub_236: f32[64, 256, 8, 8] = torch.ops.aten.sub.Tensor(sub_235, unsqueeze_757);  sub_235 = unsqueeze_757 = None
        mul_783: f32[64, 256, 8, 8] = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_763);  sub_236 = unsqueeze_763 = None
        mul_784: f32[256] = torch.ops.aten.mul.Tensor(sum_93, squeeze_22);  sum_93 = squeeze_22 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:154, code: out = self.conv3(out)
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_783, relu_5, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_783 = primals_22 = None
        getitem_243: f32[64, 64, 8, 8] = convolution_backward_45[0]
        getitem_244: f32[256, 64, 1, 1] = convolution_backward_45[1];  convolution_backward_45 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:152, code: out = self.relu(out)
        le_43: b8[64, 64, 8, 8] = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_43: f32[64, 64, 8, 8] = torch.ops.aten.where.self(le_43, scalar_tensor, getitem_243);  le_43 = getitem_243 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:151, code: out = self.bn2(out)
        sum_94: f32[64] = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
        sub_237: f32[64, 64, 8, 8] = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_766);  convolution_6 = unsqueeze_766 = None
        mul_785: f32[64, 64, 8, 8] = torch.ops.aten.mul.Tensor(where_43, sub_237)
        sum_95: f32[64] = torch.ops.aten.sum.dim_IntList(mul_785, [0, 2, 3]);  mul_785 = None
        mul_786: f32[64] = torch.ops.aten.mul.Tensor(sum_94, 0.000244140625)
        unsqueeze_767: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_786, 0);  mul_786 = None
        unsqueeze_768: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_767, 2);  unsqueeze_767 = None
        unsqueeze_769: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_768, 3);  unsqueeze_768 = None
        mul_787: f32[64] = torch.ops.aten.mul.Tensor(sum_95, 0.000244140625)
        mul_788: f32[64] = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_789: f32[64] = torch.ops.aten.mul.Tensor(mul_787, mul_788);  mul_787 = mul_788 = None
        unsqueeze_770: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_789, 0);  mul_789 = None
        unsqueeze_771: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_770, 2);  unsqueeze_770 = None
        unsqueeze_772: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_771, 3);  unsqueeze_771 = None
        mul_790: f32[64] = torch.ops.aten.mul.Tensor(squeeze_19, primals_20);  primals_20 = None
        unsqueeze_773: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_790, 0);  mul_790 = None
        unsqueeze_774: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_773, 2);  unsqueeze_773 = None
        unsqueeze_775: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_774, 3);  unsqueeze_774 = None
        mul_791: f32[64, 64, 8, 8] = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_772);  sub_237 = unsqueeze_772 = None
        sub_239: f32[64, 64, 8, 8] = torch.ops.aten.sub.Tensor(where_43, mul_791);  where_43 = mul_791 = None
        sub_240: f32[64, 64, 8, 8] = torch.ops.aten.sub.Tensor(sub_239, unsqueeze_769);  sub_239 = unsqueeze_769 = None
        mul_792: f32[64, 64, 8, 8] = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_775);  sub_240 = unsqueeze_775 = None
        mul_793: f32[64] = torch.ops.aten.mul.Tensor(sum_95, squeeze_19);  sum_95 = squeeze_19 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:150, code: out = self.conv2(out)
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_792, relu_4, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_792 = primals_19 = None
        getitem_246: f32[64, 64, 8, 8] = convolution_backward_46[0]
        getitem_247: f32[64, 64, 3, 3] = convolution_backward_46[1];  convolution_backward_46 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:148, code: out = self.relu(out)
        le_44: b8[64, 64, 8, 8] = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_44: f32[64, 64, 8, 8] = torch.ops.aten.where.self(le_44, scalar_tensor, getitem_246);  le_44 = getitem_246 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:147, code: out = self.bn1(out)
        sum_96: f32[64] = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
        sub_241: f32[64, 64, 8, 8] = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_778);  convolution_5 = unsqueeze_778 = None
        mul_794: f32[64, 64, 8, 8] = torch.ops.aten.mul.Tensor(where_44, sub_241)
        sum_97: f32[64] = torch.ops.aten.sum.dim_IntList(mul_794, [0, 2, 3]);  mul_794 = None
        mul_795: f32[64] = torch.ops.aten.mul.Tensor(sum_96, 0.000244140625)
        unsqueeze_779: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_795, 0);  mul_795 = None
        unsqueeze_780: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_779, 2);  unsqueeze_779 = None
        unsqueeze_781: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_780, 3);  unsqueeze_780 = None
        mul_796: f32[64] = torch.ops.aten.mul.Tensor(sum_97, 0.000244140625)
        mul_797: f32[64] = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_798: f32[64] = torch.ops.aten.mul.Tensor(mul_796, mul_797);  mul_796 = mul_797 = None
        unsqueeze_782: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_798, 0);  mul_798 = None
        unsqueeze_783: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_782, 2);  unsqueeze_782 = None
        unsqueeze_784: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_783, 3);  unsqueeze_783 = None
        mul_799: f32[64] = torch.ops.aten.mul.Tensor(squeeze_16, primals_17);  primals_17 = None
        unsqueeze_785: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_799, 0);  mul_799 = None
        unsqueeze_786: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_785, 2);  unsqueeze_785 = None
        unsqueeze_787: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_786, 3);  unsqueeze_786 = None
        mul_800: f32[64, 64, 8, 8] = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_784);  sub_241 = unsqueeze_784 = None
        sub_243: f32[64, 64, 8, 8] = torch.ops.aten.sub.Tensor(where_44, mul_800);  where_44 = mul_800 = None
        sub_244: f32[64, 64, 8, 8] = torch.ops.aten.sub.Tensor(sub_243, unsqueeze_781);  sub_243 = unsqueeze_781 = None
        mul_801: f32[64, 64, 8, 8] = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_787);  sub_244 = unsqueeze_787 = None
        mul_802: f32[64] = torch.ops.aten.mul.Tensor(sum_97, squeeze_16);  sum_97 = squeeze_16 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_801, relu_3, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_801 = primals_16 = None
        getitem_249: f32[64, 256, 8, 8] = convolution_backward_47[0]
        getitem_250: f32[64, 256, 1, 1] = convolution_backward_47[1];  convolution_backward_47 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        add_295: f32[64, 256, 8, 8] = torch.ops.aten.add.Tensor(where_42, getitem_249);  where_42 = getitem_249 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:161, code: out = self.relu(out)
        le_45: b8[64, 256, 8, 8] = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_45: f32[64, 256, 8, 8] = torch.ops.aten.where.self(le_45, scalar_tensor, add_295);  le_45 = add_295 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:158, code: identity = self.downsample(x)
        sum_98: f32[256] = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
        sub_245: f32[64, 256, 8, 8] = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_790);  convolution_4 = unsqueeze_790 = None
        mul_803: f32[64, 256, 8, 8] = torch.ops.aten.mul.Tensor(where_45, sub_245)
        sum_99: f32[256] = torch.ops.aten.sum.dim_IntList(mul_803, [0, 2, 3]);  mul_803 = None
        mul_804: f32[256] = torch.ops.aten.mul.Tensor(sum_98, 0.000244140625)
        unsqueeze_791: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_804, 0);  mul_804 = None
        unsqueeze_792: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_791, 2);  unsqueeze_791 = None
        unsqueeze_793: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_792, 3);  unsqueeze_792 = None
        mul_805: f32[256] = torch.ops.aten.mul.Tensor(sum_99, 0.000244140625)
        mul_806: f32[256] = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_807: f32[256] = torch.ops.aten.mul.Tensor(mul_805, mul_806);  mul_805 = mul_806 = None
        unsqueeze_794: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_807, 0);  mul_807 = None
        unsqueeze_795: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_794, 2);  unsqueeze_794 = None
        unsqueeze_796: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_795, 3);  unsqueeze_795 = None
        mul_808: f32[256] = torch.ops.aten.mul.Tensor(squeeze_13, primals_14);  primals_14 = None
        unsqueeze_797: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_808, 0);  mul_808 = None
        unsqueeze_798: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_797, 2);  unsqueeze_797 = None
        unsqueeze_799: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_798, 3);  unsqueeze_798 = None
        mul_809: f32[64, 256, 8, 8] = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_796);  sub_245 = unsqueeze_796 = None
        sub_247: f32[64, 256, 8, 8] = torch.ops.aten.sub.Tensor(where_45, mul_809);  mul_809 = None
        sub_248: f32[64, 256, 8, 8] = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_793);  sub_247 = None
        mul_810: f32[64, 256, 8, 8] = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_799);  sub_248 = unsqueeze_799 = None
        mul_811: f32[256] = torch.ops.aten.mul.Tensor(sum_99, squeeze_13);  sum_99 = squeeze_13 = None
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_810, getitem_2, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_810 = primals_13 = None
        getitem_252: f32[64, 64, 8, 8] = convolution_backward_48[0]
        getitem_253: f32[256, 64, 1, 1] = convolution_backward_48[1];  convolution_backward_48 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:155, code: out = self.bn3(out)
        sub_249: f32[64, 256, 8, 8] = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_802);  convolution_3 = unsqueeze_802 = None
        mul_812: f32[64, 256, 8, 8] = torch.ops.aten.mul.Tensor(where_45, sub_249)
        sum_101: f32[256] = torch.ops.aten.sum.dim_IntList(mul_812, [0, 2, 3]);  mul_812 = None
        mul_814: f32[256] = torch.ops.aten.mul.Tensor(sum_101, 0.000244140625)
        mul_815: f32[256] = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_816: f32[256] = torch.ops.aten.mul.Tensor(mul_814, mul_815);  mul_814 = mul_815 = None
        unsqueeze_806: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_816, 0);  mul_816 = None
        unsqueeze_807: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_806, 2);  unsqueeze_806 = None
        unsqueeze_808: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_807, 3);  unsqueeze_807 = None
        mul_817: f32[256] = torch.ops.aten.mul.Tensor(squeeze_10, primals_11);  primals_11 = None
        unsqueeze_809: f32[1, 256] = torch.ops.aten.unsqueeze.default(mul_817, 0);  mul_817 = None
        unsqueeze_810: f32[1, 256, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_809, 2);  unsqueeze_809 = None
        unsqueeze_811: f32[1, 256, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_810, 3);  unsqueeze_810 = None
        mul_818: f32[64, 256, 8, 8] = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_808);  sub_249 = unsqueeze_808 = None
        sub_251: f32[64, 256, 8, 8] = torch.ops.aten.sub.Tensor(where_45, mul_818);  where_45 = mul_818 = None
        sub_252: f32[64, 256, 8, 8] = torch.ops.aten.sub.Tensor(sub_251, unsqueeze_793);  sub_251 = unsqueeze_793 = None
        mul_819: f32[64, 256, 8, 8] = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_811);  sub_252 = unsqueeze_811 = None
        mul_820: f32[256] = torch.ops.aten.mul.Tensor(sum_101, squeeze_10);  sum_101 = squeeze_10 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:154, code: out = self.conv3(out)
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_819, relu_2, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_819 = primals_10 = None
        getitem_255: f32[64, 64, 8, 8] = convolution_backward_49[0]
        getitem_256: f32[256, 64, 1, 1] = convolution_backward_49[1];  convolution_backward_49 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:152, code: out = self.relu(out)
        le_46: b8[64, 64, 8, 8] = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_46: f32[64, 64, 8, 8] = torch.ops.aten.where.self(le_46, scalar_tensor, getitem_255);  le_46 = getitem_255 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:151, code: out = self.bn2(out)
        sum_102: f32[64] = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
        sub_253: f32[64, 64, 8, 8] = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_814);  convolution_2 = unsqueeze_814 = None
        mul_821: f32[64, 64, 8, 8] = torch.ops.aten.mul.Tensor(where_46, sub_253)
        sum_103: f32[64] = torch.ops.aten.sum.dim_IntList(mul_821, [0, 2, 3]);  mul_821 = None
        mul_822: f32[64] = torch.ops.aten.mul.Tensor(sum_102, 0.000244140625)
        unsqueeze_815: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_822, 0);  mul_822 = None
        unsqueeze_816: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_815, 2);  unsqueeze_815 = None
        unsqueeze_817: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_816, 3);  unsqueeze_816 = None
        mul_823: f32[64] = torch.ops.aten.mul.Tensor(sum_103, 0.000244140625)
        mul_824: f32[64] = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_825: f32[64] = torch.ops.aten.mul.Tensor(mul_823, mul_824);  mul_823 = mul_824 = None
        unsqueeze_818: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_825, 0);  mul_825 = None
        unsqueeze_819: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_818, 2);  unsqueeze_818 = None
        unsqueeze_820: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_819, 3);  unsqueeze_819 = None
        mul_826: f32[64] = torch.ops.aten.mul.Tensor(squeeze_7, primals_8);  primals_8 = None
        unsqueeze_821: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_826, 0);  mul_826 = None
        unsqueeze_822: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_821, 2);  unsqueeze_821 = None
        unsqueeze_823: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_822, 3);  unsqueeze_822 = None
        mul_827: f32[64, 64, 8, 8] = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_820);  sub_253 = unsqueeze_820 = None
        sub_255: f32[64, 64, 8, 8] = torch.ops.aten.sub.Tensor(where_46, mul_827);  where_46 = mul_827 = None
        sub_256: f32[64, 64, 8, 8] = torch.ops.aten.sub.Tensor(sub_255, unsqueeze_817);  sub_255 = unsqueeze_817 = None
        mul_828: f32[64, 64, 8, 8] = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_823);  sub_256 = unsqueeze_823 = None
        mul_829: f32[64] = torch.ops.aten.mul.Tensor(sum_103, squeeze_7);  sum_103 = squeeze_7 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:150, code: out = self.conv2(out)
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_828, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_828 = primals_7 = None
        getitem_258: f32[64, 64, 8, 8] = convolution_backward_50[0]
        getitem_259: f32[64, 64, 3, 3] = convolution_backward_50[1];  convolution_backward_50 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:148, code: out = self.relu(out)
        le_47: b8[64, 64, 8, 8] = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_47: f32[64, 64, 8, 8] = torch.ops.aten.where.self(le_47, scalar_tensor, getitem_258);  le_47 = getitem_258 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:147, code: out = self.bn1(out)
        sum_104: f32[64] = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
        sub_257: f32[64, 64, 8, 8] = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_826);  convolution_1 = unsqueeze_826 = None
        mul_830: f32[64, 64, 8, 8] = torch.ops.aten.mul.Tensor(where_47, sub_257)
        sum_105: f32[64] = torch.ops.aten.sum.dim_IntList(mul_830, [0, 2, 3]);  mul_830 = None
        mul_831: f32[64] = torch.ops.aten.mul.Tensor(sum_104, 0.000244140625)
        unsqueeze_827: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_831, 0);  mul_831 = None
        unsqueeze_828: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_827, 2);  unsqueeze_827 = None
        unsqueeze_829: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_828, 3);  unsqueeze_828 = None
        mul_832: f32[64] = torch.ops.aten.mul.Tensor(sum_105, 0.000244140625)
        mul_833: f32[64] = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_834: f32[64] = torch.ops.aten.mul.Tensor(mul_832, mul_833);  mul_832 = mul_833 = None
        unsqueeze_830: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_834, 0);  mul_834 = None
        unsqueeze_831: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_830, 2);  unsqueeze_830 = None
        unsqueeze_832: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_831, 3);  unsqueeze_831 = None
        mul_835: f32[64] = torch.ops.aten.mul.Tensor(squeeze_4, primals_5);  primals_5 = None
        unsqueeze_833: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_835, 0);  mul_835 = None
        unsqueeze_834: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_833, 2);  unsqueeze_833 = None
        unsqueeze_835: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_834, 3);  unsqueeze_834 = None
        mul_836: f32[64, 64, 8, 8] = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_832);  sub_257 = unsqueeze_832 = None
        sub_259: f32[64, 64, 8, 8] = torch.ops.aten.sub.Tensor(where_47, mul_836);  where_47 = mul_836 = None
        sub_260: f32[64, 64, 8, 8] = torch.ops.aten.sub.Tensor(sub_259, unsqueeze_829);  sub_259 = unsqueeze_829 = None
        mul_837: f32[64, 64, 8, 8] = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_835);  sub_260 = unsqueeze_835 = None
        mul_838: f32[64] = torch.ops.aten.mul.Tensor(sum_105, squeeze_4);  sum_105 = squeeze_4 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_837, getitem_2, primals_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_837 = getitem_2 = primals_4 = None
        getitem_261: f32[64, 64, 8, 8] = convolution_backward_51[0]
        getitem_262: f32[64, 64, 1, 1] = convolution_backward_51[1];  convolution_backward_51 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:146, code: out = self.conv1(x)
        add_296: f32[64, 64, 8, 8] = torch.ops.aten.add.Tensor(getitem_252, getitem_261);  getitem_252 = getitem_261 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:271, code: x = self.maxpool(x)
        max_pool2d_with_indices_backward: f32[64, 64, 16, 16] = torch.ops.aten.max_pool2d_with_indices_backward.default(add_296, relu, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_3);  add_296 = getitem_3 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:270, code: x = self.relu(x)
        le_48: b8[64, 64, 16, 16] = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_48: f32[64, 64, 16, 16] = torch.ops.aten.where.self(le_48, scalar_tensor, max_pool2d_with_indices_backward);  le_48 = scalar_tensor = max_pool2d_with_indices_backward = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:269, code: x = self.bn1(x)
        sum_106: f32[64] = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
        sub_261: f32[64, 64, 16, 16] = torch.ops.aten.sub.Tensor(convolution, unsqueeze_838);  convolution = unsqueeze_838 = None
        mul_839: f32[64, 64, 16, 16] = torch.ops.aten.mul.Tensor(where_48, sub_261)
        sum_107: f32[64] = torch.ops.aten.sum.dim_IntList(mul_839, [0, 2, 3]);  mul_839 = None
        mul_840: f32[64] = torch.ops.aten.mul.Tensor(sum_106, 6.103515625e-05)
        unsqueeze_839: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_840, 0);  mul_840 = None
        unsqueeze_840: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_839, 2);  unsqueeze_839 = None
        unsqueeze_841: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_840, 3);  unsqueeze_840 = None
        mul_841: f32[64] = torch.ops.aten.mul.Tensor(sum_107, 6.103515625e-05)
        mul_842: f32[64] = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_843: f32[64] = torch.ops.aten.mul.Tensor(mul_841, mul_842);  mul_841 = mul_842 = None
        unsqueeze_842: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_843, 0);  mul_843 = None
        unsqueeze_843: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_842, 2);  unsqueeze_842 = None
        unsqueeze_844: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_843, 3);  unsqueeze_843 = None
        mul_844: f32[64] = torch.ops.aten.mul.Tensor(squeeze_1, primals_2);  primals_2 = None
        unsqueeze_845: f32[1, 64] = torch.ops.aten.unsqueeze.default(mul_844, 0);  mul_844 = None
        unsqueeze_846: f32[1, 64, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_845, 2);  unsqueeze_845 = None
        unsqueeze_847: f32[1, 64, 1, 1] = torch.ops.aten.unsqueeze.default(unsqueeze_846, 3);  unsqueeze_846 = None
        mul_845: f32[64, 64, 16, 16] = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_844);  sub_261 = unsqueeze_844 = None
        sub_263: f32[64, 64, 16, 16] = torch.ops.aten.sub.Tensor(where_48, mul_845);  where_48 = mul_845 = None
        sub_264: f32[64, 64, 16, 16] = torch.ops.aten.sub.Tensor(sub_263, unsqueeze_841);  sub_263 = unsqueeze_841 = None
        mul_846: f32[64, 64, 16, 16] = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_847);  sub_264 = unsqueeze_847 = None
        mul_847: f32[64] = torch.ops.aten.mul.Tensor(sum_107, squeeze_1);  sum_107 = squeeze_1 = None
        
        # File: /opt/conda/lib/python3.10/site-packages/torchvision/models/resnet.py:268, code: x = self.conv1(x)
        convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_846, primals_321, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_846 = primals_321 = primals_1 = None
        getitem_265: f32[64, 3, 7, 7] = convolution_backward_52[1];  convolution_backward_52 = None
        return [getitem_265, mul_847, sum_106, getitem_262, mul_838, sum_104, getitem_259, mul_829, sum_102, getitem_256, mul_820, sum_98, getitem_253, mul_811, sum_98, getitem_250, mul_802, sum_96, getitem_247, mul_793, sum_94, getitem_244, mul_784, sum_92, getitem_241, mul_775, sum_90, getitem_238, mul_766, sum_88, getitem_235, mul_757, sum_86, getitem_232, mul_748, sum_84, getitem_229, mul_739, sum_82, getitem_226, mul_730, sum_78, getitem_223, mul_721, sum_78, getitem_220, mul_712, sum_76, getitem_217, mul_703, sum_74, getitem_214, mul_694, sum_72, getitem_211, mul_685, sum_70, getitem_208, mul_676, sum_68, getitem_205, mul_667, sum_66, getitem_202, mul_658, sum_64, getitem_199, mul_649, sum_62, getitem_196, mul_640, sum_60, getitem_193, mul_631, sum_58, getitem_190, mul_622, sum_56, getitem_187, mul_613, sum_52, getitem_184, mul_604, sum_52, getitem_181, mul_595, sum_50, getitem_178, mul_586, sum_48, getitem_175, mul_577, sum_46, getitem_172, mul_568, sum_44, getitem_169, mul_559, sum_42, getitem_166, mul_550, sum_40, getitem_163, mul_541, sum_38, getitem_160, mul_532, sum_36, getitem_157, mul_523, sum_34, getitem_154, mul_514, sum_32, getitem_151, mul_505, sum_30, getitem_148, mul_496, sum_28, getitem_145, mul_487, sum_26, getitem_142, mul_478, sum_24, getitem_139, mul_469, sum_22, getitem_136, mul_460, sum_20, getitem_133, mul_451, sum_18, getitem_130, mul_442, sum_14, getitem_127, mul_433, sum_14, getitem_124, mul_424, sum_12, getitem_121, mul_415, sum_10, getitem_118, mul_406, sum_8, getitem_115, mul_397, sum_6, getitem_112, mul_388, sum_4, getitem_109, mul_379, sum_2, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
        