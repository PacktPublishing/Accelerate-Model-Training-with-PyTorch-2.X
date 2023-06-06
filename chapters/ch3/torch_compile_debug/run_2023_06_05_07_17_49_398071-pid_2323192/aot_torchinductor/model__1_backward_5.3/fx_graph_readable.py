class GraphModule(torch.nn.Module):
    def forward(self, primals_1: f32[32, 1, 3, 3], primals_3: f32[64, 32, 3, 3], primals_9: f32[48, 1, 28, 28], relu: f32[48, 32, 28, 28], getitem: f32[48, 32, 14, 14], getitem_1: i64[48, 32, 14, 14], relu_1: f32[48, 64, 14, 14], getitem_3: i64[48, 64, 7, 7], view: f32[48, 3136], addmm: f32[48, 512], permute_2: f32[10, 512], permute_6: f32[512, 3136], tangents_1: f32[48, 10]):
        # File: /tmp/ipykernel_2323192/3432332073.py:23, code: out = self.fc2(out)
        mm: f32[48, 512] = torch.ops.aten.mm.default(tangents_1, permute_2);  permute_2 = None
        permute_3: f32[10, 48] = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: f32[10, 512] = torch.ops.aten.mm.default(permute_3, addmm);  permute_3 = addmm = None
        permute_4: f32[512, 10] = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        sum_1: f32[1, 10] = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_1: f32[10] = torch.ops.aten.view.default(sum_1, [10]);  sum_1 = None
        permute_5: f32[10, 512] = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        
        # File: /tmp/ipykernel_2323192/3432332073.py:22, code: out = self.fc1(out)
        mm_2: f32[48, 3136] = torch.ops.aten.mm.default(mm, permute_6);  permute_6 = None
        permute_7: f32[512, 48] = torch.ops.aten.permute.default(mm, [1, 0])
        mm_3: f32[512, 3136] = torch.ops.aten.mm.default(permute_7, view);  permute_7 = view = None
        permute_8: f32[3136, 512] = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
        sum_2: f32[1, 512] = torch.ops.aten.sum.dim_IntList(mm, [0], True);  mm = None
        view_2: f32[512] = torch.ops.aten.view.default(sum_2, [512]);  sum_2 = None
        permute_9: f32[512, 3136] = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        
        # File: /tmp/ipykernel_2323192/3432332073.py:21, code: out = out.reshape(out.size(0), -1)
        view_3: f32[48, 64, 7, 7] = torch.ops.aten.view.default(mm_2, [48, 64, 7, 7]);  mm_2 = None
        
        # File: /tmp/ipykernel_2323192/3432332073.py:20, code: out = self.layer2(out)
        max_pool2d_with_indices_backward: f32[48, 64, 14, 14] = torch.ops.aten.max_pool2d_with_indices_backward.default(view_3, relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_3);  view_3 = getitem_3 = None
        le: b8[48, 64, 14, 14] = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        scalar_tensor: f32[] = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
        where: f32[48, 64, 14, 14] = torch.ops.aten.where.self(le, scalar_tensor, max_pool2d_with_indices_backward);  le = max_pool2d_with_indices_backward = None
        convolution_backward = torch.ops.aten.convolution_backward.default(where, getitem, primals_3, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  where = getitem = primals_3 = None
        getitem_4: f32[48, 32, 14, 14] = convolution_backward[0]
        getitem_5: f32[64, 32, 3, 3] = convolution_backward[1]
        getitem_6: f32[64] = convolution_backward[2];  convolution_backward = None
        
        # File: /tmp/ipykernel_2323192/3432332073.py:19, code: out = self.layer1(x)
        max_pool2d_with_indices_backward_1: f32[48, 32, 28, 28] = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_4, relu, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_1);  getitem_4 = getitem_1 = None
        le_1: b8[48, 32, 28, 28] = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_1: f32[48, 32, 28, 28] = torch.ops.aten.where.self(le_1, scalar_tensor, max_pool2d_with_indices_backward_1);  le_1 = scalar_tensor = max_pool2d_with_indices_backward_1 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(where_1, primals_9, primals_1, [32], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, True]);  where_1 = primals_9 = primals_1 = None
        getitem_8: f32[32, 1, 3, 3] = convolution_backward_1[1]
        getitem_9: f32[32] = convolution_backward_1[2];  convolution_backward_1 = None
        return [getitem_8, getitem_9, getitem_5, getitem_6, permute_9, view_2, permute_5, view_1, None]
        