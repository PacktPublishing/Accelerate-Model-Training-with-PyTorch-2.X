class GraphModule(torch.nn.Module):
    def forward(self, primals_1: f32[32, 1, 3, 3], primals_2: f32[32], primals_3: f32[64, 32, 3, 3], primals_4: f32[64], primals_5: f32[512, 3136], primals_6: f32[512], primals_7: f32[10, 512], primals_8: f32[10], primals_9: f32[64, 1, 28, 28]):
        # File: /tmp/ipykernel_2329238/3432332073.py:19, code: out = self.layer1(x)
        convolution: f32[64, 32, 28, 28] = torch.ops.aten.convolution.default(primals_9, primals_1, primals_2, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_2 = None
        relu: f32[64, 32, 28, 28] = torch.ops.aten.relu.default(convolution);  convolution = None
        max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu, [2, 2], [2, 2])
        getitem: f32[64, 32, 14, 14] = max_pool2d_with_indices[0]
        getitem_1: i64[64, 32, 14, 14] = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
        
        # File: /tmp/ipykernel_2329238/3432332073.py:20, code: out = self.layer2(out)
        convolution_1: f32[64, 64, 14, 14] = torch.ops.aten.convolution.default(getitem, primals_3, primals_4, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_4 = None
        relu_1: f32[64, 64, 14, 14] = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
        max_pool2d_with_indices_1 = torch.ops.aten.max_pool2d_with_indices.default(relu_1, [2, 2], [2, 2])
        getitem_2: f32[64, 64, 7, 7] = max_pool2d_with_indices_1[0]
        getitem_3: i64[64, 64, 7, 7] = max_pool2d_with_indices_1[1];  max_pool2d_with_indices_1 = None
        
        # File: /tmp/ipykernel_2329238/3432332073.py:21, code: out = out.reshape(out.size(0), -1)
        view: f32[64, 3136] = torch.ops.aten.view.default(getitem_2, [64, 3136]);  getitem_2 = None
        
        # File: /tmp/ipykernel_2329238/3432332073.py:22, code: out = self.fc1(out)
        permute: f32[3136, 512] = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        addmm: f32[64, 512] = torch.ops.aten.addmm.default(primals_6, view, permute);  primals_6 = None
        
        # File: /tmp/ipykernel_2329238/3432332073.py:23, code: out = self.fc2(out)
        permute_1: f32[512, 10] = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        addmm_1: f32[64, 10] = torch.ops.aten.addmm.default(primals_8, addmm, permute_1);  primals_8 = None
        permute_2: f32[10, 512] = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        
        # File: /tmp/ipykernel_2329238/3432332073.py:22, code: out = self.fc1(out)
        permute_6: f32[512, 3136] = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return [addmm_1, primals_1, primals_3, primals_9, relu, getitem, getitem_1, relu_1, getitem_3, view, addmm, permute_2, permute_6]
        