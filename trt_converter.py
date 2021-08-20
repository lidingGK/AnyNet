# from torch2trt.torch2trt import *
# import torch

# # @tensorrt_converter('torch.sub')
# # def convert_sub(ctx):
# #     input_a = ctx.method_args[0]
# #     input_b = ctx.method_args[1]
# #     output = ctx.method_return
# #     input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
# #     input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape) - 1)
# #     layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.SUB)
# #     output._trt = layer.get_output(0)



# # import tensorrt as trt
# # from torch2trt import tensorrt_converter

# @tensorrt_converter('torch.nn.ReLU.forward')
# def convert_ReLU(ctx):
#     input = ctx.method_args[1]
#     output = ctx.method_return
#     layer = ctx.network.add_activation(input=input._trt, type=trt.ActivationType.RELU)  
#     output._trt = layer.get_output(0)


# @tensorrt_converter('torch.zeros')
# def convert_zeros(ctx):
#     input = ctx.method_args[0]
#     print(input)
#     output = ctx.method_return

#     val_tensor = torch.ones(tuple(input), dtype=torch.float32).cpu().numpy()
#     layer = ctx.network.add_constant(tuple(input), val_tensor)
#     output._trt = layer.get_output(0)


# class Zeros(torch.nn.Module):
#     def __init__(self):
#         super(Zeros, self).__init__()

#     def forward(self, shape):
#         return torch.zeros(shape)
# model = Zeros()
# shape = [1, 2, 3, 4]

# print(model(shape))

# # from torch2trt import torch2trt
# # model_trt = torch2trt(model, [torch.tensor(shape, dtype=torch.int32)])
# # y = torch.tensor([2], dtype=torch.int32).cuda()
# # print(model_trt(y))



from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.tensor')
def convert_mod(ctx):
    print('='*20)
    input = ctx.method_args[0]
    output = ctx.method_return
    layer = ctx.network.add_constant(tuple(output.shape), output.detach().cpu().numpy() )
    output._trt = layer.get_output(0)


class TorchTensor(torch.nn.Module):
    def __init__(self):
        super(TorchTensor, self).__init__()

    def forward(self, x):
        return x + torch.tensor([[1., 2., 3.], [4., 5., 6.]], device=torch.device('cuda'))

model = TorchTensor()
print(model(1))

import torch2trt
x = torch.ones((2, 3)).cuda()
model_trt = torch2trt.torch2trt(model, [x])



@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 3)])
def test_tensor_creation():
    return TorchTensor()

test_tensor_creation()