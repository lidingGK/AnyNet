import torch
import onnx

from mmcv.onnx.symbolic import register_extra_symbolics
register_extra_symbolics(11)


from mmcv.tensorrt import (TRTWrapper, onnx2trt, save_trt_engine,
                                   is_tensorrt_plugin_loaded)

assert is_tensorrt_plugin_loaded(), 'Requires to complie TensorRT plugins in mmcv'

onnx_file = '/workspace/depth/AnyNet/anynet_cspn.onnx'
trt_file = 'sample.trt'
onnx_model = onnx_file

# Model input
inputs = torch.rand(1, 3, 720, 1280).cuda()
# Model input shape info
opt_shape_dict = {
    'left': [list(inputs.shape),
              list(inputs.shape),
              list(inputs.shape)],
    'right': [list(inputs.shape),
              list(inputs.shape),
              list(inputs.shape)]
}

# Create TensorRT engine
max_workspace_size = 1 << 31
trt_engine = onnx2trt(
    onnx_model,
    opt_shape_dict,
    max_workspace_size=max_workspace_size)

# Save TensorRT engine
save_trt_engine(trt_engine, trt_file)

# Run inference with TensorRT
trt_model = TRTWrapper(trt_file, ['left', 'right'], ['output'])

with torch.no_grad():
    trt_outputs = trt_model({'left': inputs, 'right': inputs})
    output = trt_outputs['output']
    print(output)