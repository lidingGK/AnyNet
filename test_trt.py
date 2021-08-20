import torch
from zed_camera import ZED_camera_and_sensor

import time
import pyzed.sl as sl

from mmcv.onnx.symbolic import register_extra_symbolics
from mmcv.tensorrt import TRTWrapper, is_tensorrt_plugin_loaded

register_extra_symbolics(11)
assert is_tensorrt_plugin_loaded(), 'Requires to complie TensorRT plugins in mmcv'

trt_file = 'sample.trt'
trt_model = TRTWrapper(trt_file, ['left', 'right'], ['output'])
for _ in range(100):
    inputs = torch.rand(1, 3, 720, 1280).cuda()
    with torch.no_grad():
        trt_outputs = trt_model({'left': inputs, 'right': inputs})
        output = trt_outputs['output']
        print(output.shape)
        print('='*100)

# def img2tensor(img):
#     return torch.from_numpy(img).cuda().permute(2,0,1).unsqueeze(0)

# def tensor2img(img_t):
#     return img_t.squeeze().cpu().numpy()

# camera = ZED_camera_and_sensor(resolution=720, fps=30)
# camera.runtime_params.sensing_mode = sl.SENSING_MODE.FILL

# print("starting")
# start_time = time.perf_counter()

# for i in range(50):
#     camera.captureImage()

# print(500/(time.perf_counter()-start_time))

idx = 0
while True:
    # camera.captureImage()
    # imgL = camera.imageL_array
    # imgR = camera.imageR_array
    # idx += 1

    # with torch.no_grad():
    #     # trt_outputs = trt_model({'left': img2tensor(imgL), 'right': img2tensor(imgR)})
    #     inputs = torch.rand(1, 3, 720, 1280).cuda()

    #     trt_outputs = trt_model({'left': inputs, 'right': inputs})
    #     output = trt_outputs['output']
    #     # print(output)

    # inputs = torch.rand(1, 3, 720, 1280).cuda()
    with torch.no_grad():
        trt_outputs = trt_model({'left': inputs, 'right': inputs})
        output = trt_outputs['output']
        print(output.shape)
        print('='*100)


    idx += 1
    if idx>10: break

        # output = tensor2img(output)

    #    # cv2.imshow("view", np.vstack((camera.imageL_array, camera.imageR_array)))
    #     cv2.imshow("view", cv2.resize(imgL, (0,0), fx=0.5, fy=0.5))
    #     cv2.imshow("depth", cv2.resize(camera.depth_array/15,(0,0), fx=0.5, fy=0.5))
    #     cv2.imshow("output", cv2.resize(output/15,(0,0), fx=0.5, fy=0.5))
    #     # cv2.imshow("depth_cv", disparity)

    #     # cv2.imwrite(f'/workspace/')

    #     cv2.waitKey(1)