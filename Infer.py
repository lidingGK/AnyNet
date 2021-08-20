import numpy as np
import tensorrt as trt
import cv2
import os
import pycuda.autoinit
import pycuda.driver as cuda

try:
    from . import TRT_exec_tools
except ImportError:
    import TRT_exec_tools
    

class Semantic_Segmentation:

    def __init__(self, trt_engine_path):
        """
        Parameters:
        -----------
        trt_engine_path: string
            Path to TRT engine.
        """
        # Create a Context on this device,
        self.cfx = cuda.Device(0).make_context()

        TRT_LOGGER = trt.Logger()
        TRT_LOGGER.min_severity = trt.Logger.Severity.VERBOSE

        # Load TRT Engine
        with open(trt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # Create Context
        self.context = self.engine.create_execution_context()

        # Allocate buffers required for engine
        self.inputs, self.outputs, self.bindings, self.stream = TRT_exec_tools.allocate_buffers(self.engine)

        # Input image height
        self.height = 720

        # Input image width
        self.width = 1280

        # RGBA Colour map for segmentation display
        self.colour_map = None


    def segment_image(self, image):
        """
        Parameters:
        -----------
        image: np.array
            HWC uint8 BGR
        """
        assert image.shape == (self.height, self.width, 3)

        # Infer
        self.inputs[0].host = np.ascontiguousarray(image.astype('float32')).ravel()

        # Make self the active context, pushing it on top of the context stack.
        self.cfx.push()

        trt_outputs = TRT_exec_tools.do_inference_v2(
                                                        context=self.context,
                                                        bindings=self.bindings,
                                                        inputs=self.inputs,
                                                        outputs=self.outputs,
                                                        stream=self.stream,
        )

        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()

        o = trt_outputs[0].reshape(self.height, self.width)

        # HW np.array uint8
        self.depth = o


if __name__ == "__main__":

    import time

    model = Semantic_Segmentation("sample.trt")

    N = 500
    images = np.random.randint(0, 255, size=[N, model.height, model.width, 3], dtype='uint8')

    t1 = time.perf_counter()

    for i in images:
        model.segment_image(i)

    t2 = time.perf_counter()

    print(N/(t2-t1))
    model.cfx.pop()
