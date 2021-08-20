import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import utils.logger as logger
import torch.backends.cudnn as cudnn

# https://forums.developer.nvidia.com/t/onnx-to-tensorrt-conversion-fails/180953/5
# https://github.com/NVIDIA/TensorRT/issues/805
# --> https://mmcv.readthedocs.io/en/latest/deployment/tensorrt_custom_ops.html#scatternd
# https://github.com/onnx/onnx-tensorrt/issues/378
# --> F.pad not surpportted. https://github.com/onnx/onnx-tensorrt/issues/378
 
import models.anynet

parser = argparse.ArgumentParser(description='Anynet fintune on KITTI')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
# parser.add_argument('--datatype', default='2015', help='datapath')
# parser.add_argument('--datapath', default=None, help='datapath')
parser.add_argument('--test_bsize', type=int, default=8,
                    help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='results/finetune_anynet',
                    help='the path of saving checkpoints and log')
parser.add_argument('--with_cspn', action='store_true', help='with cspn network or not')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
parser.add_argument('--cspn_init_channels', type=int, default=8, help='initial channels for cspnet')
# parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--start_epoch_for_spn', type=int, default=121)
parser.add_argument('--pretrained', type=str, default='results/pretrained_anynet/checkpoint.tar',
                    help='pretrained model path')
parser.add_argument('--split_file', type=str, default=None)
parser.add_argument('--evaluate', action='store_true')


args = parser.parse_args()
args.with_spn = False

import numpy as np
import common
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# import torch.onnx.symbolic_opset11 as sym_opset
# import torch.onnx.symbolic_helper as sym_help


# def grid_sampler(g, input, grid, mode, padding_mode, align_corners): #long, long, long: contants dtype
#     mode_i = sym_help._maybe_get_scalar(mode)
#     paddingmode_i = sym_help._maybe_get_scalar(padding_mode)
#     aligncorners_i = sym_help._maybe_get_scalar(align_corners)

#     return g.op("GridSampler", input, grid, interpolationmode_i=mode_i, paddingmode_i=paddingmode_i,
#      aligncorners_i=aligncorners_i) #just a dummy definition for onnx runtime since we don't need onnx inference

# sym_opset.grid_sampler = grid_sampler

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def export_trt_model():
    global args
    log = logger.setup_logger(args.save_path + '/convert.log')

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    model = models.anynet.AnyNet(args)
    model.to('cuda')
    model.eval()

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            log.info("=> loaded pretrained model '{}'"
                     .format(args.pretrained))
        else:
            log.info("=> no pretrained model found at '{}'".format(args.pretrained))
            log.info("=> Will start from scratch.")

    

    from torch2trt import torch2trt
    x_l = torch.ones((1, 3, 368, 1232)).cuda()
    x_r = torch.ones((1, 3, 368, 1232)).cuda()

    print(model(x_l, x_r))

    try:
        model_trt = torch2trt(model, 
                              [x_l, x_r],
                              use_onnx=True)
                            #   use_onnx=False)
    except:
        print('Error','='*100)
    pass

def export_onnx_model(onnx_model_file):
    global args
    log = logger.setup_logger(args.save_path + '/convert.log')

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    model = models.anynet.AnyNet(args)
    model.to('cuda')
    model.eval()

    print(model)

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            log.info("=> loaded pretrained model '{}'"
                     .format(args.pretrained))
        else:
            log.info("=> no pretrained model found at '{}'".format(args.pretrained))
            log.info("=> Will start from scratch.")

    cudnn.benchmark = True

    x_l = torch.ones((1, 3, 368, 1232)).cuda()
    x_r = torch.ones((1, 3, 368, 1232)).cuda()

    x_l = torch.ones((1, 3, 320, 640)).cuda()
    x_r = torch.ones((1, 3, 320, 640)).cuda()

    print(model(x_l, x_r))

    torch.onnx.export(
        model,
        (x_l, x_r),
        onnx_model_file,
        export_params=True,
        # verbose=True, 
        do_constant_folding=True,
        input_names=['left', 'right'],
        output_names=['output'],
        opset_version=11,
        dynamic_axes={"left" : {2:'height', 3:'width'},
                      "right" : {2:'height', 3:'width'},
                      'output':{2:'height', 3:'width'}
                      },
        enable_onnx_checker=False
    )

    # x_lr = torch.ones((1, 6, 368, 1232)).cuda()
    # print(model(x_lr))

    # torch.onnx.export(
    #     model,
    #     (x_lr),
    #     onnx_model_file,
    #     export_params=True,
    #     verbose=True, 
    #     input_names=['left_right'],
    #     output_names=['output'],
    #     opset_version=11,
    #     dynamic_axes={"left_right" : {0: "batch_size"}},
    #     enable_onnx_checker=True
    # )

    print(f'===================\n Onnx File Saved: {onnx_model_file}')

def export_onnx_model_by_mmcv(onnx_model_file):

    global args
    log = logger.setup_logger(args.save_path + '/convert.log')

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    model = models.anynet.AnyNet(args)
    model.to('cuda')
    model.eval()

    print(model)

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            log.info("=> loaded pretrained model '{}'"
                     .format(args.pretrained))
        else:
            log.info("=> no pretrained model found at '{}'".format(args.pretrained))
            log.info("=> Will start from scratch.")

    cudnn.benchmark = True


    x_l = torch.ones((1, 3, 720, 1280)).cuda()
    x_r = torch.ones((1, 3, 720, 1280)).cuda()
    # x_l = torch.ones((1, 3, 576, 576)).cuda()
    # x_r = torch.ones((1, 3, 576, 576)).cuda()

    print(model(x_l, x_r))

    from mmcv.onnx.symbolic import register_extra_symbolics
    register_extra_symbolics(11)

    torch.onnx.export(
        model,
        (x_l, x_r),
        onnx_model_file,
        export_params=True,
        # verbose=True, 
        do_constant_folding=True,
        input_names=['left', 'right'],
        output_names=['output'],
        opset_version=11,
        # dynamic_axes={"left" : {2:'height', 3:'width'},
        #               "right" : {2:'height', 3:'width'},
        #               'output':{2:'height', 3:'width'}
        #               },
        enable_onnx_checker=True
    )
    print(f'===================\n Onnx File Saved: {onnx_model_file}')


def simplify_onnx(onnx_model_file):
    import onnx
    from onnxsim import simplify

    model = onnx.load(onnx_model_file)

    # convert model
    model_simp, check = simplify(model, 
                                 perform_optimization=False,
                                 skip_shape_inference=True,
                                 dynamic_input_shape=True,
                                #  input_shapes={'left': [1, 3, 368, 1232], 'right': [1, 3, 368, 1232]}
                                 input_shapes={'left': [1, 3, 320, 640], 'right': [1, 3, 320, 640]}
                                 )

    assert check, "Simplified ONNX model could not be validated"

    onnx.save(model_simp, onnx_model_file)

def modify_onnx(onnx_model_file):
    import onnx_graphsurgeon as gs
    import onnx

    graph = gs.import_onnx(onnx.load(onnx_model_file))
    assert(graph is not None)

    for node in graph.nodes:
        if node.op == 'GridSampler':
            _, c, h, w = node.inputs[0].shape
            _, h_g, w_g, _ = node.inputs[1].shape
            align_corners = node.attrs['aligncorners']
            inter_mode = node.attrs['interpolationmode']
            pad_mode = node.attrs['paddingmode']
            m_type = 0 if node.inputs[0].dtype == np.float32 else 1
            buffer = np.array([c, h, w, h_g, w_g], dtype=np.int64).tobytes('C') \
              + np.array([inter_mode, pad_mode], dtype=np.int32).tobytes('C') \
              + np.array([align_corners], dtype=np.bool).tobytes('C') \
              + np.array([m_type], dtype=np.int32).tobytes('C')
            node.attrs = {'name':'GridSampler', 'version':'1', 'namespace':"", 'data':buffer}
            node.op = 'TRT_PluginV2'
    
    onnx.save(gs.export_onnx(graph), onnx_model_file)

def modify_onnx2(onnx_model_file):
    import onnx_graphsurgeon as gs
    import onnx

    graph = gs.import_onnx(onnx.load(onnx_model_file))
    assert(graph is not None)

    node = [node for node in graph.nodes if node.op == "Resize"]
    for n in node:
        n.attrs['mode'] = 'bilinear'

    onnx.save(gs.export_onnx(graph), onnx_model_file)

def modify_onnx3(onnx_model_file):
    import onnx_graphsurgeon as gs
    import onnx

    graph = gs.import_onnx(onnx.load(onnx_model_file))
    assert(graph is not None)

    for node in graph.nodes:
        if node.op == 'Neg':
            pass

    # for node in graph.node:
    #     if node.op_type == 'Expand':
    #         for initializer in model.graph.initializer:
    #             if initializer.name == node.input[0]:
    #                 for n in model.graph.node:
    #                     if node.input[1] in n.output:
    #                         if n.op_type == 'Constant':
    #                             b, x, y, z = unpack("qqqq", n.attribute[0].t.raw_data)
    #                             if b == -1:
    #                                 b = initializer.dims[0]
    #                             if x == -1:
    #                                 x = initializer.dims[1]
    #                             if y == -1:
    #                                 y = initializer.dims[2]
    #                             if z == -1:
    #                                 z = initializer.dims[3]

    #                             n.attribute[0].t.raw_data = pack("qqqq", b, x, y, z)
    #                         break
    #                 break

    onnx.save_model(model, "model_opt.onnx")

# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
    with trt.Builder(TRT_LOGGER) as builder,\
        builder.create_network(common.EXPLICIT_BATCH) as network,\
        trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        builder.max_workspace_size = common.GiB(5)
        builder.fp16_mode = True
        # builder.max_batch_size = 1 # always 1 for explicit batch
        config = builder.create_builder_config()
        # # need to be set along with fp16_mode if config is specified.        
        # config.set_flag(trt.BuilderFlag.FP16)
        # profile = builder.create_optimization_profile()
        # profile.set_shape('left', (1, 3, 576, 1024), (1, 3, 576, 1024), (1, 3, 576, 1024))
        # profile.set_shape('right', (1, 3, 576, 1024), (1, 3, 576, 1024), (1, 3, 576, 1024))
        # config.add_optimization_profile(profile)

        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            print(f'Converting Onnx File: {model_file}... ...')
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        return builder.build_engine(network, config)

if __name__ == '__main__':

    onnx_model_file = 'anynet_cspn.onnx'
    # export_onnx_model(onnx_model_file)
    export_onnx_model_by_mmcv(onnx_model_file)
    # simplify_onnx(onnx_model_file)
    # modify_onnx2(onnx_model_file)
    # build_engine_onnx(onnx_model_file)

    # export_trt_model()




