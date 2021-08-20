import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
import utils.logger as logger
import torch.backends.cudnn as cudnn


parser = argparse.ArgumentParser(description='Anynet fintune on KITTI')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--datapath', default=None, help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=6,
                    help='batch size for training (default: 6)')
parser.add_argument('--test_bsize', type=int, default=8,
                    help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='results/infer',
                    help='the path of saving checkpoints and log')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--with_spn', action='store_true', default=False)
parser.add_argument('--with_cspn', action='store_true', help='with cspn network or not')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
parser.add_argument('--cspn_init_channels', type=int, default=8, help='initial channels for cspnet')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--start_epoch_for_spn', type=int, default=121)
parser.add_argument('--pretrained', type=str, default='checkpoint/kitti2015_ck/checkpoint.tar',
                    help='pretrained model path')


args = parser.parse_args()


from PIL import Image
from dataloader.preprocess import get_transform
import numpy as np

import pyzed.sl as sl
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms.functional import hflip


from mmcv.onnx.symbolic import register_extra_symbolics
from mmcv.tensorrt import TRTWrapper, is_tensorrt_plugin_loaded

register_extra_symbolics(11)
assert is_tensorrt_plugin_loaded(), 'Requires to complie TensorRT plugins in mmcv'

class Trt_model:
    def __init__(self, trt_file):
        self.model = TRTWrapper(trt_file, ['left', 'right'], ['output'])
     
    def inference(self, imgL, imgR):
        imgL = imgL.cuda()
        imgR = imgR.cuda()

        input_L = imgL
        input_R = imgR

        output = self.model({'left': input_L, 'right': input_R})
        disp = output['output']
        
        disp = torch.squeeze(disp)
        pred_disp = disp.data.cpu().numpy()

        return pred_disp


def test(model, imgL, imgR, left_right_left=False):

    imgL = imgL.cuda()
    imgR = imgR.cuda()

    if left_right_left:
        input_L = torch.cat((imgL, hflip(imgR)), dim=0)
        input_R = torch.cat((imgR, hflip(imgL)), dim=0)
    else:
        input_L = imgL
        input_R = imgR

    model.eval()
    with torch.no_grad():
        disp = model(input_L,input_R)

        if left_right_left:
            disp_L, disp_R = disp[0], disp[1]
            disp = torch.cat((disp_L, hflip(disp_R)), dim=0)
        
        disp = torch.squeeze(disp)
        pred_disp = disp.data.cpu().numpy()

    return pred_disp

class trackbar:
    def __init__(self, trackbarName, windowName, minValue, maxValue, defaultValue, handler):
        cv2.createTrackbar(trackbarName, windowName, minValue, maxValue, handler)
        cv2.setTrackbarPos(trackbarName, windowName, defaultValue)

class WLSFilter():
    wlsStream = "wlsFilter"

    def on_trackbar_change_lambda(self, value):
        self._lambda = value * 100
    
    def on_trackbar_change_sigma(self, value):
        self._sigma = value / float(10)

    def __init__(self, _lambda, _sigma):
        self._lambda = _lambda
        self._sigma = _sigma
        self.wlsFilter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
        cv2.namedWindow(self.wlsStream)
        self.lambdaTrackbar = trackbar('Lambda', self.wlsStream, 0, 255, 80, self.on_trackbar_change_lambda)
        self.sigmaTrackbar  = trackbar('Sigma',  self.wlsStream, 0, 100, 15, self.on_trackbar_change_sigma)

    def filter(self, disparity, right):
        self.wlsFilter.setLambda(self._lambda)
        self.wlsFilter.setSigmaColor(self._sigma)
        filteredDisp = self.wlsFilter.filter(disparity, right)
        return filteredDisp

def main():
    global args
    log = logger.setup_logger(args.save_path + '/infer.log')


    # model = models.anynet.AnyNet(args)
    # model = model.cuda()
    # model = nn.DataParallel(model).cuda()


    model_trt = Trt_model('sample.trt')

    # log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # if args.pretrained:
    #     if os.path.isfile(args.pretrained):
    #         checkpoint = torch.load(args.pretrained)
    #         model.load_state_dict(checkpoint['state_dict'], strict=False)
    #         log.info("=> loaded pretrained model '{}'"
    #                  .format(args.pretrained))
    #     else:
    #         log.info("=> no pretrained model found at '{}'".format(args.pretrained))
    #         log.info("=> Will start from scratch.")

    log.info(f"=> {args.with_spn}, with spn network")
    log.info(f"=> {args.with_cspn}, with cspn network")

    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))
    
    cudnn.benchmark = False
    torch.backends.cudnn.enabled = False 


    import torchvision.transforms as transforms
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(**normal_mean_var),
                                        # transforms.Resize((32*9, 32*16)),
                                        ])  
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    # init.depth_mode = sl.DEPTH_MODE.QUALITY

    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(1)

    runtime_params = sl.RuntimeParameters()
    # runtime_params.sensing_mode = sl.SENSING_MODE.FILL
    mat = sl.Mat()


    # Create DisparityWLSFilter
    wsize=31
    max_disp = 128
    left_matcher = cv2.StereoBM_create(max_disp, wsize)

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter = WLSFilter(_lambda=8000, _sigma=1.5)
    # right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # disp_filter = cv2.ximgproc.createDisparityFilter(left_matcher)
    # disp_filter.setLambda(lmbda)
    # disp_filter.setSigmaColor(sigma)

    key = ''
    print("  Quit : CTRL+C\n")
    while key != ord('q'):
        err = cam.grab(runtime_params)
        if (err == sl.ERROR_CODE.SUCCESS) :
            cam.retrieve_image(mat, sl.VIEW.LEFT)
            imgL = mat.get_data()[..., :3]
            cam.retrieve_image(mat, sl.VIEW.RIGHT)
            imgR = mat.get_data()[..., :3]
            
            cv2.imshow("ZED", cv2.resize(np.vstack((imgL,imgR)), (0,0), fx=0.5, fy=0.5))


            # cam.retrieve_image(mat, sl.VIEW.RIGHT)
            # cam.retrieve_measure(mat, sl.MEASURE.DEPTH)
            # imgD = mat.get_data()
            # cv2.imshow("ZED_d", cv2.resize(imgD/10000.0, (0,0), fx=0.5, fy=0.5))
            imgL_8U = imgL.copy()
            imgR_8U = imgR.copy()


            imgR = Image.fromarray(imgR).convert('RGB')
            imgL = Image.fromarray(imgL).convert('RGB')

            imgL = infer_transform(imgL)
            imgR = infer_transform(imgR)
            
            # pad to width and hight to 16 times
            if imgL.shape[1] % 16 != 0:
                times = imgL.shape[1]//16       
                top_pad = (times+1)*16 -imgL.shape[1]
            else:
                top_pad = 0

            if imgL.shape[2] % 16 != 0:
                times = imgL.shape[2]//16                       
                right_pad = (times+1)*16-imgL.shape[2]
            else:
                right_pad = 0    

            imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
            imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

            start_time = time.perf_counter()
            # pred_disp = test(model, imgL, imgR)

            pred_disp = model_trt.inference(imgL, imgR)
            print(f'Speed = {1/(time.perf_counter() - start_time):3.0f} FPS')

            if top_pad !=0 and right_pad != 0:
                img = pred_disp[top_pad:,:-right_pad]
            elif top_pad ==0 and right_pad != 0:
                img = pred_disp[:,:-right_pad]
            elif top_pad !=0 and right_pad == 0:
                img = pred_disp[top_pad:,:]
            else:
                img = pred_disp

            disp = img


            # cv2.imshow("disp", cv2.resize(np.vstack((img[0],img[1]))/192, (0,0), fx=0.5, fy=0.5))
            # disp = np.vstack((img[0],img[1]))
            # img = np.vstack((imgL_8U, imgR_8U))
            # filtered_disp = wls_filter.filter(disp, img)
            # cv2.imshow(WLSFilter.wlsStream, cv2.resize(530*0.12/filtered_disp/10, (0,0), fx=0.5, fy=0.5))

            cv2.imshow("disp", cv2.resize(img/192, (0,0), fx=0.5, fy=0.5))
            filtered_disp = wls_filter.filter(disp, imgL_8U)
            cv2.imshow(WLSFilter.wlsStream, cv2.resize(530*0.12/filtered_disp/10, (0,0), fx=0.5, fy=0.5))
            # WSLfiltered_disp = wls_filter.filter(disp, imgL_8U, None, -disp)


            # from disp2color import disp_map
            # depth = np.clip(500*0.12/img, 0.1, 25)
            # cv2.imshow('disp_anynet', cv2.resize(img/100, (0,0), fx=0.5, fy=0.5))
            # cv2.imshow('depth', cv2.resize(depth/20, (0,0), fx=0.5, fy=0.5))


            key = cv2.waitKey(1)
        else :
            key = cv2.waitKey(1)

    plt.show()

    cam.close()

if __name__ == "__main__":
    main()
