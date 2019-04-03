import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import random
import time
import argparse
import json
import glob
from draw.draw_keypoint import draw_keypoint
cfg = None 
Network = None
Preprocessing = None
Tester = None
mem_info = None
colorlogger = None
gpu_nms = None
cpu_soft_nms = None
COCOJoints = None
COCO = None
COCOeval = None
COCOmask = None

SHOW_BIMG = False

temp_glob = 0
def PreprocessingSimple(img, bbox_, out_cpn_input=None):
    height, width = cfg.data_shape
    ret_ims = []
    # img = cv2.imread(os.path.join(image_path))
#    cv2.imshow("hoge", img)
#    cv2.waitKey(0)
    
    add = max(img.shape[0], img.shape[1])

    bimg = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT,
                          value=cfg.pixel_means.reshape(-1))

    if SHOW_BIMG:
        cv2.imshow("bimg", bimg)
        cv2.waitKey(10)

    bbox = np.array(bbox_).reshape(4, ).astype(np.float32)
    bbox[:2] += add

    extend_ratio = 0.05 # cfg.imgExtXBorder
    crop_width = bbox[2] * (1 + extend_ratio * 2)
    crop_height = bbox[3] * (1 + extend_ratio * 2)
    objcenter = np.array([bbox[0] + bbox[2] / 2., bbox[1] + bbox[3] / 2.])

    if crop_height / height > crop_width / width:
        crop_size = crop_height
        min_shape = height
    else:
        crop_size = crop_width
        min_shape = width
    crop_size = min(crop_size, objcenter[0] / width * min_shape * 2. - 1.)
    crop_size = min(crop_size, (bimg.shape[1] - objcenter[0]) / width * min_shape * 2. - 1)
    crop_size = min(crop_size, objcenter[1] / height * min_shape * 2. - 1.)
    crop_size = min(crop_size, (bimg.shape[0] - objcenter[1]) / height * min_shape * 2. - 1)

    min_x = int(objcenter[0] - crop_size / 2. / min_shape * width)
    max_x = int(objcenter[0] + crop_size / 2. / min_shape * width)
    min_y = int(objcenter[1] - crop_size / 2. / min_shape * height)
    max_y = int(objcenter[1] + crop_size / 2. / min_shape * height)

    x_ratio = float(width) / (max_x - min_x)
    y_ratio = float(height) / (max_y - min_y)

    img = cv2.resize(bimg[min_y:max_y, min_x:max_x, :], (width, height))
    details = np.asarray([min_x - add, min_y - add, max_x - add, max_y - add])

    '''
    global temp_glob
    if temp_glob > -1:
        crop_ = img.copy()
        cv2.imshow("hoge", img)
        cv2.waitKey(10)
        if out_cpn_input is not None:
            out_cpn_input.write(img)
    temp_glob += 1
    '''
    crop_ = img.copy()

    img = img - cfg.pixel_means
    if cfg.pixel_norm:
        img = img / 255.
    img = img.transpose(2, 0, 1)
    ret_ims.append(img)

    return [np.asarray(ret_ims).astype(np.float32), details, crop_]



def load_json(json_path):
    bbox = None
    score = None
    det = json.load(open(json_path, "r"))

    if 'bbox' in det:
        bbox = np.asarray(det['bbox'])
        score = det['score']

    return {"bbox": bbox, "score": score}


def crop_img(bboxes, imgs):
    crop_data = [PreprocessingSimple(imgs[i], bboxes[i]["bbox"]) for i in range(len(imgs))] 
    return [list(x) for x in zip(*crop_data)]


def make_feed(inputs, flip):
    batch_size = len(inputs)
    test_imgs = inputs
    feed = test_imgs

    if flip:
        for i in range(batch_size):
            ori_img = test_imgs[i][0].transpose(1, 2, 0)
            flip_img = cv2.flip(ori_img, 1)
            feed.append(flip_img.copy().transpose(2, 0, 1)[np.newaxis, ...])

    feed = np.vstack(feed)
    return feed

def run_cpn(tester, inputs, flip):
    feed = make_feed(inputs, flip)
    '''
    batch_size = len(inputs)
    test_imgs = inputs
    feed = test_imgs

    if flip:
        for i in range(batch_size):
            ori_img = test_imgs[i][0].transpose(1, 2, 0)
            flip_img = cv2.flip(ori_img, 1)
            feed.append(flip_img.copy().transpose(2, 0, 1)[np.newaxis, ...])

    feed = np.vstack(feed)
    '''
    res = tester.predict_one([feed.transpose(0, 2, 3, 1).astype(np.float32)])[0]

    return res

    '''
    batch_size = len(inputs)
    res = res.transpose(0, 3, 1, 2)

    if flip:
        for i in range(batch_size):
            fmp = res[batch_size + i].transpose((1, 2, 0))
            fmp = cv2.flip(fmp, 1)
            fmp = list(fmp.transpose((2, 0, 1)))
            for (q, w) in cfg.symmetry:
                fmp[q], fmp[w] = fmp[w], fmp[q]
            fmp = np.array(fmp)
            res[i] += fmp
            res[i] /= 2

    return res[:batch_size]
    '''


def calculate_pose_from_cpn_result(cpn_out, detail, flip, batch_size):
    cpn_out = cpn_out.transpose(0, 3, 1, 2)

    if flip:
        for i in range(batch_size):
            fmp = cpn_out[batch_size + i].transpose((1, 2, 0))
            fmp = cv2.flip(fmp, 1)
            fmp = list(fmp.transpose((2, 0, 1)))
            for (q, w) in cfg.symmetry:
                fmp[q], fmp[w] = fmp[w], fmp[q]
            fmp = np.array(fmp)
            cpn_out[i] += fmp
            cpn_out[i] /= 2

    cpn_out = cpn_out[:batch_size]

    cls_skeleton = np.zeros((len(cpn_out), cfg.nr_skeleton, 3))
    crops = np.zeros((len(cpn_out), 4))

    for i in range(len(cpn_out)):
        r0 = cpn_out[i].copy()
        r0 /= 255.
        r0 += 0.5
        for w in range(cfg.nr_skeleton):
            cpn_out[i, w] /= np.amax(cpn_out[i, w])
        border = 10
        dr = np.zeros((cfg.nr_skeleton, cfg.output_shape[0] + 2 * border, cfg.output_shape[1] + 2 * border))
        dr[:, border:-border, border:-border] = cpn_out[i][:cfg.nr_skeleton].copy()
        for w in range(cfg.nr_skeleton):
            dr[w] = cv2.GaussianBlur(dr[w], (21, 21), 0)
        for w in range(cfg.nr_skeleton):
            lb = dr[w].argmax()
            y, x = np.unravel_index(lb, dr[w].shape)
            dr[w, y, x] = 0
            lb = dr[w].argmax()
            py, px = np.unravel_index(lb, dr[w].shape)
            y -= border
            x -= border
            py -= border + y
            px -= border + x
            ln = (px ** 2 + py ** 2) ** 0.5
            delta = 0.25
            if ln > 1e-3:
                x += delta * px / ln
                y += delta * py / ln
            x = max(0, min(x, cfg.output_shape[1] - 1))
            y = max(0, min(y, cfg.output_shape[0] - 1))
            cls_skeleton[i, w, :2] = (x * 4 + 2, y * 4 + 2)
            cls_skeleton[i, w, 2] = r0[w, int(round(y) + 1e-10), int(round(x) + 1e-10)]

        # map back to original images
        crops[i, :] = detail[i]
        for w in range(cfg.nr_skeleton):
            cls_skeleton[i, w, 0] = cls_skeleton[i, w, 0] / cfg.data_shape[1] * (
            crops[i][2] - crops[i][0]) + crops[i][0]
            cls_skeleton[i, w, 1] = cls_skeleton[i, w, 1] / cfg.data_shape[0] * (
            crops[i][3] - crops[i][1]) + crops[i][1]

    cls_partsco = cls_skeleton[:, :, 2].copy().reshape(-1, cfg.nr_skeleton)
    # cls_skeleton[:, :, 2] = 1
    # cls_scores = cls_dets[:, -1].copy()

    # rescore
    cls_skeleton = cls_skeleton.reshape(-1, cfg.nr_skeleton * 3)

    return cls_skeleton


def show_result(cls_skeleton, ori_image, input_image):
    pose = cls_skeleton[-1]
    drawn_image = ori_image.copy()
    draw_keypoint(drawn_image, pose)
    # cv2.namedWindow("key_img", cv2.WINDOW_NORMAL)
    # cv2.imshow("key_img", drawn_image)
    # cv2.imshow("input_image", input_image)

    # cv2.waitKey(10)
    return drawn_image

def set_import(resnet_type, input_size, batch_size, flip):
    global cfg, Network, Preprocessing, Tester, mem_info, colorlogger, gpu_nms, cpu_soft_nms, COCOJoints, COCO, COCOeval, COCOmask 
    if resnet_type == 50:
        if input_size == "384x288":
            pass
        elif input_size == "256x192":
            from models.res50_256x192.config import cfg
            from models.res50_256x192.network import Network
            from models.res50_256x192.dataset import Preprocessing      
    elif resnet_type == 101:
        if input_size == "384x288":
            from models.t_res101_384x288.config import cfg
            from models.t_res101_384x288.network import Network
            from models.t_res101_384x288.dataset import Preprocessing         
        elif input_size == "256x192":
            pass

    if flip:
        cfg.batch_size = batch_size * 2
    else:
        cfg.batch_size = batch_size

    from tfflat.base import Tester
    from tfflat.utils import mem_info
    from tfflat.logger import colorlogger
    from lib_kernel.lib_nms.gpu_nms import gpu_nms
    from lib_kernel.lib_nms.cpu_nms import cpu_soft_nms
    from COCOAllJoints import COCOJoints
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as COCOmask


def main(args):
    set_import(args.resnet_type, args.input_size, args.batch_size, args.flip)
    tester = Tester(Network(), cfg)
    tester.load_weights(args.model_path)

    video_cap = cv2.VideoCapture(args.video_path)
    bbox_data = [load_json(json_path) for json_path in sorted(glob.glob(os.path.join(args.json_folder, "*.json")))]

    count = 0
    
    while True:
        count_frame = 0
        frames = []

        while len(frames) < args.batch_size:
            ret, frame = video_cap.read()
            if not ret:
                break
            else:
                frames.append(frame)

        if len(frames) == 0:
            break

        input_, detail_, crop_ = crop_img(bbox_data[count: count + len(frames)], frames)
        cpn_out = run_cpn(tester, input_, args.flip)
        cls_skeleton = calculate_pose_from_cpn_result(cpn_out, detail_, args.flip, len(frames))
        show_result(cls_skeleton, frames[-1], crop_[-1])
        count += args.batch_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pose detection with tf-cpn')
    parser.add_argument('--resnet_type', type=int, default=101, choices=[50,101],
                        help='')
    parser.add_argument('--input_size', type=str, default="384x288", choices=["384x288", "256x192"],
                    help='')
    parser.add_argument('--model_path', type=str,
                    help='')
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--json_folder', type=str)
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--flip', action='store_true')
    args = parser.parse_args()
    main(args)





