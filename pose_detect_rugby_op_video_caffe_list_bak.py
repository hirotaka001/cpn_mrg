

import os
import glob
from pathlib import Path
import protocol_data
import zmq
import json
import argparse
import time
import sys
import queue
from multiprocessing import Process, Manager, Lock
import tf_cpn.tf_cpn as cpn
import cv2
import numpy as np
# import bbox_detect
import ctypes
import tensorrt as trt
import zmq_data
import trt_modules.uff_ssd.utils as utils
#import time
import math
import caffe

ssd_inference_utils = utils.inference
ssd_model_utils = utils.model
ssd_boxes_utils = utils.boxes
ssd_coco_utils = utils.coco
ssd_PATHS = utils.paths.PATHS

COCO_LABELS = ssd_coco_utils.COCO_CLASSES_LIST

# image_data_lock = Lock()
# pose_data_lock = Lock()
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
VISUALIZATION_THRESHOLD = 0.1
TRT_PRECISION_TO_DATATYPE = {
    16: trt.DataType.HALF,
    32: trt.DataType.FLOAT
}

COCO_PERSON_LABEL = 1
COCO_BALL_LABEL = 37
COCO_RUGBYBALL_LABEL = 34

TRT_PREDICTION_LAYOUT = {
    "image_id": 0,
    "label": 1,
    "confidence": 2,
    "xmin": 3,
    "ymin": 4,
    "xmax": 5,
    "ymax": 6
}

# setting for hand detection
protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]

threshold = 0.2

def fetch_prediction_field(field_name, detection_out, pred_start_idx):
    return detection_out[pred_start_idx + TRT_PREDICTION_LAYOUT[field_name]]


def select_bbox(bboxes_with_score, img_height, img_width):#, min_ratio=1/3):
    ret_bbox = None
    ret_score = None
    # max_conf = -1
    max_width_height = -1

    if len(bboxes_with_score) == 1:
        ret_score = bboxes_with_score[0][4]
        ret_bbox = bboxes_with_score[0][:4]
    elif len(bboxes_with_score) > 0:
        for bbox in bboxes_with_score:
            if max_width_height < bbox[2] + bbox[3]: #if max_conf < bbox[4] and (bbox[2] + bbox[3]) > (img_height + img_width) * min_ratio:
                max_width_height = bbox[2] + bbox[3]
                ret_score = bbox[4]
                ret_bbox = bbox[:4]

    return ret_bbox, ret_score


def analyze_prediction(detection_out, pred_start_idx, img_cv):
    image_id = int(fetch_prediction_field("image_id", detection_out, pred_start_idx))
    label = int(fetch_prediction_field("label", detection_out, pred_start_idx))
    confidence = fetch_prediction_field("confidence", detection_out, pred_start_idx)
    xmin = fetch_prediction_field("xmin", detection_out, pred_start_idx)
    ymin = fetch_prediction_field("ymin", detection_out, pred_start_idx)
    xmax = fetch_prediction_field("xmax", detection_out, pred_start_idx)
    ymax = fetch_prediction_field("ymax", detection_out, pred_start_idx)
    bbox = None

    if confidence > VISUALIZATION_THRESHOLD and label == COCO_PERSON_LABEL:
        class_name = COCO_LABELS[label]
        confidence_percentage = "{0:.0%}".format(confidence)
        print("Detected {} with confidence {}".format(
            class_name, confidence_percentage))
        ssd_boxes_utils.draw_bounding_boxes_on_image(
            img_cv, np.array([[ymin, xmin, ymax, xmax]]),
            display_str_list=["{}: {}".format(
                class_name, confidence_percentage)],
            color=ssd_coco_utils.COCO_COLORS[label],
            thickness=5,
            cv=True
        )

        x_min_im = int(xmin * img_cv.shape[1])
        y_min_im = int(ymin * img_cv.shape[0])
        width_im = int((xmax - xmin) * img_cv.shape[1])
        height_im = int((ymax - ymin) * img_cv.shape[0])
        bbox = (x_min_im, y_min_im, width_im, height_im, confidence)

    return bbox

def analyze_ball_prediction(detection_out, pred_start_idx, img_cv):
    image_id = int(fetch_prediction_field("image_id", detection_out, pred_start_idx))
    label = int(fetch_prediction_field("label", detection_out, pred_start_idx))
    confidence = fetch_prediction_field("confidence", detection_out, pred_start_idx)
    xmin = fetch_prediction_field("xmin", detection_out, pred_start_idx)
    ymin = fetch_prediction_field("ymin", detection_out, pred_start_idx)
    xmax = fetch_prediction_field("xmax", detection_out, pred_start_idx)
    ymax = fetch_prediction_field("ymax", detection_out, pred_start_idx)
    bbox = None

    if confidence > VISUALIZATION_THRESHOLD and (label == COCO_BALL_LABEL or label == COCO_RUGBYBALL_LABEL):
        class_name = COCO_LABELS[label]
        confidence_percentage = "{0:.0%}".format(confidence)
        print("Detected {} with confidence {}".format(
            class_name, confidence_percentage))
        ssd_boxes_utils.draw_bounding_boxes_on_image(
            img_cv, np.array([[ymin, xmin, ymax, xmax]]),
            display_str_list=["{}: {}".format(
                class_name, confidence_percentage)],
            color=ssd_coco_utils.COCO_COLORS[label],
            thickness=2,
            cv=True
        )

        x_min_im = int(xmin * img_cv.shape[1])
        y_min_im = int(ymin * img_cv.shape[0])
        width_im = int((xmax - xmin) * img_cv.shape[1])
        height_im = int((ymax - ymin) * img_cv.shape[0])
        bbox = (x_min_im, y_min_im, width_im, height_im, confidence)
        # print('ball: ', x_min_im, y_min_im, width_im, height_im, confidence)

    return bbox

def handDetect(cls_skeleton, oriImg):
    # cls_skeleton [ 0 "nose", 1 "left_eye", 2 "right_eye", 3 "left_ear", 4 "right_ear", 5 "left_shoulder", 6 "right_shoulder", 7 "left_elbow", 8 "right_elbow", 9 "left_wrist", 10 "right_wrist", 11 "left_hip", 12 "right_hip", 13 "left_knee", 14 "right_knee", 15 "left_ankle", 16 "right_ankle" ]
    # right hand: wrist 10, elbow 8, shoulder 6
    # left hand: wrist 9, elbow 7, shoulder 5

    ratioWristElbow = 0.33
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]
    for ii in range(1):
        # if any of three not detected
        # person = cls_skeleton
        has_left = np.sum(cls_skeleton[9*3] < 0.01) == 0
        has_right = np.sum(cls_skeleton[10*3] < 0.01) == 0
        if not (has_left or has_right):
            continue
        hands = []
        candidate = cls_skeleton
        #left hand
        if has_left:
            left_shoulder_index = 5
            left_elbow_index = 7
            left_wrist_index = 9 #person[[5, 6, 7]]
            x1 = candidate[left_shoulder_index*3]
            y1 = candidate[left_shoulder_index*3+1]
            x2 = candidate[left_elbow_index*3]
            y2 = candidate[left_elbow_index*3+1]
            x3 = candidate[left_wrist_index*3]
            y3 = candidate[left_wrist_index*3+1]
            hands.append([x1, y1, x2, y2, x3, y3, True])
        # right hand
        if has_right:
            right_shoulder_index = 6
            right_elbow_index = 8
            right_wrist_index = 10 #person[[2, 3, 4]]
            x1 = candidate[right_shoulder_index*3]
            y1 = candidate[right_shoulder_index*3+1]
            x2 = candidate[right_elbow_index*3]
            y2 = candidate[right_elbow_index*3+1]
            x3 = candidate[right_wrist_index*3]
            y3 = candidate[right_wrist_index*3+1]
            hands.append([x1, y1, x2, y2, x3, y3, False])

        for x1, y1, x2, y2, x3, y3, is_left in hands:
            # pos_hand = pos_wrist + ratio * (pos_wrist - pos_elbox) = (1 + ratio) * pos_wrist - ratio * pos_elbox
            # handRectangle.x = posePtr[wrist*3] + ratioWristElbow * (posePtr[wrist*3] - posePtr[elbow*3]);
            # handRectangle.y = posePtr[wrist*3+1] + ratioWristElbow * (posePtr[wrist*3+1] - posePtr[elbow*3+1]);
            # const auto distanceWristElbow = getDistance(poseKeypoints, person, wrist, elbow);
            # const auto distanceElbowShoulder = getDistance(poseKeypoints, person, elbow, shoulder);
            # handRectangle.width = 1.5f * fastMax(distanceWristElbow, 0.9f * distanceElbowShoulder);
            x = x3 + ratioWristElbow * (x3 - x2)
            y = y3 + ratioWristElbow * (y3 - y2)
            distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            width = 1.2 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
            # x-y refers to the center --> offset to topLeft point
            # handRectangle.x -= handRectangle.width / 2.f;
            # handRectangle.y -= handRectangle.height / 2.f;
            x -= width / 2
            y -= width / 2  # width = height
            # overflow the image
            if x < 0: x = 0
            if y < 0: y = 0
            width1 = width
            width2 = width
            if x + width > image_width: width1 = image_width - x
            if y + width > image_height: width2 = image_height - y
            width = min(width1, width2)
            # the min hand box value is 20 pixels
            if width >= 20:
                detect_result.append([int(x), int(y), int(width), is_left])

    '''
    return value: [[x, y, w, True if left hand else False]].
    width=height since the network require squared input.
    x, y is the coordinate of top left 
    '''
    return detect_result

def detect_pose_loop(args):

    cpn_tester = make_cpn_tester(args)
    ssd_inference_wrapper = make_ssd_inferece_wrapper(args)

    bbox_save_filenames = None
    bbox_filenames = None
    pose_bbox_save_filenames = None

    if args.bbox_save_filenames is not None:
        with open(args.bbox_save_filenames) as f:
            bbox_save_filenames = f.read().splitlines()

    if args.bbox_filenames is not None:
        with open(args.bbox_filenames) as f:
            bbox_filenames = f.read().splitlines()

    if args.pose_bbox_save_filenames is not None:
        with open(args.pose_bbox_save_filenames) as f:
            pose_bbox_save_filenames = f.read().splitlines()

    if args.use_camera:
        cap = cv2.VideoCapture(0)
        #cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        #cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        width = cap.get(3)  # float
        height = cap.get(4) # float
        print(width, height, cap.get(cv2.CAP_PROP_EXPOSURE))
    elif Path(args.image_folder_or_list).is_file:
        with open(args.image_folder_or_list, "r") as image_list_f:
            images = image_list_f.read().splitlines() 
    else: 
        images = glob.glob(str(Path(args.image_folder_or_list) / "*"))

    # caffe.set_mode_cpu()
    caffe.set_device(0)
    caffe.set_mode_gpu()

    #load the model
    hand_net = caffe.Net(protoFile,
                    weightsFile,
                    caffe.TEST)

    # hand_net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    # vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (640,480))

    #for i, image_path in enumerate(images):
     #   np_img = cv2.imread(image_path) #cv2.imdecode(np.frombuffer(images_binary[0], dtype = 'int8'), 3)
      #  print('frame %d' % i)

    frameID = 0
    for i, image_path in enumerate(images):
        np_img = cv2.imread(image_path) #cv2.imdecode(np.frombuffer(images_binary[0], dtype = 'int8'), 3)
        # ret, np_img = cap.read()
        # np_img = cv2.warpAffine(np_img, rotation_matrix, orig_size, flags=cv2.INTER_CUBIC)
        # np_img = orig_img[y:y+h, x:x+w].copy()
        print('frame %d' % frameID)
        # print(np_img.shape)
        frameID = frameID+1
        start_t = time.time()

        if args.bbox_filenames is None:
            input_list = ssd_inference_wrapper._load_np_imgs([np_img])
            t1 = time.time()
            ssd_inference_wrapper.set_input(input_list)
            detection_out , keep_count_out = ssd_inference_wrapper.infer_batch_t(len(input_list))
            print('person detection time: ', time.time() - t1)

            prediction_fields = len(TRT_PREDICTION_LAYOUT)

            # detect persons using ssd
            bboxes_with_score = []
            ball_bboxes_with_score = []
            np_img_copy = np_img.copy()
            for det in range(int(keep_count_out[0])):
                bbox_ = analyze_prediction(detection_out, det * prediction_fields, np_img_copy)
                if bbox_ is not None:
                    bboxes_with_score.append(bbox_)
                # ball bbox
                ball_bbox_ = analyze_ball_prediction(detection_out, det * prediction_fields, np_img_copy)
                if ball_bbox_ is not None:
                    ball_bboxes_with_score.append(ball_bbox_)

            # cv2.imshow("bbox", np_img_copy)
            # cv2.waitKey(1)
            write_path = str(Path(args.pose_img_save_folder) )
            cv2.imwrite(write_path + '/bbox_%06d.jpg' % frameID, np_img_copy)

            selected_bbox, score = select_bbox(bboxes_with_score, np_img.shape[1], np_img.shape[0])#, 0)
            if args.bbox_save_filenames is not None:
                json_dict = {}

                if selected_bbox is None:
                    if len(bboxes_with_score) > 0:
                        json_dict = {"bbox":bboxes_with_score[0][0:4]}
                else:
                    json_dict = {"bbox":selected_bbox}

                json_obj = json.loads(json.dumps(json_dict))

                # write json 
                json_file = open(bbox_save_filenames[frameID], "w")
                json.dump(json_obj, json_file)
                json_file.close()
        else:
            with open(bbox_filenames[frameID], "r") as bbox_f:
                bbox_json = json.load(bbox_f)
                selected_bbox = bbox_json["bbox"]

        if selected_bbox is None:
            #cv2.imshow("bbox_none", np_img_copy)
            #cv2.waitKey(0)

            if args.pose_bbox_save_filenames is not None:
                pose_bbox_path = pose_bbox_save_filenames[frameID]
                pose_bbox_obj = json.loads("{}")
                with open(pose_bbox_path, "w") as pose_bbox_f:
                    json.dump(pose_bbox_obj, pose_bbox_f)

        else:
            cpn_input_bbox = {"bbox":list(selected_bbox), "score":score}
            frames = [np_img]
            bboxes = [cpn_input_bbox]
            t2 = time.time()
            input_, detail_, crop_ = cpn.crop_img(bboxes, frames)
            cpn_out = cpn.run_cpn(cpn_tester, input_, args.cpn_flip)
            cls_skeleton = cpn.calculate_pose_from_cpn_result(cpn_out, detail_, args.cpn_flip, len(frames))
            print('pose estimation time: ', time.time() - t2)
            pose_bbox_img = cpn.show_result(cls_skeleton, frames[0], crop_[0])

            if args.pose_bbox_save_filenames is not None:
                pose_bbox_path = pose_bbox_save_filenames[frameID]
                pose_bbox_dict = {"bbox": selected_bbox, "keypoints": list(cls_skeleton[0])}
                pose_bbox_obj = json.loads(json.dumps(pose_bbox_dict))
                with open(pose_bbox_path, "w") as pose_bbox_f:
                    json.dump(pose_bbox_obj, pose_bbox_f)
   
            if args.pose_img_save_folder is not None:
               write_path = str(Path(args.pose_img_save_folder) )
              #  cv2.imwrite(write_path + '/pose_%06d.jpg' % frameID, pose_bbox_img)

        #print('hand pose estimation')
        ttmp = time.time()        
        #image_orig = frames[0].copy() #scipy.misc.imread(image_path)
        image_orig = [np_img][0].copy() #scipy.misc.imread(image_path)
        frameCopy = np.copy(image_orig)

        hands_list = handDetect(cls_skeleton[0], image_orig)
        print('hand searching time: ', time.time() - ttmp)
        # startx = int(cls_skeleton[0][9*3])
        # starty = int(cls_skeleton[0][9*3+1])
        # print('#left_wrist: ', cls_skeleton[0][9*3], cls_skeleton[0][9*3+1])
        # startx = int(cls_skeleton[0][10*3])
        # starty = int(cls_skeleton[0][10*3+1])
        # print('#right_wrist: ', cls_skeleton[0][10*3], cls_skeleton[0][10*3+1])
        # cropw = 120
        # croph = 134
        inHeight = 368 # 240 # 
        inWidth = inHeight # int(((aspect_ratio*inHeight)*8)//8)
        hand_net.blobs['image'].reshape(2,3,inHeight,inWidth)
        handID = 0
        tic = time.time()
        for x, y, w, is_left in hands_list:
            image_raw = image_orig[y:y+w, x:x+w] #rugby1.jpg (1080, 333), 120x134

            frameWidth = image_raw.shape[1]
            frameHeight = image_raw.shape[0]
            # aspect_ratio = frameWidth*1.0/frameHeight

            hand_net.blobs['image'].data[handID, :, :, :] = cv2.dnn.blobFromImage(image_raw, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False) 
            #inpBlob = cv2.dnn.blobFromImage(image_raw, 1.0 / 255, (inWidth, inHeight),
             #                        (0, 0, 0), swapRB=False, crop=False)
            handID = handID + 1

            #hand_net.setInput(inpBlob)

        output = hand_net.forward()
        print("hand net forward = {}".format(time.time() - ttmp))

        handID = 0
        for x, y, w, is_left in hands_list:
            image_raw = image_orig[y:y+w, x:x+w] #rugby1.jpg (1080, 333), 120x134

            frameWidth = image_raw.shape[1]
            frameHeight = image_raw.shape[0]

            # Empty list to store the detected keypoints
            points = []
            points_orig = []

            for i in range(nPoints):
                # confidence map of corresponding body's part.
                # probMap = output[0, i, :, :]
                probMap = output['net_output'][handID][i]
                probMap = cv2.resize(probMap, (frameWidth, frameHeight))
 
                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                if prob > threshold :
                    cv2.circle(frameCopy, (int(point[0]), int(point[1])), 6, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, lineType=cv2.LINE_AA)

                    # Add the point to the list if the probability is greater than the threshold
                    points.append((int(point[0]), int(point[1])))
                    points_orig.append((int(point[0]+x), int(point[1]+y)))
                else :
                    points.append(None)
                    points_orig.append(None)

            # Draw Skeleton
            for pair in POSE_PAIRS:
                partA = pair[0]
                partB = pair[1]

                if points[partA] and points[partB]:
                    cv2.line(image_raw, points[partA], points[partB], (0, 255, 255), 2, lineType=cv2.LINE_AA)
                    cv2.line(pose_bbox_img, points_orig[partA], points_orig[partB], (0, 255, 255), 2, lineType=cv2.LINE_AA)
                    #cv2.circle(image_raw, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                    #cv2.circle(image_raw, points[partB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

            handID = handID + 1

            #cv2.imwrite(write_path + '/hand_%06d_%s.png' % (frameID, is_left), image_raw)
            #vid_writer.write(image_raw)

        end_t = time.time()
        cv2.imshow('Output-Skeleton', pose_bbox_img)
        key = cv2.waitKey(1)
        if key == 27:
            break
        cv2.imwrite(write_path + '/detect_%06d.png' % (frameID), pose_bbox_img)
        print('####### fps : ', 1./(end_t - start_t))

    #vid_writer.release()
    cap.release()
    cv2.destroyAllWindows()    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ssd_precision', type=int, choices=[32, 16], default=32, help='desired TensorRT float precision to build an engine with')
    parser.add_argument('--ssd_max_batch_size', type=int, default=1, help='max TensorRT engine batch size')
    parser.add_argument('--ssd_flatten_concat', help='path of built FlattenConcat plugin')
    parser.add_argument('--ssd_model_uff_path', help='path of ssd model folder')

    parser.add_argument('--cpn_resnet_type', type=int, default=101, choices=[50,101], help='')
    parser.add_argument('--cpn_input_size', type=str, default="384x288", choices=["384x288", "256x192"], help='')
    parser.add_argument('--cpn_model_path', type=str, help='')
    parser.add_argument('--cpn_gpu', type=str, dest='gpu_ids')
    parser.add_argument('--cpn_batch_size', type=int, default=32)
    parser.add_argument('--cpn_flip', action='store_true')

    parser.add_argument('--image_folder_or_list', type=str,  required=True, help='')
    parser.add_argument('--pose_img_save_folder', type=str, help='')
    parser.add_argument('--bbox_save_filenames', type=str, help='')
    parser.add_argument('--bbox_filenames', type=str, help='')
    parser.add_argument('--pose_bbox_save_filenames', type=str, help='')
    parser.add_argument('--use_camera', type=int, default=0, help='')
    parser.add_argument('--video', type=str, help='input video file')

    return parser.parse_args()


def make_ssd_inferece_wrapper(args):

    ctypes.CDLL(args.ssd_flatten_concat)
    trt_engine_datatype = TRT_PRECISION_TO_DATATYPE[args.ssd_precision]
    trt_engine_path = ssd_PATHS.get_engine_path(trt_engine_datatype, args.ssd_max_batch_size)
    if not os.path.exists(os.path.dirname(trt_engine_path)):
        os.makedirs(os.path.dirname(trt_engine_path))

    ssd_model_uff_path = args.ssd_model_uff_path
    ssd_max_batch_size = args.ssd_max_batch_size
    print('debug+', trt_engine_path)
    print('debug++', ssd_model_uff_path)

    if not os.path.exists(ssd_model_uff_path):
        ssd_model_utils.prepare_ssd_model(MODEL_NAME)
    
    ssd_inference_wrapper =  ssd_inference_utils.TRTInference(
        trt_engine_path, ssd_model_uff_path,
        trt_engine_datatype=trt_engine_datatype,
        batch_size=ssd_max_batch_size)

    np_img = np.zeros((300,300,3), np.uint8)
    np_img[:,:] = (255,0,0)
    prediction_fields = len(TRT_PREDICTION_LAYOUT)
    # np_img = cv2.imread("/mnt/nas/nas4storm_disk1/xna-tsukamoto/181225/Images/yoga10/18.jpg")
    input_list = ssd_inference_wrapper._load_np_imgs([np_img])
    ssd_inference_wrapper.set_input(input_list)
    detection_out , keep_count_out = ssd_inference_wrapper.infer_batch_t(len(input_list))
    for det in range(int(keep_count_out[0])):
        bbox_ = analyze_prediction(detection_out, det * prediction_fields, np_img)

    return ssd_inference_wrapper


def make_cpn_tester(args):
    cpn.set_import(args.cpn_resnet_type, args.cpn_input_size, args.cpn_batch_size, args.cpn_flip)
    tester = cpn.Tester(cpn.Network(), cpn.cfg)
    tester.load_weights(args.cpn_model_path)
    return tester


def main():
    args = parse_args()
    detect_pose_loop(args)

if __name__ == "__main__":
    main()
