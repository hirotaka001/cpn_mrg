import cv2
import copy

def draw_rect(img, bbox_):
    bbox = list(map(lambda x: int(x), bbox_))
    points = [
        (int(bbox[0]), int(bbox[1])),
        (int(bbox[0] + bbox[2]), int(bbox[1])),
        (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
        (int(bbox[0]), int(bbox[1] + bbox[3])),
    ]

    img = cv2.line(img, points[0], points[1], (0,0,255), 5)
    img = cv2.line(img, points[1], points[2], (0,0,255), 5)
    img = cv2.line(img, points[2], points[3], (0,0,255), 5)
    img = cv2.line(img, points[3], points[0], (0,0,255), 5)

def draw_line(img, cls_skeleton, from_number, to_number):
    from_ =  (int(cls_skeleton[3 * from_number]), int(cls_skeleton[3 * from_number + 1]))
    to_ = (int(cls_skeleton[3 * to_number]), int(cls_skeleton[3 * to_number + 1]))
    img = cv2.line(img,from_, to_,(255,0,0),5)

def draw_keypoint(img, cls_skeleton, bbox=None):
    draw_line(img, cls_skeleton, 0, 5)
    draw_line(img, cls_skeleton, 0, 6)
    draw_line(img, cls_skeleton, 5, 7)
    draw_line(img, cls_skeleton, 7, 9)
    draw_line(img, cls_skeleton, 5, 11)
    draw_line(img, cls_skeleton, 11, 13)
    draw_line(img, cls_skeleton, 13, 15)
    draw_line(img, cls_skeleton, 6, 8)
    draw_line(img, cls_skeleton, 8, 10)
    draw_line(img, cls_skeleton, 6, 12)
    
    draw_line(img, cls_skeleton, 12, 14)
    draw_line(img, cls_skeleton, 14, 16)

    if bbox is not None:
        draw_rect(img, bbox)