#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

# main script for testing output from cutted Tiny Yolo v2

import tensorflow as tf
import PIL
from PIL import Image
import numpy as np
import cv2
import copy
from aug import test_img_adjust
from utils import decode_netout, draw_boxes
# from functions_for_yolo2 import restore_bbox_coords, convert_yolo_output
import pdb
import sys


def restore_bbox_coords(bbox, orig_width, orig_height):
    netw = 416
    neth = 416
    scale = min(float(netw) / orig_width, float(neth) / orig_height)
    new_width = orig_width * scale
    new_height = orig_height * scale
    pad_w = (netw - new_width) / 2.0
    pad_h = (neth - new_height) / 2.0
    for v in bbox:
        v.xmin = max(0, float(v.xmin * netw - pad_w) / scale)
        v.xmax = min(orig_width - 1, float(v.xmax * netw - pad_w) / scale)
        v.ymin = max(0, float(v.ymin * neth - pad_h) / scale)
        v.ymax = min(orig_height - 1, float(v.ymax * neth - pad_h) / scale)

    return bbox


def convert_yolo_output(x):
    # x = (x.astype(np.int) - 221) * 0.3371347486972809

    x = x.reshape(1, 13, 13, 128)
    print("after 1x13x13x128 :  ")
    print(x.shape)
    print("\n")

    # print("This is the output tensor shape: {}".format(x.shape))
    x = x[..., 0:125]
    print("This is the output tensor shape after removing the last 3 : {}".format(x.shape))
    x = (x.astype(np.int) - 221) * 0.3371347486972809
    print("\n")
    x = x.reshape(grid_h, grid_w, nb_box, -1)
    return x


def print_box_info(boxes):
    if len(boxes) == 0:
        print("There are not any bounding boxes")
    else:
        print("Bounding boxes:")
        i = 0
        for box in boxes:
            print("Bounding box #" + str(i))
            i += 1
            print("Object : " + labels[box.get_label()])
            print("Probability = " + str(box.get_score()))
            print("Coordinates (x_min, y_min - x_max, y_max) : " + str(int(box.xmin)) + ", " + str(int(box.ymin)) + " - " +
                  str(int(box.xmax)) + ", " + str(int(box.ymax)))
            print()
    return  # end of function print_box_info( )

# Start of main script


if len(sys.argv) == 1:
    print("Too small arguments! Usage: python3 main.py <imageFileName> <probabilityForObjectDetection>=0.4")
    quit()
elif len(sys.argv) == 2:
    imageFileName = sys.argv[1]
    probabilityForObjectDetection = 0.4
elif len(sys.argv) == 3:
    imageFileName = sys.argv[1]
    probabilityForObjectDetection = float(sys.argv[2])

grid_h = 13
grid_w = 13
nb_box = 5
nb_class = 20
anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
          "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# this Tiny Yolo v2 has not postprocessing layer (aka region)
model_path = "tiny-yolov2-128output.tflite"
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details[0]["name"], input_details[0]["shape"])
print(output_details[0]["name"], output_details[0]["shape"])


input_img = Image.open(imageFileName)
img_row = input_details[0]["shape"][1]
img_column = input_details[0]["shape"][2]
input_img = input_img.resize((img_row, img_column))
input_img = np.expand_dims(np.array(input_img, input_details[0]["dtype"]), axis=0)

interpreter.set_tensor(input_details[0]['index'], input_img)

interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])

# call function that dequantize and reshape output-0.bin
x = convert_yolo_output(output)
# determining the labels that have a confidence score higher than value of probabilityForObjectDetection
# and suppressing boxes that overlap and have the same label
boxes = decode_netout(x, anchors, nb_class, probabilityForObjectDetection, 0.45)

image = cv2.imread(imageFileName)
image_h, image_w, _ = image.shape
boxes = restore_bbox_coords(boxes, image_w, image_h)
print_box_info(boxes)
image = draw_boxes(image, boxes, labels)
outputImageFileName = imageFileName[:-4] + "_result." + imageFileName[-3:]
cv2.imwrite(outputImageFileName, image)
