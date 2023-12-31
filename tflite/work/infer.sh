#!/bin/bash

SSD_MODEL_NAME=ssd_mobilenet_v2_320x320_coco17_tpu-8
#SSD_MODEL_NAME=ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8
#SSD_MODEL_NAME=ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8
#SSD_MODEL_NAME=ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8

image_file=image1.jpg

python3 infer.py \
    --quantization full-integer \
    --score_threshold 0.5 \
    ${SSD_MODEL_NAME}.tflite images/input/${image_file} images/output/${image_file}
