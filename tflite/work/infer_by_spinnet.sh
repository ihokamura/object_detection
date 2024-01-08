#!/bin/bash

SPINNET_MODEL_NAME=coco_spinenet49

image_file=image1.jpg

python3 infer_by_spinnet.py \
    --quantization none \
    --score_threshold 0.5 \
    ${SPINNET_MODEL_NAME}.tflite images/input/${image_file} images/output/${image_file}
