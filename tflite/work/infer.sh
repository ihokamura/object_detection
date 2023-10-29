#!/bin/bash

ssd_model_name=ssd_mobilenet_v2_320x320_coco17_tpu-8
image_file=image1.jpg

python3 infer.py \
    --quantization full-integer \
    --score_threshold 0.5 \
    ${ssd_model_name}.tflite images/input/${image_file} images/output/${image_file}
