#!/bin/bash

convert_script=convert_to_tflite.py
ssd_model_directory=ssd_mobilenet_v2_320x320_coco17_tpu-8

python3 $convert_script \
    --model_name $ssd_model_directory \
    --input_width 300 \
    --input_height 300 \
    --calibration_size 100 \
    $ssd_model_directory/exported_model/saved_model
