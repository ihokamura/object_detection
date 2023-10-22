#!/bin/bash

ssd_model_name=ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8
image_file=image1.jpg

python3 infer.py ${ssd_model_name}.tflite images/input/${image_file} images/output/${image_file}