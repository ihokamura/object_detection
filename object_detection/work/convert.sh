#!/bin/bash

convert_script=convert_to_tflite.py
ssd_model_directory=ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8

python3 $convert_script $ssd_model_directory/exported_model/saved_model --model_name $ssd_model_directory
