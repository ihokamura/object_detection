#!/bin/bash

export_script=/home/tensorflow/models/research/object_detection/export_tflite_graph_tf2.py
ssd_model_directory=ssd_mobilenet_v2_320x320_coco17_tpu-8

python3 $export_script \
    --pipeline_config_path $ssd_model_directory/pipeline.config \
    --trained_checkpoint_dir $ssd_model_directory/checkpoint \
    --output_directory $ssd_model_directory/exported_model \
    --max_detections 10 \
    --ssd_use_regular_nms
