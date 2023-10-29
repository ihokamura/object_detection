#!/bin/bash

export_script=/home/tensorflow/models/research/object_detection/export_tflite_graph_tf2.py

python3 $export_script \
    --pipeline_config_path $SSD_MODEL_NAME/pipeline.config \
    --trained_checkpoint_dir $SSD_MODEL_NAME/checkpoint \
    --output_directory $SSD_MODEL_NAME/exported_model \
    --max_detections 10 \
    --ssd_use_regular_nms
