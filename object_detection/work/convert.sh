#!/bin/bash

convert_script=convert_to_tflite.py

python3 $convert_script \
    --model_name $SSD_MODEL_NAME \
    --quantization full-integer \
    --input_width $INPUT_WIDTH \
    --input_height $INPUT_HEIGHT \
    --calibration_size $CALIBRATION_SIZE \
    $SSD_MODEL_NAME/exported_model/saved_model
