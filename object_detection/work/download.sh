#!/bin/bash

base_url=http://download.tensorflow.org/models/object_detection/tf2/20200711
ssd_model_name=ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8

wget "${base_url}/${ssd_model_name}.tar.gz"
gzip -d ${ssd_model_name}.tar.gz
tar -xf ${ssd_model_name}.tar
rm ${ssd_model_name}.tar
