#!/bin/bash

base_url=http://download.tensorflow.org/models/object_detection/tf2/20200711

wget "${base_url}/${SSD_MODEL_NAME}.tar.gz"
gzip -d ${SSD_MODEL_NAME}.tar.gz
tar -xf ${SSD_MODEL_NAME}.tar
rm ${SSD_MODEL_NAME}.tar
