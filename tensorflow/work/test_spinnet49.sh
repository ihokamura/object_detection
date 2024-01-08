#!/bin/bash

git clone https://github.com/tensorflow/models.git
cd models
git checkout 24ae1f51df065e38e89c3bda5e4a53448f40b426

cat <<EOF > ./override_param.yaml
task:
  export_config:
    cast_detection_classes_to_float: true
    cast_num_detections_to_float: true
    output_normalized_coordinates: true
  model:
    detection_generator:
      apply_nms: true
      tflite_post_processing:
        max_classes_per_detection: 5
        max_detections: 200
        nms_iou_threshold: 0.5
        nms_score_threshold: 0.1
        use_regular_nms: false
      use_cpu_nms: true
EOF

python3 official/vision/serving/export_saved_model.py \
    --experiment=retinanet_spinenet_coco \
    --export_dir=./coco_spinenet49 \
    --checkpoint_path=../coco_spinenet49/ckpt-231000 \
    --config_file=official/vision/configs/experiments/retinanet/coco_spinenet49_tpu.yaml \
    --params_override=./override_param.yaml \
    --batch_size=1 \
    --input_image_size=640,640 \
    --input_type=tflite \
    --log_model_flops_and_params

python3 official/vision/serving/export_tflite.py \
    --experiment=retinanet_spinenet_coco \
    --saved_model_dir=./coco_spinenet49/saved_model \
    --config_file=./coco_spinenet49/params.yaml \
    --tflite_path=./coco_spinenet49/coco_spinenet49.tflite
