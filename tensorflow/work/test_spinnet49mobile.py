import matplotlib.pyplot as plt
import numpy as np
from official.core import exp_factory
from official.vision import registry_imports
from official.vision.modeling import factory
from official.vision.ops import preprocess_ops
from official.vision.utils.object_detection import visualization_utils

import tensorflow as tf

height, width = 384, 384
category_index = {
    1: {'id': 1, 'name': 'person'},
    2: {'id': 2, 'name': 'bicycle'},
    3: {'id': 3, 'name': 'car'},
    4: {'id': 4, 'name': 'motorcycle'},
    5: {'id': 5, 'name': 'airplane'},
    6: {'id': 6, 'name': 'bus'},
    7: {'id': 7, 'name': 'train'},
    8: {'id': 8, 'name': 'truck'},
    9: {'id': 9, 'name': 'boat'},
    10: {'id': 10, 'name': 'traffic light'},
    11: {'id': 11, 'name': 'fire hydrant'},
    12: {'id': 12, 'name': 'street sign'},
    13: {'id': 13, 'name': 'stop sign'},
    14: {'id': 14, 'name': 'parking meter'},
    15: {'id': 15, 'name': 'bench'},
    16: {'id': 16, 'name': 'bird'},
    17: {'id': 17, 'name': 'cat'},
    18: {'id': 18, 'name': 'dog'},
    19: {'id': 19, 'name': 'horse'},
    20: {'id': 20, 'name': 'sheep'},
    21: {'id': 21, 'name': 'cow'},
    22: {'id': 22, 'name': 'elephant'},
    23: {'id': 23, 'name': 'bear'},
    24: {'id': 24, 'name': 'zebra'},
    25: {'id': 25, 'name': 'giraffe'},
    26: {'id': 26, 'name': 'hat'},
    27: {'id': 27, 'name': 'backpack'},
    28: {'id': 28, 'name': 'umbrella'},
    29: {'id': 29, 'name': 'shoe'},
    30: {'id': 30, 'name': 'eye glasses'},
    31: {'id': 31, 'name': 'handbag'},
    32: {'id': 32, 'name': 'tie'},
    33: {'id': 33, 'name': 'suitcase'},
    34: {'id': 34, 'name': 'frisbee'},
    35: {'id': 35, 'name': 'skis'},
    36: {'id': 36, 'name': 'snowboard'},
    37: {'id': 37, 'name': 'sports ball'},
    38: {'id': 38, 'name': 'kite'},
    39: {'id': 39, 'name': 'baseball bat'},
    40: {'id': 40, 'name': 'baseball glove'},
    41: {'id': 41, 'name': 'skateboard'},
    42: {'id': 42, 'name': 'surfboard'},
    43: {'id': 43, 'name': 'tennis racket'},
    44: {'id': 44, 'name': 'bottle'},
    45: {'id': 45, 'name': 'plate'},
    46: {'id': 46, 'name': 'wine glass'},
    47: {'id': 47, 'name': 'cup'},
    48: {'id': 48, 'name': 'fork'},
    49: {'id': 49, 'name': 'knife'},
    50: {'id': 50, 'name': 'spoon'},
    51: {'id': 51, 'name': 'bowl'},
    52: {'id': 52, 'name': 'banana'},
    53: {'id': 53, 'name': 'apple'},
    54: {'id': 54, 'name': 'sandwich'},
    55: {'id': 55, 'name': 'orange'},
    56: {'id': 56, 'name': 'broccoli'},
    57: {'id': 57, 'name': 'carrot'},
    58: {'id': 58, 'name': 'hot dog'},
    59: {'id': 59, 'name': 'pizza'},
    60: {'id': 60, 'name': 'donut'},
    61: {'id': 61, 'name': 'cake'},
    62: {'id': 62, 'name': 'chair'},
    63: {'id': 63, 'name': 'couch'},
    64: {'id': 64, 'name': 'potted plant'},
    65: {'id': 65, 'name': 'bed'},
    66: {'id': 66, 'name': 'mirror'},
    67: {'id': 67, 'name': 'dining table'},
    68: {'id': 68, 'name': 'window'},
    69: {'id': 69, 'name': 'desk'},
    70: {'id': 70, 'name': 'toilet'},
    71: {'id': 71, 'name': 'door'},
    72: {'id': 72, 'name': 'tv'},
    73: {'id': 73, 'name': 'laptop'},
    74: {'id': 74, 'name': 'mouse'},
    75: {'id': 75, 'name': 'remote'},
    76: {'id': 76, 'name': 'keyboard'},
    77: {'id': 77, 'name': 'cell phone'},
    78: {'id': 78, 'name': 'microwave'},
    79: {'id': 79, 'name': 'oven'},
    80: {'id': 80, 'name': 'toaster'},
    81: {'id': 81, 'name': 'sink'},
    82: {'id': 82, 'name': 'refrigerator'},
    83: {'id': 83, 'name': 'blender'},
    84: {'id': 84, 'name': 'book'},
    85: {'id': 85, 'name': 'clock'},
    86: {'id': 86, 'name': 'vase'},
    87: {'id': 87, 'name': 'scissors'},
    88: {'id': 88, 'name': 'teddy bear'},
    89: {'id': 89, 'name': 'hair drier'},
    90: {'id': 90, 'name': 'toothbrush'},
    91: {'id': 91, 'name': 'hair brush'},
}

print('start build')
exp_config = exp_factory.get_exp_config('retinanet_mobile_coco')
model_config = exp_config.task.model
model_config.norm_activation.activation = 'hard_swish'
input_specs = tf.keras.layers.InputSpec(shape=[None, height, width, 3])
model = factory.build_retinanet(
    input_specs=input_specs, model_config=model_config)
model.build([None, height, width, 3])
print('end build')

print('start load')
checkpoint = tf.train.Checkpoint(model)
checkpoint.restore(tf.train.latest_checkpoint('coco_spinenet49_mobile'))
print('end load')
# model.summary()

image = tf.keras.utils.load_img('000000000139.jpg', target_size=(height, width))
image_array = tf.keras.utils.img_to_array(image, dtype=int)
input_data = preprocess_ops.normalize_image(image_array)
input_data = tf.expand_dims(input_data, axis=0)

print('start inference')
result = model(input_data, training=False)
print('end inference')

print(result)
visualization_utils.visualize_boxes_and_labels_on_image_array(
    image_array,
    result['detection_boxes'][0].numpy(),
    result['detection_classes'][0].numpy().astype(int),
    result['detection_scores'][0].numpy(),
    category_index=category_index,
    use_normalized_coordinates=False,
    max_boxes_to_draw=200,
    min_score_thresh=0.3,
    agnostic_mode=False,
    instance_masks=None,
    line_thickness=4)
plt.imshow(image_array)
plt.savefig('result.png')
