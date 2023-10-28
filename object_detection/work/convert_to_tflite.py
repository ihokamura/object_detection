import argparse
import os

import numpy as np
import tensorflow as tf
from PIL import Image

DATASET_DIRECTORY = 'coco_dataset/val2017'

parser = argparse.ArgumentParser()
parser.add_argument('saved_model', help='path to saved model')
parser.add_argument('--model_name', default='model', help='model name')
parser.add_argument('--input_width', type=int, help='width of model input')
parser.add_argument('--input_height', type=int, help='height of model input')
parser.add_argument('--calibration_size', type=int, default='100',
                    help='number of calibration data')
args = parser.parse_args()
saved_model = args.saved_model
model_name = args.model_name
input_width = args.input_width
input_height = args.input_height
calibration_size = args.calibration_size


def generate_representative_dataset():
    dataset_files = np.array(os.listdir(DATASET_DIRECTORY))
    for dataset_file in np.random.choice(dataset_files, calibration_size):
        input_image = Image.open(os.path.join(DATASET_DIRECTORY, dataset_file))
        resized_image = input_image.resize((input_width, input_height))
        input_data = np.array(resized_image, dtype=np.float32)[np.newaxis, ...]
        input_data = input_data * (2.0 / 255.0) - 1.0
        yield [input_data]


converter = tf.lite.TFLiteConverter.from_saved_model(saved_model)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = generate_representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

with open(f'{model_name}.tflite', 'wb') as f:
    f.write(tflite_model)
