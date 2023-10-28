import argparse
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('saved_model', help='path to saved model')
parser.add_argument('--model_name', default='model', help='model name')

args = parser.parse_args()
model_name = args.model_name
saved_model = args.saved_model

dataset_directory = '/work/coco_dataset/val2017'
number_of_calibration_data = 100
shuffle = True
input_width, input_height = (300, 300)


def representative_dataset_gen():
    dataset_files = np.array(os.listdir(dataset_directory))
    if shuffle:
        np.random.shuffle(dataset_files)
    dataset_files = dataset_files[:number_of_calibration_data]

    for dataset_file in dataset_files:
        input_image = Image.open(os.path.join(dataset_directory, dataset_file))
        resized_image = input_image.resize((input_width, input_height))
        input_data = np.array(resized_image, dtype=np.float32)[np.newaxis, ...]
        input_data = input_data * (2.0 / 255.0) - 1.0
        yield [input_data]


converter = tf.lite.TFLiteConverter.from_saved_model(saved_model)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
tflite_model = converter.convert()

with open(f'{model_name}.tflite', 'wb') as f:
    f.write(tflite_model)
