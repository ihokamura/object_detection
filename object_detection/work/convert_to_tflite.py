import argparse
import os

import numpy as np
import tensorflow as tf
from PIL import Image

DATASET_DIRECTORY = 'coco_dataset/val2017'


class ModelConverter:
    def __init__(self, saved_model, model_name, quantization):
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model)
        if quantization == 'full-integer':
            converter.experimental_new_converter = True
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.representative_dataset = generate_representative_dataset
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8

        self.converter = converter
        self.model_name = model_name

    def convert(self):
        tflite_model = self.converter.convert()
        with open(f'{self.model_name}.tflite', 'wb') as f:
            f.write(tflite_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('saved_model', help='path to saved model')
    parser.add_argument('--model_name', default='model', help='model name')
    parser.add_argument('--quantization', default='none',
                        help='quantization type (none, full-integer)')
    parser.add_argument('--input_width', type=int,
                        help='width of model input (only for --quantization=full-integer)')
    parser.add_argument('--input_height', type=int,
                        help='height of model input (only for --quantization=full-integer)')
    parser.add_argument('--calibration_size', type=int, default='100',
                        help='number of calibration data (only for --quantization=full-integer)')

    args = parser.parse_args()
    saved_model = args.saved_model
    model_name = args.model_name
    quantization = args.quantization
    if quantization not in ('none', 'full-integer'):
        print(f'invalid argument for --quantization: {quantization:}')
        exit(1)
    if quantization == 'full-integer':
        input_width = args.input_width
        input_height = args.input_height
        calibration_size = args.calibration_size

    def generate_representative_dataset():
        dataset_files = np.array(os.listdir(DATASET_DIRECTORY))
        for dataset_file in np.random.choice(dataset_files, calibration_size):
            input_image = Image.open(DATASET_DIRECTORY + '/' + dataset_file)
            resized_image = input_image.resize((input_width, input_height))
            input_data = np.expand_dims(resized_image, 0).astype(np.float32)
            input_data = input_data * (2.0 / 255.0) - 1.0
            yield [input_data]

    converter = ModelConverter(saved_model, model_name, quantization)
    converter.convert()
