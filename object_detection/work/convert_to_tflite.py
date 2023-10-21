import argparse

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('saved_model', help='path to saved model')
parser.add_argument('--model_name', default='model', help='model name')

args = parser.parse_args()
model_name = args.model_name
saved_model = args.saved_model

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model)
tflite_model = converter.convert()

with open(f'{model_name}.tflite', 'wb') as f:
    f.write(tflite_model)
