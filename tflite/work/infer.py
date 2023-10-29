import argparse

import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image, ImageDraw, ImageFont


def load_mapping(category_file):
    id_to_category = {}
    with open(category_file, 'r') as file:
        for line in file:
            id, category = line.strip('\n').split(':')
            id_to_category[int(id)] = category
    return id_to_category


def convert_image_to_input(image, input_width, input_height):
    resized_image = image.resize((input_width, input_height))
    input_data = np.expand_dims(resized_image, 0).astype(np.float32)
    input_data = input_data * (2.0 / 255.0) - 1.0
    return input_data


def draw_box(draw, xmin, ymin, xmax, ymax, box_color=(255, 255, 255), box_width=5):
    draw.rectangle((xmin, ymin, xmax, ymax), fill=None,
                   outline=box_color, width=box_width)


def draw_label(draw, xmin, ymin, label, box_color=(255, 255, 255), font_color=(0, 0, 0), font_name='DejaVuSerif.ttf', font_size=16):
    font = ImageFont.truetype(font_name, font_size)
    bbox = font.getbbox(label)
    text_width, text_height = bbox[2], bbox[3]
    width_margin = np.ceil(0.1 * text_width)
    height_margin = np.ceil(0.1 * text_height)
    draw.rectangle((xmin, ymin, xmin + text_width + 2*width_margin, ymin +
                   text_height + 2*height_margin), fill=box_color, outline=box_color)
    draw.text((xmin + width_margin, ymin + height_margin),
              label, fill=font_color, font=font)


def draw_detection_result(draw, image_width, image_height, detection_class, detection_score, detection_box, id_to_category):
    ymin, xmin, ymax, xmax = detection_box
    if xmin < xmax and ymin < ymax:
        ymin = int(ymin * image_height)
        xmin = int(xmin * image_width)
        ymax = int(ymax * image_height)
        xmax = int(xmax * image_width)

        box_color = 'red'
        font_color = 'black'
        category = id_to_category[int(detection_class) + 1]
        label = category + f':{detection_score:.2f}'
        draw_box(draw, xmin, ymin, xmax, ymax, box_color)
        draw_label(draw, xmin, ymin, label, box_color, font_color)


def quantize(float_data, scale, zero_point, dtype):
    return (float_data / scale + zero_point).astype(dtype)


def dequantize(int_data, scale, zero_point, dtype):
    return ((int_data - zero_point) * scale).astype(dtype)


class Detector:
    def __init__(self, tflite_file, quantization, category_file='coco_category.txt'):
        self.interpreter = tflite.Interpreter(tflite_file)
        self.quantization = quantization
        self.id_to_category = load_mapping(category_file)

    def infer(self, input_image):
        self.interpreter.allocate_tensors()
        input_detail = self.interpreter.get_input_details()[0]
        input_index = input_detail['index']
        _, input_width, input_height, _ = input_detail['shape']
        input_data = convert_image_to_input(
            input_image, input_width, input_height)
        if self.quantization == 'full-integer':
            scale, zero_point = input_detail['quantization']
            input_data = quantize(input_data, scale, zero_point, np.uint8)
        self.interpreter.set_tensor(input_index, input_data)
        self.interpreter.invoke()

    def get_output_image(self, input_image, score_threshold=0.5):
        def get_output_data(output_detail):
            output_data = self.interpreter.get_tensor(
                output_detail['index'])[0]
            if self.quantization == 'full-integer':
                scale, zero_point = output_detail['quantization']
                output_data = dequantize(
                    output_data, scale, zero_point, np.float32)
            return output_data

        output_details = self.interpreter.get_output_details()
        detection_classes = np.round(get_output_data(
            output_details[3])).astype(np.int32)
        detection_scores = get_output_data(output_details[0])
        detection_boxes = get_output_data(output_details[1])

        output_image = input_image
        draw = ImageDraw.Draw(output_image)
        image_width, image_height = output_image.size
        for detection_class, detection_score, detection_box in zip(detection_classes, detection_scores, detection_boxes):
            if detection_score >= score_threshold:
                draw_detection_result(
                    draw, image_width, image_height, detection_class, detection_score, detection_box, self.id_to_category)
        return output_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tflite_file', help='path to tflite file')
    parser.add_argument('input_image_file', help='path to input image file')
    parser.add_argument('output_image_file', help='path to output image file')
    parser.add_argument('--quantization', default='none',
                        help='quantization type (none, full-integer)')
    parser.add_argument('--score_threshold', default=0.5, type=float,
                        help='threshold of score to show inference box')

    args = parser.parse_args()
    tflite_file = args.tflite_file
    input_image_file = args.input_image_file
    output_image_file = args.output_image_file
    quantization = args.quantization
    score_threshold = args.score_threshold
    if quantization not in ('none', 'full-integer'):
        print(f'invalid argument for --quantization: {quantization:}')
        exit(1)

    detector = Detector(tflite_file, quantization)
    input_image = Image.open(input_image_file)
    detector.infer(input_image)
    output_image = detector.get_output_image(input_image, score_threshold)
    output_image.save(output_image_file)
