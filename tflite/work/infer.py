import argparse

import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image, ImageDraw, ImageFont


def load_mapping(category_file='coco_category.txt'):
    id_to_category = {}
    with open(category_file, 'r') as file:
        for line in file:
            id, category = line.strip('\n').split(':')
            id_to_category[int(id)] = category
    return id_to_category


def convert_image_to_input(image, input_width, input_height):
    resized_image = image.resize((input_width, input_height))
    input_data = np.array(resized_image, dtype=np.float32)[np.newaxis, ...]
    input_data = input_data * (2.0 / 255.0) - 1.0
    return input_data


def infer(interpreter, input_image):
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    input_index = input_detail['index']
    _, input_width, input_height, _ = input_detail['shape']
    input_data = convert_image_to_input(input_image, input_width, input_height)
    input_scale, input_zero_point = input_detail['quantization']
    input_data = (input_data / input_scale + input_zero_point).astype(np.uint8)
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()


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


def get_infer_result(interpreter, input_image, id_to_category, score_threshold=0.5):
    def get_adjusted_output(output_detail, dtype=np.float32):
        scale, zero_point = output_detail['quantization']
        raw_output = interpreter.get_tensor(output_detail['index'])[0]
        output = (raw_output - zero_point) * scale
        return output.astype(dtype)

    output_details = interpreter.get_output_details()
    detection_classes = np.round(get_adjusted_output(
        output_details[3])).astype(np.int32)
    detection_scores = get_adjusted_output(output_details[0])
    detection_boxes = get_adjusted_output(output_details[1])
    output_image = input_image
    draw = ImageDraw.Draw(output_image)
    image_width, image_height = output_image.size
    for detection_class, detection_score, detection_box in zip(detection_classes, detection_scores, detection_boxes):
        if detection_score >= score_threshold:
            draw_detection_result(
                draw, image_width, image_height, detection_class, detection_score, detection_box, id_to_category)
    return output_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tflite_file', help='path to tflite file')
    parser.add_argument('input_image_file', help='path to input image file')
    parser.add_argument('output_image_file', help='path to output image file')
    args = parser.parse_args()
    tflite_file = args.tflite_file
    input_image_file = args.input_image_file
    output_image_file = args.output_image_file

    interpreter = tflite.Interpreter(tflite_file)
    id_to_category = load_mapping()
    input_image = Image.open(input_image_file)
    infer(interpreter, input_image)
    output_image = get_infer_result(interpreter, input_image, id_to_category)
    output_image.save(output_image_file)
