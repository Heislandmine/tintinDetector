import argparse
import pathlib
import os
from PIL import ImageDraw, ImageFont

from tintin_detector.detector import Detector
from tintin_detector.utils import read_image

parser = argparse.ArgumentParser(description="tintin detector demo")
parser.add_argument("--input", default="./data")
parser.add_argument("--output", default="./results/")
parser.add_argument("--th", default=0.5)

args = parser.parse_args()

model_path = "./model/model_v4_003.tflite"

font_size = 40
font = ImageFont.truetype("./font/IPAMTTC00303/ipam.ttc", size=font_size)

detector = Detector(model_path)

for image_path in pathlib.Path(args.input).glob("*.*"):
    file_name = image_path.name
    input_image = read_image(image_path)
    boxes, scores = detector.run(input_image, args.th)

    draw = ImageDraw.ImageDraw(input_image)

    for box, score in zip(boxes, scores):
        ymin, xmin, ymax, xmax = box

        ymin = input_image.height * ymin
        xmin = input_image.width * xmin
        ymax = input_image.height * ymax
        xmax = input_image.width * xmax

        score = round(score * 100, 1)

        draw.rectangle(((xmin, ymin), (xmax, ymax)), width=10)
        draw.text(
            (xmin, ymin - font_size),
            f"tintin({str(score)}%)",
            fill=(0, 0, 0),
            font=font,
            stroke_width=2,
        )

        if not os.path.exists(args.output):
            os.makedirs(args.output)

    input_image.save(args.output + "/" + file_name)
